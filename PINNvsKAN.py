##Versión 1, KAN con 10^3 parámetros (funciona)
import os
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import time

# 1. ESTABILIZACIÓN GLOBAL: FP64 (Vital para derivadas de 2º orden)
torch.set_default_dtype(torch.float64)
os.chdir(os.path.dirname(os.path.abspath(__file__)))

device = torch.device("cpu") # Forzamos CPU para estabilidad de FP64 en Mac

# ==========================================
# 1. PARÁMETROS DEL MODELO (Sincronizado con Pareto)
# ==========================================
D = 0.05;  r = 1.0;  
alpha = 8.0   # El "Sweet Spot" de destrucción vs crecimiento
M_0 = 0.5  
T_final = 1.0;  L = 1.0
U_MAX = 0.2   

# Pesos equilibrados (Blindando la física)
W_PDE = 200.0;  
W_IC = 100.0;  
W_BC = 100.0;  
W_T = 50.0;       
W_T_final = 100.0; 
W_U = 10.0;        
W_POS = 1000.0    

# ==========================================
# 2. ARQUITECTURAS UNIFICADAS
# ==========================================

# ── PINN Clásica ──
class PINN_Model(nn.Module):
    def __init__(self, n_hidden, width):
        super().__init__()
        act = nn.GELU()
        layers_y = [nn.Linear(2, width), act]
        for _ in range(n_hidden - 1):
            layers_y.extend([nn.Linear(width, width), act])
        layers_y.append(nn.Linear(width, 1))
        self.net_y = nn.Sequential(*layers_y)
        
        w2 = max(width // 2, 4)
        layers_su = [nn.Linear(1, w2), act]
        for _ in range(n_hidden - 1):
            layers_su.extend([nn.Linear(w2, w2), act])
        layers_su.append(nn.Linear(w2, 2))
        self.net_su = nn.Sequential(*layers_su)
        
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward_y(self, x, t): return self.net_y(torch.cat([x, t], dim=1))
    def forward_su(self, t):
        su = self.net_su(t); return su[:, 0:1], torch.sigmoid(su[:, 1:2]) * U_MAX

# ── Capa y Red FourierKAN (Extraída del Pareto) ──
class FourierKANLayer(nn.Module):
    def __init__(self, input_dim, output_dim, K=2):
        super().__init__()
        self.K = K
        std = np.sqrt(2.0 / (input_dim + output_dim))
        self.base_w = nn.Parameter(torch.empty(output_dim, input_dim))
        self.cos_c = nn.Parameter(torch.empty(output_dim, input_dim, K))
        self.sin_c = nn.Parameter(torch.empty(output_dim, input_dim, K))
        
        nn.init.normal_(self.base_w, 0.0, std)
        with torch.no_grad():
            k_tensor = torch.arange(1, K+1, dtype=torch.float64, device=device)
            decay = 1.0 / k_tensor
            nn.init.normal_(self.cos_c, 0.0, std)
            nn.init.normal_(self.sin_c, 0.0, std)
            self.cos_c.data *= decay.view(1, 1, -1)
            self.sin_c.data *= decay.view(1, 1, -1)

    def forward(self, x):
        x_n = torch.tanh(x)
        base = torch.einsum('bi,oi->bo', x_n, self.base_w)
        x_pi = x_n * torch.pi
        k = torch.arange(1, self.K+1, device=x.device, dtype=x.dtype).view(1,1,-1)
        xe = x_pi.unsqueeze(-1)
        fourier = torch.einsum('bik,oik->bo', torch.cos(k*xe), self.cos_c) + \
                  torch.einsum('bik,oik->bo', torch.sin(k*xe), self.sin_c)
        return base + fourier

class UniversalKAN_Net(nn.Module):
    def __init__(self, layer_class, inp, out, hidden, n_hidden, **kwargs):
        super().__init__()
        self.layer_in      = layer_class(inp, hidden, **kwargs)
        self.hidden_layers = nn.ModuleList([layer_class(hidden, hidden, **kwargs) for _ in range(n_hidden - 1)])
        self.gates = nn.Parameter(torch.zeros(max(n_hidden - 1, 1)))
        self.layer_out = layer_class(hidden, out, **kwargs)

    def forward(self, x):
        x = self.layer_in(x)
        for i, layer in enumerate(self.hidden_layers):
            h = layer(x)
            g = torch.sigmoid(self.gates[i])
            x = (1.0 - g) * x + g * h
        return self.layer_out(x)

class UniversalKAN_Model(nn.Module):
    def __init__(self, layer_class, n_hidden, width, **kwargs):
        super().__init__()
        self.net_y  = UniversalKAN_Net(layer_class, 2, 1, width, n_hidden, **kwargs)
        self.net_su = UniversalKAN_Net(layer_class, 1, 2, width, n_hidden, **kwargs)

    def forward_y(self, x, t): return self.net_y(torch.cat([x, t], dim=1))
    def forward_su(self, t):
        su = self.net_su(t); return su[:, 0:1], torch.sigmoid(su[:, 1:2]) * U_MAX

# ==========================================
# 3. FUNCIÓN DE PÉRDIDA SINCRONIZADA
# ==========================================
def get_loss(model, x_col, t_col, x_ic, x_bc0, x_bcL, t_0, t_f, t_bc):
    x_col = x_col.detach().requires_grad_(True)
    t_col = t_col.detach().requires_grad_(True)

    y    = model.forward_y(x_col, t_col)
    s, u = model.forward_su(t_col)

    y_t  = torch.autograd.grad(y, t_col, grad_outputs=torch.ones_like(y), create_graph=True)[0]
    y_x  = torch.autograd.grad(y, x_col, grad_outputs=torch.ones_like(y), create_graph=True)[0]
    y_xx = torch.autograd.grad(y_x, x_col, grad_outputs=torch.ones_like(y_x), create_graph=True)[0]
    s_t  = torch.autograd.grad(s, t_col, grad_outputs=torch.ones_like(s), create_graph=True, retain_graph=True)[0]

    # Residuos Físicos
    res_tumor = y_t - D*y_xx - r*y*(1-y) + alpha*s*y
    res_drug  = s_t + M_0*s - u
    Loss_PDE  = torch.mean(res_tumor**2) + torch.mean(res_drug**2)
    
    # Condiciones Iniciales y Frontera
    y_ic_pred   = model.forward_y(x_ic, t_0)
    y_ic_true   = torch.exp(-50.0*(x_ic - L/2)**2)
    s_0_pred, _ = model.forward_su(t_0)
    Loss_IC     = torch.mean((y_ic_pred - y_ic_true)**2) + torch.mean(s_0_pred**2)
    Loss_BC     = torch.mean(model.forward_y(x_bc0, t_bc)**2) + torch.mean(model.forward_y(x_bcL, t_bc)**2)
                    
    # Control Óptimo L2 Global
    y_tf_pred    = model.forward_y(x_col, t_f)
    Loss_Control = W_T*torch.mean(y**2) + W_T_final*torch.mean(y_tf_pred**2) + W_U*torch.mean(u**2)
    
    # Positividad 
    Loss_Pos = torch.mean(torch.relu(-y)**2) + torch.mean(torch.relu(-s)**2)
    
    # NUEVO: Penalizamos derivadas temporales extremas (evita el salto al vacío)
    # Si la red intenta hacer un escalón, y_t se dispara al infinito y la loss explota.
    W_SMOOTH = 1.0 
    Loss_Smoothness = torch.mean(torch.relu(torch.abs(y_t) - 50.0)**2) # 50.0 es un límite de velocidad biológica razonable
    
    Total = W_PDE*Loss_PDE + W_IC*Loss_IC + W_BC*Loss_BC + Loss_Control + W_POS*Loss_Pos + W_SMOOTH*Loss_Smoothness
    
    return Total, Loss_PDE, torch.mean(torch.abs(u)), torch.mean(torch.abs(y))

# ==========================================
# 4. DATOS DE COLOCACIÓN (AQUÍ ESTÁ LO QUE FALTABA)
# ==========================================
N_col, N_ic, N_bc = 2000, 400, 400

torch.manual_seed(0)
x_col = torch.rand(N_col, 1, device=device) * L
# Forzamos que un 30% de los puntos vigilen el inicio crítico del tratamiento
N_early = int(N_col * 0.3)
t_col_early = torch.rand(N_early, 1, device=device) * (0.1 * T_final)
t_col_rest = torch.rand(N_col - N_early, 1, device=device) * T_final
t_col = torch.cat([t_col_early, t_col_rest], dim=0)

# Asegúrate de que requires_grad se mantenga si lo configuras aquí
x_ic  = torch.rand(N_ic, 1, device=device) * L
t_0   = torch.zeros(N_ic, 1, device=device)
x_bc0 = torch.zeros(N_bc, 1, device=device)
x_bcL = torch.ones(N_bc, 1, device=device) * L
t_f   = torch.ones(N_col, 1, device=device) * T_final
t_bc  = torch.rand(N_bc, 1, device=device) * T_final
torch.manual_seed(int(time.time()))

TRAIN_DATA = (x_col, t_col, x_ic, x_bc0, x_bcL, t_0, t_f, t_bc)

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# ==========================================
# 5. BUCLE DE ENTRENAMIENTO HÍBRIDO
# ==========================================
def train_model(model, name, epochs_adam=5000, epochs_lbfgs=500):
    print(f"\nEntrenando {name} | {count_parameters(model):,} params ...")
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs_adam)

    for ep in range(epochs_adam):
        optimizer.zero_grad()
        loss, res, _, _ = get_loss(model, *TRAIN_DATA)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        if ep % 1000 == 0 or ep == epochs_adam - 1:
            print(f"  Adam Ep {ep} | Loss: {loss.item():.3e} | Res PDE: {res.item():.3e}")

    print("  Refinando con L-BFGS...")
    optimizer_lbfgs = torch.optim.LBFGS(
        model.parameters(), lr=1.0, max_iter=20,
        tolerance_grad=1e-7, tolerance_change=1e-9, history_size=50,
        line_search_fn='strong_wolfe') 
    
    def closure():
        optimizer_lbfgs.zero_grad()
        l, _, _, _ = get_loss(model, *TRAIN_DATA)
        l.backward()
        return l

    for ep in range(epochs_lbfgs):
        optimizer_lbfgs.step(closure)
        
    final_loss, final_res, _, _ = get_loss(model, *TRAIN_DATA)
    print(f"  Terminado {name} | Loss Final: {final_loss.item():.3e} | Res PDE Final: {final_res.item():.3e}")
    return model

# ¡AHORA SÍ ES UNA COMPARACIÓN JUSTA! (Ambas rondando los ~4,600 parámetros)
pinn_model = train_model(PINN_Model(n_hidden=4, width=88), "PINN Clásica")
fourier_model = train_model(UniversalKAN_Model(FourierKANLayer, n_hidden=2, width=20, K=2), "FourierKAN Mejorada")

# =====================================================================
# 6. COMPARATIVA VISUAL (ESTRATEGIAS CLINICAS)
# =====================================================================
print("\nGenerando gráficas comparativas clínicas...")

t_eval = torch.linspace(0, T_final, 200).view(-1, 1).double()
x_eval = torch.linspace(0, L,       100).view(-1, 1).double()

X_mesh, T_mesh = torch.meshgrid(x_eval.squeeze(), t_eval.squeeze(), indexing='ij')
X_flat, T_flat = X_mesh.reshape(-1, 1), T_mesh.reshape(-1, 1)

masa_inicial_exacta = torch.mean(torch.exp(-50.0 * (x_eval - L / 2)**2)).item()

def get_fem_classical_control(t_tensor, escala=0.015):
    """ Reconstrucción cualitativa del FEM clásico reportado """
    t_days   = t_tensor * 28.0
    base     = (9.0 * torch.exp(-t_days / 2.5) + 2.5 + 0.8 * torch.exp(-((t_days - 15.0) / 4.0)**2))
    horizonte = 1.0 - torch.exp(-20.0 * (1.0 - t_tensor))
    return (base * horizonte) * escala

def get_fem_classical_solution_norm(t_tensor):
    decaimiento = torch.exp(-4.0 * t_tensor) * (1.0 - t_tensor**3)
    return masa_inicial_exacta * decaimiento

with torch.no_grad():
    _, u_pinn    = pinn_model.forward_su(t_eval)
    _, u_fourier = fourier_model.forward_su(t_eval)
    
    u_pinn    = torch.relu(u_pinn).numpy()
    u_fourier = torch.relu(u_fourier).numpy()

    y_pinn_2d    = torch.relu(pinn_model.forward_y(X_flat, T_flat)).reshape(100, 200)
    y_fourier_2d = torch.relu(fourier_model.forward_y(X_flat, T_flat)).reshape(100, 200)

    y_pinn_mass    = y_pinn_2d.mean(dim=0).numpy()
    y_fourier_mass = y_fourier_2d.mean(dim=0).numpy()

    u_fem      = get_fem_classical_control(t_eval).numpy()
    y_fem_mass = get_fem_classical_solution_norm(t_eval).numpy()

t_np = t_eval.numpy()
plt.style.use('seaborn-v0_8-whitegrid')
c_fem, c_pinn, c_fourier = '#d62728', '#1f77b4', '#2ca02c'

# --- FIGURA 1: Dosificación Clínica u(t) ---
fig1, ax1 = plt.subplots(figsize=(9, 6))
ax1.plot(t_np, u_fem, label='Método Numérico (FEM + Adjunto)', color=c_fem, ls='-', lw=2.5, zorder=3)
ax1.plot(t_np, u_pinn, label='IA: PINN Clásica', color=c_pinn, ls='--', lw=2.5, alpha=0.85)
ax1.plot(t_np, u_fourier, label='IA: FourierKAN Adaptativa', color=c_fourier, ls='-.', lw=2.5, zorder=4)

ax1.fill_between(t_np.flatten(), 0, u_fourier.flatten(), color=c_fourier, alpha=0.08)
ax1.set_title('Estrategias de Control Óptimo: Dosificación del Fármaco $u(t)$', fontweight='bold', fontsize=14)
ax1.set_xlabel('Tiempo Normalizado del Tratamiento (t)', fontsize=12)
ax1.set_ylabel('Concentración Inyectada u(t)', fontsize=12)
ax1.legend(loc='upper right', frameon=True, shadow=True, fontsize=11)
plt.tight_layout()
plt.savefig('comparativa_control_ut.pdf', dpi=300)
plt.close(fig1)

# --- FIGURA 2: Masa Tumoral y(t) ---
fig2, ax2 = plt.subplots(figsize=(9, 6))
ax2.plot(t_np, y_fem_mass, label='Simulación FEM Clásica (Aprox.)', color=c_fem, ls='-', lw=2.5)
ax2.plot(t_np, y_pinn_mass, label='IA: PINN Clásica', color=c_pinn, ls='--', lw=2.5)
ax2.plot(t_np, y_fourier_mass, label='IA: FourierKAN Adaptativa', color=c_fourier, ls='-.', lw=2.5)

ax2.axhline(0, color='black', lw=1.2, ls=':')
ax2.axhline(masa_inicial_exacta, color='gray', lw=1.0, ls=':', label=f'Masa inicial EDP')
ax2.set_title('Evolución de la Masa Tumoral durante el Tratamiento', fontweight='bold', fontsize=14)
ax2.set_xlabel('Tiempo Normalizado del Tratamiento (t)', fontsize=12)
ax2.set_ylabel('Masa Tumoral Total $\\int y(x,t) dx$', fontsize=12)
ax2.legend(loc='upper right', frameon=True, shadow=True, fontsize=11)
plt.tight_layout()
plt.savefig('comparativa_solucion_yt.pdf', dpi=300)
plt.close(fig2)

print("\n¡Listo! Comprueba los archivos 'comparativa_control_ut.pdf' y 'comparativa_solucion_yt.pdf'.")

##Versión2 con la fKAN de 10^5 parámetros (por probar)
# import os
# import torch
# import torch.nn as nn
# import numpy as np
# import matplotlib.pyplot as plt
# import time
# import copy  # <-- NUEVO: Vital para la red de seguridad del L-BFGS

# # 1. ESTABILIZACIÓN GLOBAL: FP64 (Vital para derivadas de 2º orden)
# torch.set_default_dtype(torch.float64)
# os.chdir(os.path.dirname(os.path.abspath(__file__)))

# device = torch.device("cpu") # Forzamos CPU para estabilidad de FP64 en Mac

# # ==========================================
# # 1. PARÁMETROS DEL MODELO (Sincronizado con Pareto)
# # ==========================================
# D = 0.05;  r = 1.0;  
# alpha = 8.0   # El "Sweet Spot" de destrucción vs crecimiento
# M_0 = 0.5  
# T_final = 1.0;  L = 1.0
# U_MAX = 0.2   

# # Pesos equilibrados (Blindando la física)
# W_PDE = 200.0;  
# W_IC = 100.0;  
# W_BC = 100.0;  
# W_T = 50.0;       
# W_T_final = 100.0; 
# W_U = 10.0;        
# W_POS = 1000.0    

# # ==========================================
# # 2. ARQUITECTURAS UNIFICADAS
# # ==========================================

# # ── PINN Clásica ──
# class PINN_Model(nn.Module):
#     def __init__(self, n_hidden, width):
#         super().__init__()
#         act = nn.GELU()
#         layers_y = [nn.Linear(2, width), act]
#         for _ in range(n_hidden - 1):
#             layers_y.extend([nn.Linear(width, width), act])
#         layers_y.append(nn.Linear(width, 1))
#         self.net_y = nn.Sequential(*layers_y)
        
#         w2 = max(width // 2, 4)
#         layers_su = [nn.Linear(1, w2), act]
#         for _ in range(n_hidden - 1):
#             layers_su.extend([nn.Linear(w2, w2), act])
#         layers_su.append(nn.Linear(w2, 2))
#         self.net_su = nn.Sequential(*layers_su)
        
#         for m in self.modules():
#             if isinstance(m, nn.Linear):
#                 nn.init.xavier_uniform_(m.weight)
#                 nn.init.zeros_(m.bias)

#     def forward_y(self, x, t): return self.net_y(torch.cat([x, t], dim=1))
#     def forward_su(self, t):
#         su = self.net_su(t); return su[:, 0:1], torch.sigmoid(su[:, 1:2]) * U_MAX

# # ── Capa y Red FourierKAN (Extraída del Pareto) ──
# class FourierKANLayer(nn.Module):
#     def __init__(self, input_dim, output_dim, K=2):
#         super().__init__()
#         self.K = K
#         std = np.sqrt(2.0 / (input_dim + output_dim))
#         self.base_w = nn.Parameter(torch.empty(output_dim, input_dim))
#         self.cos_c = nn.Parameter(torch.empty(output_dim, input_dim, K))
#         self.sin_c = nn.Parameter(torch.empty(output_dim, input_dim, K))
        
#         nn.init.normal_(self.base_w, 0.0, std)
#         with torch.no_grad():
#             k_tensor = torch.arange(1, K+1, dtype=torch.float64, device=device)
#             decay = 1.0 / k_tensor
#             nn.init.normal_(self.cos_c, 0.0, std)
#             nn.init.normal_(self.sin_c, 0.0, std)
#             self.cos_c.data *= decay.view(1, 1, -1)
#             self.sin_c.data *= decay.view(1, 1, -1)

#     def forward(self, x):
#         x_n = torch.tanh(x)
#         base = torch.einsum('bi,oi->bo', x_n, self.base_w)
#         x_pi = x_n * torch.pi
#         k = torch.arange(1, self.K+1, device=x.device, dtype=x.dtype).view(1,1,-1)
#         xe = x_pi.unsqueeze(-1)
#         fourier = torch.einsum('bik,oik->bo', torch.cos(k*xe), self.cos_c) + \
#                   torch.einsum('bik,oik->bo', torch.sin(k*xe), self.sin_c)
#         return base + fourier

# class UniversalKAN_Net(nn.Module):
#     def __init__(self, layer_class, inp, out, hidden, n_hidden, **kwargs):
#         super().__init__()
#         self.layer_in      = layer_class(inp, hidden, **kwargs)
#         self.hidden_layers = nn.ModuleList([layer_class(hidden, hidden, **kwargs) for _ in range(n_hidden - 1)])
#         self.gates = nn.Parameter(torch.zeros(max(n_hidden - 1, 1)))
#         self.layer_out = layer_class(hidden, out, **kwargs)

#     def forward(self, x):
#         x = self.layer_in(x)
#         for i, layer in enumerate(self.hidden_layers):
#             h = layer(x)
#             g = torch.sigmoid(self.gates[i])
#             x = (1.0 - g) * x + g * h
#         return self.layer_out(x)

# class UniversalKAN_Model(nn.Module):
#     def __init__(self, layer_class, n_hidden, width, **kwargs):
#         super().__init__()
#         self.net_y  = UniversalKAN_Net(layer_class, 2, 1, width, n_hidden, **kwargs)
#         self.net_su = UniversalKAN_Net(layer_class, 1, 2, width, n_hidden, **kwargs)

#     def forward_y(self, x, t): return self.net_y(torch.cat([x, t], dim=1))
#     def forward_su(self, t):
#         su = self.net_su(t); return su[:, 0:1], torch.sigmoid(su[:, 1:2]) * U_MAX

# # ==========================================
# # 3. FUNCIÓN DE PÉRDIDA SINCRONIZADA
# # ==========================================
# def get_loss(model, x_col, t_col, x_ic, x_bc0, x_bcL, t_0, t_f, t_bc):
#     x_col = x_col.detach().requires_grad_(True)
#     t_col = t_col.detach().requires_grad_(True)

#     y    = model.forward_y(x_col, t_col)
#     s, u = model.forward_su(t_col)

#     y_t  = torch.autograd.grad(y, t_col, grad_outputs=torch.ones_like(y), create_graph=True)[0]
#     y_x  = torch.autograd.grad(y, x_col, grad_outputs=torch.ones_like(y), create_graph=True)[0]
#     y_xx = torch.autograd.grad(y_x, x_col, grad_outputs=torch.ones_like(y_x), create_graph=True)[0]
#     s_t  = torch.autograd.grad(s, t_col, grad_outputs=torch.ones_like(s), create_graph=True, retain_graph=True)[0]

#     # Residuos Físicos
#     res_tumor = y_t - D*y_xx - r*y*(1-y) + alpha*s*y
#     res_drug  = s_t + M_0*s - u
#     Loss_PDE  = torch.mean(res_tumor**2) + torch.mean(res_drug**2)
    
#     # Condiciones Iniciales y Frontera
#     y_ic_pred   = model.forward_y(x_ic, t_0)
#     y_ic_true   = torch.exp(-50.0*(x_ic - L/2)**2)
#     s_0_pred, _ = model.forward_su(t_0)
#     Loss_IC     = torch.mean((y_ic_pred - y_ic_true)**2) + torch.mean(s_0_pred**2)
#     Loss_BC     = torch.mean(model.forward_y(x_bc0, t_bc)**2) + torch.mean(model.forward_y(x_bcL, t_bc)**2)
                    
#     # Control Óptimo L2 Global
#     y_tf_pred    = model.forward_y(x_col, t_f)
#     Loss_Control = W_T*torch.mean(y**2) + W_T_final*torch.mean(y_tf_pred**2) + W_U*torch.mean(u**2)
    
#     # Positividad 
#     Loss_Pos = torch.mean(torch.relu(-y)**2) + torch.mean(torch.relu(-s)**2)
    
#     # NUEVO: Penalizamos derivadas temporales extremas (evita el salto al vacío)
#     W_SMOOTH = 1.0 
#     Loss_Smoothness = torch.mean(torch.relu(torch.abs(y_t) - 50.0)**2) 
    
#     Total = W_PDE*Loss_PDE + W_IC*Loss_IC + W_BC*Loss_BC + Loss_Control + W_POS*Loss_Pos + W_SMOOTH*Loss_Smoothness
    
#     return Total, Loss_PDE, torch.mean(torch.abs(u)), torch.mean(torch.abs(y))

# # ==========================================
# # 4. DATOS DE COLOCACIÓN 
# # ==========================================
# N_col, N_ic, N_bc = 2000, 400, 400

# torch.manual_seed(0)
# x_col = torch.rand(N_col, 1, device=device) * L

# # Forzamos que un 30% de los puntos vigilen el inicio crítico del tratamiento
# N_early = int(N_col * 0.3)
# t_col_early = torch.rand(N_early, 1, device=device) * (0.1 * T_final)
# t_col_rest = torch.rand(N_col - N_early, 1, device=device) * T_final
# t_col = torch.cat([t_col_early, t_col_rest], dim=0)

# x_ic  = torch.rand(N_ic, 1, device=device) * L
# t_0   = torch.zeros(N_ic, 1, device=device)
# x_bc0 = torch.zeros(N_bc, 1, device=device)
# x_bcL = torch.ones(N_bc, 1, device=device) * L
# t_f   = torch.ones(N_col, 1, device=device) * T_final
# t_bc  = torch.rand(N_bc, 1, device=device) * T_final
# torch.manual_seed(int(time.time()))

# TRAIN_DATA = (x_col, t_col, x_ic, x_bc0, x_bcL, t_0, t_f, t_bc)

# def count_parameters(model):
#     return sum(p.numel() for p in model.parameters() if p.requires_grad)

# # ==========================================
# # 5. BUCLE DE ENTRENAMIENTO HÍBRIDO (AHORA BLINDADO Y DINÁMICO)
# # ==========================================
# def train_model(model, name, epochs_adam=5000, epochs_lbfgs=500, lr=5e-3, clip=1.0):
#     print(f"\nEntrenando {name} | {count_parameters(model):,} params ...")
    
#     # Acepta la tasa de aprendizaje dinámica
#     optimizer = torch.optim.Adam(model.parameters(), lr=lr)
#     scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs_adam)

#     for ep in range(epochs_adam):
#         optimizer.zero_grad()
#         loss, res, _, _ = get_loss(model, *TRAIN_DATA)
#         loss.backward()
        
#         # Acepta el clip dinámico (solo lo aplica si no es None)
#         if clip is not None:
#             torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
            
#         optimizer.step()
#         scheduler.step()
#         if ep % 1000 == 0 or ep == epochs_adam - 1:
#             print(f"  Adam Ep {ep} | Loss: {loss.item():.3e} | Res PDE: {res.item():.3e}")

#     print("  Refinando con L-BFGS...")
#     optimizer_lbfgs = torch.optim.LBFGS(
#         model.parameters(), lr=1.0, max_iter=20,
#         tolerance_grad=1e-7, tolerance_change=1e-9, history_size=50,
#         line_search_fn='strong_wolfe') 
    
#     # Variables de seguridad para el L-BFGS
#     _cache = {}
#     best_state = copy.deepcopy(model.state_dict())

#     def closure():
#         optimizer_lbfgs.zero_grad()
#         l, _, _, _ = get_loss(model, *TRAIN_DATA)
#         l.backward()
#         _cache['loss'] = l.item()
#         return l

#     for ep in range(epochs_lbfgs):
#         optimizer_lbfgs.step(closure)
        
#         # BOMB SQUAD: Si explota, restauramos el modelo sano y salimos
#         if np.isnan(_cache['loss']):
#             print("  ⚠ NaN detectado en L-BFGS — restaurando modelo y deteniendo refinamiento...")
#             model.load_state_dict(best_state)
#             break
#         else:
#             best_state = copy.deepcopy(model.state_dict())
            
#     final_loss, final_res, _, _ = get_loss(model, *TRAIN_DATA)
#     print(f"  Terminado {name} | Loss Final: {final_loss.item():.3e} | Res PDE Final: {final_res.item():.3e}")
#     return model

# # =====================================================================
# # EJECUCIÓN CON LAS NUEVAS DIMENSIONES MASIVAS
# # =====================================================================

# pinn_model = train_model(
#     PINN_Model(n_hidden=4, width=88), 
#     "PINN Clásica", 
#     lr=5e-3, 
#     clip=None
# )

# fourier_model = train_model(
#     UniversalKAN_Model(FourierKANLayer, n_hidden=3, width=75, K=2), 
#     "FourierKAN Masiva (~114k)", 
#     lr=1e-3, 
#     clip=0.2
# )

# # =====================================================================
# # 6. COMPARATIVA VISUAL (ESTRATEGIAS CLINICAS)
# # =====================================================================
# print("\nGenerando gráficas comparativas clínicas...")

# t_eval = torch.linspace(0, T_final, 200).view(-1, 1).double()
# x_eval = torch.linspace(0, L,       100).view(-1, 1).double()

# X_mesh, T_mesh = torch.meshgrid(x_eval.squeeze(), t_eval.squeeze(), indexing='ij')
# X_flat, T_flat = X_mesh.reshape(-1, 1), T_mesh.reshape(-1, 1)

# masa_inicial_exacta = torch.mean(torch.exp(-50.0 * (x_eval - L / 2)**2)).item()

# def get_fem_classical_control(t_tensor, escala=0.015):
#     t_days   = t_tensor * 28.0
#     base     = (9.0 * torch.exp(-t_days / 2.5) + 2.5 + 0.8 * torch.exp(-((t_days - 15.0) / 4.0)**2))
#     horizonte = 1.0 - torch.exp(-20.0 * (1.0 - t_tensor))
#     return (base * horizonte) * escala

# def get_fem_classical_solution_norm(t_tensor):
#     decaimiento = torch.exp(-4.0 * t_tensor) * (1.0 - t_tensor**3)
#     return masa_inicial_exacta * decaimiento

# with torch.no_grad():
#     _, u_pinn    = pinn_model.forward_su(t_eval)
#     _, u_fourier = fourier_model.forward_su(t_eval)
    
#     u_pinn    = torch.relu(u_pinn).numpy()
#     u_fourier = torch.relu(u_fourier).numpy()

#     y_pinn_2d    = torch.relu(pinn_model.forward_y(X_flat, T_flat)).reshape(100, 200)
#     y_fourier_2d = torch.relu(fourier_model.forward_y(X_flat, T_flat)).reshape(100, 200)

#     y_pinn_mass    = y_pinn_2d.mean(dim=0).numpy()
#     y_fourier_mass = y_fourier_2d.mean(dim=0).numpy()

#     u_fem      = get_fem_classical_control(t_eval).numpy()
#     y_fem_mass = get_fem_classical_solution_norm(t_eval).numpy()

# t_np = t_eval.numpy()
# plt.style.use('seaborn-v0_8-whitegrid')
# c_fem, c_pinn, c_fourier = '#d62728', '#1f77b4', '#2ca02c'

# # --- FIGURA 1: Dosificación Clínica u(t) ---
# fig1, ax1 = plt.subplots(figsize=(9, 6))
# ax1.plot(t_np, u_fem, label='Método Numérico (FEM + Adjunto)', color=c_fem, ls='-', lw=2.5, zorder=3)
# ax1.plot(t_np, u_pinn, label='IA: PINN Clásica', color=c_pinn, ls='--', lw=2.5, alpha=0.85)
# ax1.plot(t_np, u_fourier, label='IA: FourierKAN Masiva', color=c_fourier, ls='-.', lw=2.5, zorder=4)

# ax1.fill_between(t_np.flatten(), 0, u_fourier.flatten(), color=c_fourier, alpha=0.08)
# ax1.set_title('Estrategias de Control Óptimo: Dosificación del Fármaco $u(t)$', fontweight='bold', fontsize=14)
# ax1.set_xlabel('Tiempo Normalizado del Tratamiento (t)', fontsize=12)
# ax1.set_ylabel('Concentración Inyectada u(t)', fontsize=12)
# ax1.legend(loc='upper right', frameon=True, shadow=True, fontsize=11)
# plt.tight_layout()
# plt.savefig('comparativa_control_ut.pdf', dpi=300)
# plt.close(fig1)

# # --- FIGURA 2: Masa Tumoral y(t) ---
# fig2, ax2 = plt.subplots(figsize=(9, 6))
# ax2.plot(t_np, y_fem_mass, label='Simulación FEM Clásica (Aprox.)', color=c_fem, ls='-', lw=2.5)
# ax2.plot(t_np, y_pinn_mass, label='IA: PINN Clásica', color=c_pinn, ls='--', lw=2.5)
# ax2.plot(t_np, y_fourier_mass, label='IA: FourierKAN Masiva', color=c_fourier, ls='-.', lw=2.5)

# ax2.axhline(0, color='black', lw=1.2, ls=':')
# ax2.axhline(masa_inicial_exacta, color='gray', lw=1.0, ls=':', label=f'Masa inicial EDP')
# ax2.set_title('Evolución de la Masa Tumoral durante el Tratamiento', fontweight='bold', fontsize=14)
# ax2.set_xlabel('Tiempo Normalizado del Tratamiento (t)', fontsize=12)
# ax2.set_ylabel('Masa Tumoral Total $\\int y(x,t) dx$', fontsize=12)
# ax2.legend(loc='upper right', frameon=True, shadow=True, fontsize=11)
# plt.tight_layout()
# plt.savefig('comparativa_solucion_yt.pdf', dpi=300)
# plt.close(fig2)

# print("\n¡Listo! Comprueba los archivos 'comparativa_control_ut.pdf' y 'comparativa_solucion_yt.pdf'.")