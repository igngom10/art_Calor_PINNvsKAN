import os
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import time
from matplotlib.backends.backend_pdf import PdfPages

# 1. ESTABILIZACIÓN GLOBAL: FP64 y DETECCIÓN DE GPU
torch.set_default_dtype(torch.float64)
os.chdir(os.path.dirname(os.path.abspath(__file__)))

# FORZAMOS CPU: El framework MPS de Mac no soporta float64, 
# el cual es estrictamente necesario para la estabilidad del Hessiano (L-BFGS).
device = torch.device("cpu")
print(f"🚀 Iniciando entrenamiento utilizando dispositivo: {device} (Precisión float64 garantizada)\n")

# ==========================================
# 1. PARÁMETROS DEL MODELO (Sincronizado con versión clínica)
# ==========================================
D = 0.05;  r = 1.0;  
alpha = 8.0   # El "Sweet Spot" de destrucción vs crecimiento
M_0 = 0.5  
T_final = 1.0;  L = 1.0
U_MAX = 0.2   # Acotado estrictamente a la escala del método clásico

# Pesos equilibrados (Blindando la física)
W_PDE = 200.0;  
W_IC = 100.0;  
W_BC = 100.0;  
W_T = 50.0;       # Presión integral del tumor
W_T_final = 100.0; # Forzar erradicación final
W_U = 2.0;        # Coste del fármaco (evita que la red inyecte a lo loco)
W_POS = 1000.0    # Barrera biológica

# Nivel de error aproximado del método de Elementos Finitos (FEM) Clásico
FEM_RESIDUAL_BASELINE = 5e-4 

# ==========================================
# 2. ARQUITECTURAS (Solo PINN, cKAN y FourierKAN)
# ==========================================

# ── PINN ─────────────────────────────────────────────────────────────────────
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


# ── Estructura universal KAN ─────────────────────────────────────────────────
class UniversalKAN_Net(nn.Module):
    def __init__(self, layer_class, inp, out, hidden, n_hidden, **kwargs):
        super().__init__()
        self.layer_in      = layer_class(inp, hidden, **kwargs)
        self.hidden_layers = nn.ModuleList(
            [layer_class(hidden, hidden, **kwargs) for _ in range(n_hidden - 1)])
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


# ── Capas KAN ─────────────────────────────────────────────────────────────────

class ChebyKANLayer(nn.Module):
    def __init__(self, input_dim, output_dim, degree=3):
        super().__init__()
        self.degree = degree
        self.coeffs = nn.Parameter(torch.empty(output_dim, input_dim, degree + 1))
        nn.init.normal_(self.coeffs, 0.0, np.sqrt(2.0 / (input_dim + output_dim)))

    def forward(self, x):
        x = torch.tanh(x)
        T = [torch.ones_like(x), x]
        for _ in range(2, self.degree + 1):
            T.append(2 * x * T[-1] - T[-2])
        return torch.einsum('bid,oid->bo', torch.stack(T, dim=-1), self.coeffs)


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


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# ==========================================
# 3. FUNCIÓN DE PÉRDIDA SINCRONIZADA
# ==========================================
def get_loss(model, x_col, t_col, x_ic, x_bc0, x_bcL, t_0, t_f, t_bc, is_train=True):
    x_col = x_col.detach().requires_grad_(True)
    t_col = t_col.detach().requires_grad_(True)

    y    = model.forward_y(x_col, t_col)
    s, u = model.forward_su(t_col)

    y_t  = torch.autograd.grad(y,   t_col, grad_outputs=torch.ones_like(y), create_graph=is_train, retain_graph=True)[0]
    y_x  = torch.autograd.grad(y,   x_col, grad_outputs=torch.ones_like(y), create_graph=True,     retain_graph=True)[0]
    y_xx = torch.autograd.grad(y_x, x_col, grad_outputs=torch.ones_like(y_x), create_graph=is_train, retain_graph=True)[0]
    s_t  = torch.autograd.grad(s,   t_col, grad_outputs=torch.ones_like(s), create_graph=is_train, retain_graph=True)[0]

    # 1. Residuos Físicos
    res_tumor    = y_t - D*y_xx - r*y*(1-y) + alpha*s*y
    res_drug     = s_t + M_0*s - u
    Loss_PDE     = torch.mean(res_tumor**2) + torch.mean(res_drug**2)
    
    # 2. Condiciones Iniciales y Frontera
    y_ic_pred    = model.forward_y(x_ic, t_0)
    y_ic_true    = torch.exp(-50.0*(x_ic - L/2)**2)
    s_0_pred, _  = model.forward_su(t_0)
    Loss_IC      = torch.mean((y_ic_pred - y_ic_true)**2) + torch.mean(s_0_pred**2)
    
    Loss_BC      = (torch.mean(model.forward_y(x_bc0, t_bc)**2) + torch.mean(model.forward_y(x_bcL, t_bc)**2))
    
    # 3. Control Óptimo L2 Global
    y_tf_pred    = model.forward_y(x_col, t_f)
    Loss_Control = W_T*torch.mean(y**2) + W_T_final*torch.mean(y_tf_pred**2) + W_U*torch.mean(u**2)
    
    # 4. Positividad
    Loss_Pos = torch.mean(torch.relu(-y)**2) + torch.mean(torch.relu(-s)**2)
    
    # NUEVO: Penalizamos derivadas temporales extremas (evita el salto al vacío)
    # Si la red intenta hacer un escalón, y_t se dispara al infinito y la loss explota.
    W_SMOOTH = 1.0 
    Loss_Smoothness = torch.mean(torch.relu(torch.abs(y_t) - 50.0)**2) # 50.0 es un límite de velocidad biológica razonable
    
    Total = W_PDE*Loss_PDE + W_IC*Loss_IC + W_BC*Loss_BC + Loss_Control + W_POS*Loss_Pos + W_SMOOTH*Loss_Smoothness

    return Total, Loss_PDE


# ==========================================
# 4. DATOS DE COLOCACIÓN 
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

# ==========================================
# 5. ENTRENAMIENTO 2 FASES (Adam + L-BFGS Seguro)
# ==========================================
def train_model(model, model_name, epochs_adam=5000, epochs_lbfgs=500, lr=5e-3, grad_clip=1.0):
    print(f"\n--- {model_name}  |  {count_parameters(model):,} params ---")

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs_adam)

    t0 = time.time()

    # ── Fase 1: Adam ──
    for ep in range(epochs_adam):
        optimizer.zero_grad()
        loss, res = get_loss(model, *TRAIN_DATA, is_train=True)
        loss.backward()
        if grad_clip: torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        scheduler.step()

        if ep % 1000 == 0 or ep == epochs_adam - 1:
            print(f"  Adam {ep:4d} | LR {optimizer.param_groups[0]['lr']:.1e} | Loss {loss.item():.3e} | Res PDE {res.item():.3e}")

    # ── Fase 2: L-BFGS (Con strong_wolfe para evitar NaNs) ──
    print(f"  → L-BFGS ({epochs_lbfgs} steps)...")
    optimizer_lbfgs = torch.optim.LBFGS(
        model.parameters(), lr=1.0, max_iter=20,
        tolerance_grad=1e-7, tolerance_change=1e-9, history_size=50,
        line_search_fn='strong_wolfe')

    def closure():
        optimizer_lbfgs.zero_grad()
        loss, _ = get_loss(model, *TRAIN_DATA, is_train=True)
        loss.backward()
        return loss

    for ep in range(epochs_lbfgs):
        optimizer_lbfgs.step(closure)

    # Evaluación final estricta sobre los puntos de física (Entrenamiento)
    final_loss_train, final_res_train = get_loss(model, *TRAIN_DATA, is_train=False)
    
    elapsed = time.time() - t0
    print(f"  Tiempo: {elapsed:.1f}s | Train Loss: {final_loss_train.item():.3e} | Train Res PDE: {final_res_train.item():.3e}")
    
    return model, final_loss_train.item(), final_res_train.item()


# ==========================================
# 6. CONFIGS DE PARETO (Solo 3 Arquitecturas)
# ==========================================
E_ADAM  = 5000
E_LBFGS = 500

pinn_configs = [
    {"n_hidden": 1, "width": 12,  "lr": 5e-3, "clip": None},
    {"n_hidden": 2, "width": 22,  "lr": 5e-3, "clip": None},
    {"n_hidden": 3, "width": 36,  "lr": 5e-3, "clip": None},
    {"n_hidden": 3, "width": 72,  "lr": 5e-3, "clip": None},
    {"n_hidden": 4, "width": 88,  "lr": 5e-3, "clip": None},
    {"n_hidden": 5, "width": 160, "lr": 2e-3, "clip": None},
]

kan_topology = [
    {"n_hidden": 2, "width": 6,  "lr": 1e-2, "clip": 1.0},
    {"n_hidden": 2, "width": 12, "lr": 1e-2, "clip": 1.0},
    {"n_hidden": 2, "width": 20, "lr": 1e-2, "clip": 1.0},
    {"n_hidden": 2, "width": 35, "lr": 5e-3, "clip": 0.5},
    {"n_hidden": 2, "width": 52, "lr": 2e-3, "clip": 0.3},
    {"n_hidden": 3, "width": 75, "lr": 1e-3, "clip": 0.2}
]

ARCHS = {
    "PINN":       {"configs": pinn_configs, "color": "#1f77b4",
                   "builder": lambda n,w,**_: PINN_Model(n, w)},
    "cKAN":       {"configs": kan_topology, "color": "#ff7f0e",
                   "builder": lambda n,w,**_: UniversalKAN_Model(ChebyKANLayer,   n, w, degree=3)},
    "FourierKAN": {"configs": kan_topology, "color": "#2ca02c",
                   "builder": lambda n,w,**_: UniversalKAN_Model(FourierKANLayer, n, w, K=2)},
}

resultados_pareto = {name: [] for name in ARCHS}

print("\n" + "="*50)
print(" BENCHMARK PARETO (FINALISTAS: PINN, cKAN, fKAN) ")
print("="*50)

for name, arch in ARCHS.items():
    for cfg in arch["configs"]:
        n_h, w = cfg["n_hidden"], cfg["width"]
        modelo = arch["builder"](n_h, w).to(device)
        n_par  = count_parameters(modelo)
        tag    = f"{name}\n{n_par:,}p"

        # Entrenamos
        modelo, loss_train, res_train = train_model(
            modelo, f"{name} n={n_h} w={w}", E_ADAM, E_LBFGS, lr=cfg["lr"], grad_clip=cfg["clip"]
        )

        resultados_pareto[name].append((n_par, res_train, tag))

# ==========================================
# 7. GENERACIÓN DEL PDF ÚNICO Y TABLA
# ==========================================
print("\nGenerando gráfico de Pareto en PDF...")
pdf_filename = 'Frontera_Pareto_Residuo_Finalistas.pdf'

with PdfPages(pdf_filename) as pdf:
    fig, ax = plt.subplots(figsize=(12, 7))
    ax.set_title('Frontera de Pareto — Residuo Físico PDE vs. Coste Paramétrico\n(Resultados en Entrenamiento)', fontweight='bold', fontsize=14)
    
    # Línea base del método FEM
    ax.axhline(FEM_RESIDUAL_BASELINE, color='red', linestyle='--', linewidth=1.5, label='Residuo FEM Clásico (Referencia)')
    
    for name, arch in ARCHS.items():
        col = arch["color"]
        pts = resultados_pareto[name]
        
        px, py, pl = zip(*pts)
        ax.plot(px, py, color=col, alpha=0.4, lw=2)
        ax.scatter(px, py, color=col, s=100, zorder=5, label=name)
        
        for x, y, nm in zip(px, py, pl):
            ax.annotate(nm, (x, y), textcoords="offset points", xytext=(0, 12), ha='center', fontsize=7, color=col)
            
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('Parámetros entrenables', fontsize=12)
    ax.set_ylabel('Residuo Físico PDE (Entrenamiento)', fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(True, which='both', ls='--', alpha=0.4)
    plt.tight_layout()
    pdf.savefig(fig)
    plt.close()

# Imprimir tabla final 
print("\n\n" + "="*60)
print(f"{'Arquitectura':<18} | {'Parámetros':<12} | {'Residuo PDE (Train)':<15}")
print("-" * 60)
for name in ARCHS.keys():
    for pt in resultados_pareto[name]:
        n_params = pt[0]
        res_train = pt[1]
        print(f"{name:<18} | {n_params:<12,d} | {res_train:<15.4e}")
print("="*60)
print(f"\n¡Proceso finalizado! Gráfico guardado como: '{pdf_filename}'")