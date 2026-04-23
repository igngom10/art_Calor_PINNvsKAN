# heat_control_benchmark_v2.py
# ============================================================
# Benchmark: Ecuación del Calor con Control Óptimo (MÉTODO HUM)
# VERSIÓN CORREGIDA: Sistema Primal-Adjunto Completo
#
# Correcciones respecto a v1:
#   1. net_u eliminada; u := net_phi / beta  (condición de optimalidad)
#   2. Ecuación adjunta retrógrada: -phi_t - D*phi_xx = 0
#   3. Condición terminal de acoplamiento: phi(x,T) = -mu * y(x,T)
#   4. Condición de contorno adjunta: phi(0,t) = phi(L,t) = 0
#   5. Peso de acoplamiento fijo y alto (no adaptativo)
#   6. Muestreo temporal simétrico: puntos densos en t=0 y t=T
#   7. Activación tanh en FourierKAN (más estable para señales suaves)
# ============================================================

import os
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import copy
from matplotlib.backends.backend_pdf import PdfPages
import time

torch.set_default_dtype(torch.float64)
os.chdir(os.path.dirname(os.path.abspath(__file__)))

# Compatibilidad NumPy: trapezoid (≥2.0) vs trapz (<2.0)
np_trapz = getattr(np, 'trapezoid', None) or np.trapz

device = torch.device("cpu")
print(f"Iniciando benchmark v2 (sistema primal-adjunto completo) en: {device}\n")

# ==========================================
# 1. PARÁMETROS DEL PROBLEMA
# ==========================================
D       = 0.1
T_final = 1.0
L       = 1.0
mu      = 10.0   # peso HUM: penalización sobre y(·,T)
beta    = 0.1    # peso HUM: penalización sobre u

# Condición inicial: gaussiana centrada
def y0_func(x):
    return np.exp(-50.0 * (x - L / 2.0)**2)

def y0_func_torch(x):
    return torch.exp(-50.0 * (x - L / 2.0)**2)

# ==========================================
# 2. SOLUCIÓN HUM EXACTA (referencia de Fourier)
# ==========================================
N_MODES = 40

def hum_on_grid(x_grid, t_grid, n_modes=N_MODES):
    """
    Solución exacta del sistema primal-adjunto HUM por separación de variables.

    Sistema:
        Primal:   y_t - D*y_xx = u,         y(x,0) = y0,  y|boundary = 0
        Adjunto: -phi_t - D*phi_xx = 0,     phi(x,T) = -mu*y(x,T),  phi|boundary = 0
        Optim:    u = phi / beta

    Solución modal (k-ésimo modo, lambda_k = D*(k*pi)^2):
        phi_k(t) = phi_k(T) * exp(-lambda_k*(T-t))
        y_k(t)   = y_k(0)*exp(-lambda_k*t)
                   + integral_0^t exp(-lambda_k*(t-s)) * (phi_k(s)/beta) ds
    """
    Nx, Nt = len(x_grid), len(t_grid)
    Y = np.zeros((Nx, Nt))
    U = np.zeros((Nx, Nt))

    x_quad = np.linspace(0, L, 4000)
    y0_quad = y0_func(x_quad)

    for k in range(1, n_modes + 1):
        lam_k  = D * (k * np.pi / L)**2
        phi_k  = np.sqrt(2.0 / L) * np.sin(k * np.pi * x_grid / L)
        phi_k_quad = np.sqrt(2.0 / L) * np.sin(k * np.pi * x_quad / L)

        # Coeficiente modal de la condición inicial
        yk0 = np_trapz(y0_quad * phi_k_quad, x_quad)

        # Factor de acoplamiento HUM: phi_k(T) = -mu * y_k(T)
        # G_k = integral_0^T exp(-2*lam_k*(T-s)) ds = (1 - exp(-2*lam_k*T)) / (2*lam_k)
        # Para lam_k grande, exp(-2*lam_k*T) → 0, así G_k → 1/(2*lam_k)
        G_k = (1.0 - np.exp(-2.0 * lam_k * T_final)) / (2.0 * lam_k)

        # Sistema cerrado: y_k(T) * (1 + (mu/beta)*G_k) = yk0 * exp(-lam_k*T)
        denom = 1.0 + (mu / beta) * G_k
        ykT   = yk0 * np.exp(-lam_k * T_final) / denom
        phikT = -mu * ykT

        for j, t in enumerate(t_grid):
            # Adjunto: phi_k(t) = phikT * exp(-lam_k*(T-t))
            # Exponente siempre negativo → sin overflow
            phi_kt = phikT * np.exp(-lam_k * (T_final - t))
            uk_t   = phi_kt / beta

            # Estado primal — FORMA SEGURA sin overflow:
            # int_0^t exp(-lam_k*(t-s)) * (phikT/beta) * exp(-lam_k*(T-s)) ds
            # = (phikT/beta) * exp(-lam_k*(T+t)) * (exp(2*lam_k*t)-1)/(2*lam_k)
            #
            # PROBLEMA: exp(2*lam_k*t) → inf para modos altos (overflow × 0 = nan)
            # FORMA EQUIVALENTE SEGURA (factorizando):
            # exp(-lam_k*(T+t))*(exp(2*lam_k*t)-1) = exp(-lam_k*(T-t)) - exp(-lam_k*(T+t))
            # Ambos exponentes son negativos → nunca hay overflow.
            integral_safe = (np.exp(-lam_k * (T_final - t))
                             - np.exp(-lam_k * (T_final + t))) / (2.0 * lam_k)
            yk_t = yk0 * np.exp(-lam_k * t) + (phikT / beta) * integral_safe

            Y[:, j] += yk_t * phi_k
            U[:, j] += uk_t * phi_k

    return Y, U


# ==========================================
# 3. ARQUITECTURAS
# ==========================================

class PINN_Model(nn.Module):
    """
    Dos redes: net_y (estado primal) y net_phi (variable adjunta).
    El control se obtiene como u = net_phi / beta.
    """
    def __init__(self, n_hidden=4, width=48):
        super().__init__()

        def make_net(out=1):
            act = nn.Tanh()
            layers = [nn.Linear(2, width), act]
            for _ in range(n_hidden - 1):
                layers.extend([nn.Linear(width, width), act])
            layers.append(nn.Linear(width, out))
            return nn.Sequential(*layers)

        self.net_y   = make_net()
        self.net_phi = make_net()

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight, gain=nn.init.calculate_gain('tanh'))
                nn.init.zeros_(m.bias)

    def _inp(self, x, t):
        # Normalización al rango [-1, 1]
        xn = x / L * 2.0 - 1.0
        tn = t / T_final * 2.0 - 1.0
        return torch.cat([xn, tn], dim=1)

    def forward_y(self, x, t):
        # Impone BC de Dirichlet exactamente: y = x*(L-x) * net_y(x,t)
        inp = self._inp(x, t)
        raw = self.net_y(inp)
        mask = x * (L - x) / (L / 2.0)**2   # normalizado para que max=1
        return raw * mask

    def forward_phi(self, x, t):
        # Impone BC de Dirichlet exactamente: phi = x*(L-x) * net_phi(x,t)
        inp = self._inp(x, t)
        raw = self.net_phi(inp)
        mask = x * (L - x) / (L / 2.0)**2
        return raw * mask

    def forward_u(self, x, t):
        return self.forward_phi(x, t) / beta


class FourierKANLayer(nn.Module):
    def __init__(self, input_dim, output_dim, K=4):
        super().__init__()
        self.K = K
        std = np.sqrt(2.0 / (input_dim + output_dim))
        self.base_w = nn.Parameter(torch.empty(output_dim, input_dim))
        self.cos_c  = nn.Parameter(torch.empty(output_dim, input_dim, K))
        self.sin_c  = nn.Parameter(torch.empty(output_dim, input_dim, K))
        nn.init.normal_(self.base_w, 0.0, std)
        with torch.no_grad():
            k_t = torch.arange(1, K + 1, dtype=torch.float64)
            decay = 1.0 / k_t
            nn.init.normal_(self.cos_c, 0.0, std)
            nn.init.normal_(self.sin_c, 0.0, std)
            self.cos_c.data *= decay.view(1, 1, -1)
            self.sin_c.data *= decay.view(1, 1, -1)

    def forward(self, x):
        x_n  = torch.tanh(x)   # tanh más estable que identidad para señales suaves
        base = torch.einsum('bi,oi->bo', x_n, self.base_w)
        x_pi = x_n * torch.pi
        k    = torch.arange(1, self.K + 1, device=x.device, dtype=x.dtype).view(1, 1, -1)
        xe   = x_pi.unsqueeze(-1)
        cos_part = torch.einsum('bik,oik->bo', torch.cos(k * xe), self.cos_c)
        sin_part = torch.einsum('bik,oik->bo', torch.sin(k * xe), self.sin_c)
        return base + cos_part + sin_part


class KAN_Net(nn.Module):
    def __init__(self, inp, out, hidden, n_hidden, K=4):
        super().__init__()
        self.layer_in  = FourierKANLayer(inp, hidden, K)
        self.hiddens   = nn.ModuleList(
            [FourierKANLayer(hidden, hidden, K) for _ in range(n_hidden - 1)]
        )
        self.gates     = nn.Parameter(torch.zeros(max(n_hidden - 1, 1)))
        self.layer_out = FourierKANLayer(hidden, out, K)

    def forward(self, x):
        x = self.layer_in(x)
        for i, layer in enumerate(self.hiddens):
            g = torch.sigmoid(self.gates[i])
            x = (1.0 - g) * x + g * layer(x)
        return self.layer_out(x)


class FourierKAN_Model(nn.Module):
    def __init__(self, n_hidden=3, width=24, K=4):
        super().__init__()
        self.net_y   = KAN_Net(2, 1, width, n_hidden, K)
        self.net_phi = KAN_Net(2, 1, width, n_hidden, K)

    def _inp(self, x, t):
        xn = x / L * 2.0 - 1.0
        tn = t / T_final * 2.0 - 1.0
        return torch.cat([xn, tn], dim=1)

    def forward_y(self, x, t):
        raw  = self.net_y(self._inp(x, t))
        mask = x * (L - x) / (L / 2.0)**2
        return raw * mask

    def forward_phi(self, x, t):
        raw  = self.net_phi(self._inp(x, t))
        mask = x * (L - x) / (L / 2.0)**2
        return raw * mask

    def forward_u(self, x, t):
        return self.forward_phi(x, t) / beta


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# ==========================================
# 4. PUNTOS DE COLOCACIÓN
# ==========================================
N_col = 4000
N_ic  = 800
torch.manual_seed(42)

# Espacial: uniforme
x_col = torch.rand(N_col, 1, dtype=torch.float64) * L

# Temporal: simétrico — capas límite en t=0 y t=T, exploración uniforme en el centro
# La ecuación primal va hacia adelante (necesita t=0) y la adjunta hacia atrás (necesita t=T)
N_early = int(N_col * 0.20)   # capa límite inicial  (primal IC)
N_late  = int(N_col * 0.20)   # capa límite terminal (adjunto terminal condition)
N_mid   = N_col - N_early - N_late

t_early = torch.rand(N_early, 1, dtype=torch.float64) * (0.05 * T_final)
t_late  = T_final - torch.rand(N_late,  1, dtype=torch.float64) * (0.05 * T_final)
t_mid   = torch.rand(N_mid,   1, dtype=torch.float64) * T_final
t_col   = torch.cat([t_early, t_late, t_mid], dim=0)

# Condición inicial (primal)
x_ic = torch.rand(N_ic, 1, dtype=torch.float64) * L
t_0  = torch.zeros(N_ic, 1, dtype=torch.float64)

# Tiempo terminal (acoplamiento adjunto)
t_f = torch.ones(N_col, 1, dtype=torch.float64) * T_final

TRAIN_DATA = (x_col, t_col, x_ic, t_0, t_f)


# ==========================================
# 5. FUNCIÓN DE PÉRDIDA: SISTEMA PRIMAL-ADJUNTO COMPLETO
# ==========================================

def get_loss_components(model, x_col, t_col, x_ic, t_0, t_f, is_train=True):
    """
    Pérdidas del sistema de optimalidad de primer orden (condiciones KKT):

    [Primal]
        PDE:   y_t - D*y_xx - u = 0,          u := phi/beta
        IC:    y(x, 0) = y0(x)
        BC:    impuesta arquitecturalmente (máscara x*(L-x))

    [Adjunto]
        PDE:  -phi_t - D*phi_xx = 0           (ecuación del calor retrógrada)
        TC:    phi(x, T) = -mu * y(x, T)      (acoplamiento — CRÍTICO)
        BC:    impuesta arquitecturalmente

    [Control]
        u = phi / beta  (definición, no pérdida — ya está en la arquitectura)
    """
    cg = is_train  # create_graph sólo en entrenamiento

    # --- Derivadas primal ---
    xc = x_col.detach().requires_grad_(True)
    tc = t_col.detach().requires_grad_(True)

    y   = model.forward_y(xc, tc)
    y_t = torch.autograd.grad(y, tc,
            grad_outputs=torch.ones_like(y), create_graph=cg, retain_graph=True)[0]
    y_x = torch.autograd.grad(y, xc,
            grad_outputs=torch.ones_like(y), create_graph=True, retain_graph=True)[0]
    y_xx = torch.autograd.grad(y_x, xc,
            grad_outputs=torch.ones_like(y_x), create_graph=cg, retain_graph=True)[0]

    u = model.forward_u(xc, tc)   # u = phi / beta

    # Residuo PDE primal: y_t - D*y_xx - u = 0
    L_primal_pde = torch.mean((y_t - D * y_xx - u) ** 2)

    # Condición inicial primal: y(x, 0) = y0(x)
    y_ic = model.forward_y(x_ic, t_0)
    L_ic = torch.mean((y_ic - y0_func_torch(x_ic)) ** 2)

    # --- Derivadas adjunto ---
    phi   = model.forward_phi(xc, tc)
    phi_t = torch.autograd.grad(phi, tc,
            grad_outputs=torch.ones_like(phi), create_graph=cg, retain_graph=True)[0]
    phi_x = torch.autograd.grad(phi, xc,
            grad_outputs=torch.ones_like(phi), create_graph=True, retain_graph=True)[0]
    phi_xx = torch.autograd.grad(phi_x, xc,
            grad_outputs=torch.ones_like(phi_x), create_graph=cg, retain_graph=True)[0]

    # Residuo PDE adjunta: -phi_t - D*phi_xx = 0  (signo negativo: va hacia atrás en tiempo)
    L_adj_pde = torch.mean((phi_t + D * phi_xx) ** 2)

    # --- Acoplamiento terminal (CONDICIÓN CRÍTICA) ---
    # phi(x, T) = -mu * y(x, T)
    y_T   = model.forward_y(x_col, t_f)
    phi_T = model.forward_phi(x_col, t_f)
    L_coupling = torch.mean((phi_T + mu * y_T) ** 2)

    # --- Regularización del funcional de control (convexidad) ---
    L_ctrl = torch.mean(u ** 2)

    return L_primal_pde, L_ic, L_adj_pde, L_coupling, L_ctrl


# ==========================================
# 6. ENTRENAMIENTO CON PESOS ADAPTATIVOS SELECTIVOS
# ==========================================

def train_model(model, name, epochs_adam=8000, epochs_lbfgs=800, lr=3e-3):
    print(f"\n{'='*65}")
    print(f"  {name}  |  {count_parameters(model):,} parámetros")
    print(f"{'='*65}")

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=2000, T_mult=2, eta_min=1e-5)

    # Pesos:
    #   - w_coupling: FIJO y alto — es la condición matemática que define la optimalidad
    #   - w_primal, w_adj: adaptativos para equilibrar magnitudes
    #   - w_ic: adaptativo
    #   - w_ctrl: fijo pequeño (regularización suave)
    W = {
        'primal' : 1.0,
        'ic'     : 50.0,
        'adj'    : 1.0,
        'coupling': 200.0,   # FIJO — no adaptar jamás
        'ctrl'   : 0.01,
    }
    alpha_ema = 0.92

    hist_loss, hist_res_p, hist_res_a, hist_coup = [], [], [], []
    t0 = time.time()

    # ---- FASE 1: Adam con ponderación adaptativa ----
    for ep in range(epochs_adam):
        optimizer.zero_grad()

        L_p, L_ic, L_a, L_c, L_ctrl = get_loss_components(model, *TRAIN_DATA, is_train=True)

        # Actualización adaptativa cada 200 épocas
        # Solo se adaptan primal, ic y adj — coupling NUNCA
        if ep % 200 == 0 and ep > 0:
            with torch.no_grad():
                ref = L_p.item()
                # Equilibrar IC con el residuo primal
                w_ic_new  = ref / (L_ic.item()  + 1e-10)
                # Equilibrar adjunto con el residuo primal
                w_adj_new = ref / (L_a.item()   + 1e-10)

                W['ic']  = alpha_ema * W['ic']  + (1 - alpha_ema) * np.clip(w_ic_new,  1.0, 500.0)
                W['adj'] = alpha_ema * W['adj'] + (1 - alpha_ema) * np.clip(w_adj_new, 0.1, 50.0)

        total = (W['primal']  * L_p
               + W['ic']      * L_ic
               + W['adj']     * L_a
               + W['coupling']* L_c
               + W['ctrl']    * L_ctrl)

        total.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        hist_loss.append(total.item())
        hist_res_p.append(L_p.item())
        hist_res_a.append(L_a.item())
        hist_coup.append(L_c.item())

        if ep % 1000 == 0 or ep == epochs_adam - 1:
            print(f"  Adam {ep:5d} | Total: {total.item():.3e} | "
                  f"PDE_y: {L_p.item():.3e} | PDE_phi: {L_a.item():.3e} | "
                  f"Coupling: {L_c.item():.3e} | "
                  f"w_ic={W['ic']:.1f} w_adj={W['adj']:.2f}")

    # ---- FASE 2: L-BFGS con pesos congelados ----
    print(f"\n  L-BFGS ({epochs_lbfgs} pasos) — pesos congelados para estabilidad Hessiana...")
    opt_lbfgs = torch.optim.LBFGS(
        model.parameters(), lr=1.0, max_iter=25,
        tolerance_grad=1e-9, tolerance_change=1e-11,
        history_size=60, line_search_fn='strong_wolfe')

    _cache = {}
    best_loss  = float('inf')
    best_state = copy.deepcopy(model.state_dict())

    def closure():
        opt_lbfgs.zero_grad()
        L_p, L_ic, L_a, L_c, L_ctrl = get_loss_components(model, *TRAIN_DATA, is_train=True)
        l = (W['primal']  * L_p
           + W['ic']      * L_ic
           + W['adj']     * L_a
           + W['coupling']* L_c
           + W['ctrl']    * L_ctrl)
        l.backward()
        _cache.update({'loss': l.item(), 'pde_p': L_p.item(),
                       'pde_a': L_a.item(), 'coup': L_c.item()})
        return l

    for ep in range(epochs_lbfgs):
        opt_lbfgs.step(closure)

        if np.isnan(_cache.get('loss', float('nan'))):
            print(f"  NaN en L-BFGS ep {ep} — restaurando mejor estado...")
            model.load_state_dict(best_state)
            break

        if _cache['loss'] < best_loss:
            best_loss  = _cache['loss']
            best_state = copy.deepcopy(model.state_dict())

        hist_loss.append(_cache['loss'])
        hist_res_p.append(_cache['pde_p'])
        hist_res_a.append(_cache['pde_a'])
        hist_coup.append(_cache['coup'])

        if ep % 100 == 0 or ep == epochs_lbfgs - 1:
            print(f"  LBFGS {ep:4d} | Loss: {_cache['loss']:.3e} | "
                  f"PDE_y: {_cache['pde_p']:.3e} | PDE_phi: {_cache['pde_a']:.3e} | "
                  f"Coupling: {_cache['coup']:.3e}")

    model.load_state_dict(best_state)
    elapsed = time.time() - t0
    print(f"\n  Tiempo: {elapsed:.0f}s | Loss final: {hist_loss[-1]:.3e}")
    return model, hist_loss, hist_res_p, hist_res_a, hist_coup, elapsed


# ==========================================
# 7. ENTRENAMIENTO
# ==========================================

lbl_pinn    = "PINN primal-adjunto (~20k)"
pinn_model  = PINN_Model(n_hidden=4, width=48)
pinn_model, pinn_loss, pinn_res_p, pinn_res_a, pinn_coup, pinn_time = \
    train_model(pinn_model, lbl_pinn, epochs_adam=8000, epochs_lbfgs=800, lr=3e-3)

lbl_fourier    = "FourierKAN primal-adjunto (~12k)"
fourier_model  = FourierKAN_Model(n_hidden=3, width=24, K=4)
fourier_model, fourier_loss, fourier_res_p, fourier_res_a, fourier_coup, fourier_time = \
    train_model(fourier_model, lbl_fourier, epochs_adam=8000, epochs_lbfgs=800, lr=5e-3)


# ==========================================
# 8. EVALUACIÓN vs SOLUCIÓN HUM EXACTA
# ==========================================
print("\nCalculando solución HUM de referencia (Fourier)...")

Nx, Nt = 120, 240
x_grid  = np.linspace(0, L, Nx)
t_grid  = np.linspace(0, T_final, Nt)
X_mesh, T_mesh = np.meshgrid(x_grid, t_grid, indexing='ij')

Y_hum, U_hum = hum_on_grid(x_grid, t_grid, n_modes=N_MODES)

X_flat = torch.tensor(X_mesh.reshape(-1, 1))
T_flat = torch.tensor(T_mesh.reshape(-1, 1))

with torch.no_grad():
    Y_pinn    = pinn_model.forward_y(X_flat, T_flat).numpy().reshape(Nx, Nt)
    U_pinn    = pinn_model.forward_u(X_flat, T_flat).numpy().reshape(Nx, Nt)
    PHI_pinn  = pinn_model.forward_phi(X_flat, T_flat).numpy().reshape(Nx, Nt)

    Y_fourier   = fourier_model.forward_y(X_flat, T_flat).numpy().reshape(Nx, Nt)
    U_fourier   = fourier_model.forward_u(X_flat, T_flat).numpy().reshape(Nx, Nt)
    PHI_fourier = fourier_model.forward_phi(X_flat, T_flat).numpy().reshape(Nx, Nt)


def rel_l2(pred, ref):
    return np.sqrt(np.mean((pred - ref)**2)) / (np.sqrt(np.mean(ref**2)) + 1e-12)

def check_coupling(Y, PHI, x_grid, t_grid):
    """Verifica ||phi(T) + mu*y(T)|| / ||mu*y(T)|| — debe ser << 1"""
    yT   = Y[:, -1]
    phiT = PHI[:, -1]
    return np.sqrt(np.mean((phiT + mu * yT)**2)) / (np.sqrt(np.mean((mu * yT)**2)) + 1e-12)

err_y_pinn    = rel_l2(Y_pinn,    Y_hum)
err_u_pinn    = rel_l2(U_pinn,    U_hum)
err_y_fourier = rel_l2(Y_fourier, Y_hum)
err_u_fourier = rel_l2(U_fourier, U_hum)

coup_pinn    = check_coupling(Y_pinn,    PHI_pinn,    x_grid, t_grid)
coup_fourier = check_coupling(Y_fourier, PHI_fourier, x_grid, t_grid)

# Verificación adicional: residuo de la PDE adjunta en la solución entrenada
PHI_hum = -mu * Y_hum   # phi = -mu*y en cada punto (para verificación rough)

print(f"\n{'Modelo':<35} {'Err_y':>10} {'Err_u':>10} {'Coupling':>12} {'Tiempo':>8}")
print("-" * 80)
print(f"{'HUM Exacto (referencia)':<35} {'0':>10} {'0':>10} {'0':>12} {'-':>8}")
print(f"{lbl_pinn:<35} {err_y_pinn:>10.4e} {err_u_pinn:>10.4e} {coup_pinn:>12.4e} {pinn_time:>8.0f}s")
print(f"{lbl_fourier:<35} {err_y_fourier:>10.4e} {err_u_fourier:>10.4e} {coup_fourier:>12.4e} {fourier_time:>8.0f}s")

# Valores medios para curvas
y_hum_mean     = Y_hum.mean(axis=0)
u_hum_mean     = U_hum.mean(axis=0)
y_pinn_mean    = Y_pinn.mean(axis=0)
u_pinn_mean    = U_pinn.mean(axis=0)
y_fourier_mean = Y_fourier.mean(axis=0)
u_fourier_mean = U_fourier.mean(axis=0)


# ==========================================
# 9. PDF DE RESULTADOS (5 PÁGINAS)
# ==========================================
print("\nGenerando PDF de resultados...")
plt.style.use('seaborn-v0_8-whitegrid')
pdf_filename = 'heat_control_v2_resultados.pdf'

c_hum     = '#d62728'
c_pinn    = '#1f77b4'
c_fourier = '#2ca02c'

with PdfPages(pdf_filename) as pdf:

    # ---- PÁG 1: Convergencia de pérdidas ----
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # PINN
    n_adam = 8000
    ep_all = np.arange(len(pinn_loss))
    axes[0].semilogy(ep_all, pinn_loss,   color=c_pinn,    lw=1.5, label='Loss total')
    axes[0].semilogy(ep_all, pinn_res_p,  color='steelblue', lw=1.2, ls='--', label='PDE primal')
    axes[0].semilogy(ep_all, pinn_res_a,  color='darkorange', lw=1.2, ls='--', label='PDE adjunta')
    axes[0].semilogy(ep_all, pinn_coup,   color='crimson',  lw=1.5, ls=':', label='Acoplamiento φ(T)+μy(T)')
    axes[0].axvline(n_adam, color='gray', ls=':', lw=1.0, label='Adam → L-BFGS')
    axes[0].set_title('Convergencia PINN primal-adjunto', fontweight='bold')
    axes[0].set_xlabel('Época'); axes[0].set_ylabel('Pérdida'); axes[0].legend(fontsize=9)

    # FourierKAN
    ep_all_f = np.arange(len(fourier_loss))
    axes[1].semilogy(ep_all_f, fourier_loss,   color=c_fourier,   lw=1.5, label='Loss total')
    axes[1].semilogy(ep_all_f, fourier_res_p,  color='steelblue', lw=1.2, ls='--', label='PDE primal')
    axes[1].semilogy(ep_all_f, fourier_res_a,  color='darkorange', lw=1.2, ls='--', label='PDE adjunta')
    axes[1].semilogy(ep_all_f, fourier_coup,   color='crimson',   lw=1.5, ls=':', label='Acoplamiento φ(T)+μy(T)')
    axes[1].axvline(n_adam, color='gray', ls=':', lw=1.0, label='Adam → L-BFGS')
    axes[1].set_title('Convergencia FourierKAN primal-adjunto', fontweight='bold')
    axes[1].set_xlabel('Época'); axes[1].legend(fontsize=9)

    fig.suptitle('Historial de entrenamiento del sistema primal-adjunto HUM', fontsize=14, fontweight='bold')
    plt.tight_layout()
    pdf.savefig(fig, bbox_inches='tight'); plt.close(fig)

    # ---- PÁG 2: Mapas del control u(x,t) ----
    fig, axs = plt.subplots(1, 3, figsize=(16, 5.5))
    vmin = min(np.nanmin(U_hum), np.nanmin(U_pinn), np.nanmin(U_fourier))
    vmax = max(np.nanmax(U_hum), np.nanmax(U_pinn), np.nanmax(U_fourier))
    kw = dict(vmin=vmin, vmax=vmax, aspect='auto', origin='lower',
              extent=[0, T_final, 0, L], cmap='RdBu_r')

    axs[0].imshow(U_hum,     **kw); axs[0].set_title('HUM Exacto (referencia)', fontweight='bold')
    axs[1].imshow(U_pinn,    **kw); axs[1].set_title(f'PINN\nError L2-rel: {err_u_pinn:.2e}', fontweight='bold')
    axs[2].imshow(U_fourier, **kw); axs[2].set_title(f'FourierKAN\nError L2-rel: {err_u_fourier:.2e}', fontweight='bold')

    for ax in axs:
        ax.set_xlabel('Tiempo (t)'); ax.set_ylabel('Espacio (x)'); ax.grid(False)
    cbar = fig.colorbar(axs[0].images[0], ax=axs.ravel().tolist(), shrink=0.8, pad=0.02)
    cbar.set_label('Control óptimo $u(x,t) = \\varphi(x,t)/\\beta$')
    fig.suptitle('Control Óptimo Distribuido $u(x,t)$', fontsize=15, fontweight='bold', y=1.02)
    pdf.savefig(fig, bbox_inches='tight'); plt.close(fig)

    # ---- PÁG 3: Mapas del estado y(x,t) ----
    fig, axs = plt.subplots(1, 3, figsize=(16, 5.5))
    vmin = min(np.nanmin(Y_hum), np.nanmin(Y_pinn), np.nanmin(Y_fourier))
    vmax = max(np.nanmax(Y_hum), np.nanmax(Y_pinn), np.nanmax(Y_fourier))
    kw = dict(vmin=vmin, vmax=vmax, aspect='auto', origin='lower',
              extent=[0, T_final, 0, L], cmap='viridis')

    axs[0].imshow(Y_hum,     **kw); axs[0].set_title('HUM Exacto (referencia)', fontweight='bold')
    axs[1].imshow(Y_pinn,    **kw); axs[1].set_title(f'PINN\nError L2-rel: {err_y_pinn:.2e}', fontweight='bold')
    axs[2].imshow(Y_fourier, **kw); axs[2].set_title(f'FourierKAN\nError L2-rel: {err_y_fourier:.2e}', fontweight='bold')

    for ax in axs:
        ax.set_xlabel('Tiempo (t)'); ax.set_ylabel('Espacio (x)'); ax.grid(False)
    cbar = fig.colorbar(axs[0].images[0], ax=axs.ravel().tolist(), shrink=0.8, pad=0.02)
    cbar.set_label('Estado del sistema $y(x,t)$')
    fig.suptitle('Evolución del Estado $y(x,t)$', fontsize=15, fontweight='bold', y=1.02)
    pdf.savefig(fig, bbox_inches='tight'); plt.close(fig)

    # ---- PÁG 4: Curvas de control y estado promediados ----
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    axes[0].plot(t_grid, u_hum_mean,    color=c_hum,     lw=2.5, ls='-',  label='HUM Exacto')
    axes[0].plot(t_grid, u_pinn_mean,   color=c_pinn,    lw=2.0, ls='--', label='PINN', alpha=0.85)
    axes[0].plot(t_grid, u_fourier_mean,color=c_fourier, lw=2.0, ls='-.', label='FourierKAN')
    axes[0].axhline(0, color='black', lw=0.8, ls=':')
    axes[0].set_title('Esfuerzo de control promedio $\\langle u(\\cdot,t)\\rangle_x$', fontweight='bold', fontsize=13)
    axes[0].set_xlabel('Tiempo (t)'); axes[0].set_ylabel('Intensidad media del control')
    axes[0].legend(frameon=True, shadow=True, fontsize=11)

    axes[1].plot(t_grid, y_hum_mean,    color=c_hum,     lw=2.5, ls='-',  label='HUM Exacto')
    axes[1].plot(t_grid, y_pinn_mean,   color=c_pinn,    lw=2.0, ls='--', label='PINN', alpha=0.85)
    axes[1].plot(t_grid, y_fourier_mean,color=c_fourier, lw=2.0, ls='-.', label='FourierKAN')
    axes[1].axhline(0, color='black', lw=0.8, ls=':')
    axes[1].set_title('Temperatura media $\\langle y(\\cdot,t)\\rangle_x$', fontweight='bold', fontsize=13)
    axes[1].set_xlabel('Tiempo (t)'); axes[1].set_ylabel('Estado promedio')
    axes[1].legend(frameon=True, shadow=True, fontsize=11)

    fig.suptitle('Comparativa de curvas promediadas', fontsize=14, fontweight='bold')
    plt.tight_layout()
    pdf.savefig(fig, bbox_inches='tight'); plt.close(fig)

    # ---- PÁG 5: Verificación del acoplamiento phi(T) = -mu*y(T) ----
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    phi_hum_T   = -mu * Y_hum[:, -1]           # phi_T exacta = -mu * y_T
    phi_pinn_T  = PHI_pinn[:, -1]
    phi_four_T  = PHI_fourier[:, -1]

    axes[0].plot(x_grid, phi_hum_T,  color=c_hum,     lw=2.5, ls='-',  label='$\\varphi(x,T)$ exacto = $-\\mu y(x,T)$')
    axes[0].plot(x_grid, phi_pinn_T, color=c_pinn,    lw=2.0, ls='--', label='$\\varphi(x,T)$ PINN')
    axes[0].plot(x_grid, phi_four_T, color=c_fourier, lw=2.0, ls='-.', label='$\\varphi(x,T)$ FourierKAN')
    axes[0].set_title('Verificación del acoplamiento en $t=T$\n'
                      '$\\varphi(x,T) = -\\mu\\, y(x,T)$  (condición HUM)', fontweight='bold')
    axes[0].set_xlabel('$x$'); axes[0].legend(fontsize=10)

    # Error puntual del acoplamiento
    err_coup_pinn  = np.abs(phi_pinn_T - phi_hum_T)
    err_coup_four  = np.abs(phi_four_T - phi_hum_T)
    axes[1].semilogy(x_grid, err_coup_pinn  + 1e-16, color=c_pinn,    lw=2.0, ls='--',
                     label=f'PINN  (err L2-rel: {coup_pinn:.2e})')
    axes[1].semilogy(x_grid, err_coup_four  + 1e-16, color=c_fourier, lw=2.0, ls='-.',
                     label=f'FourierKAN (err L2-rel: {coup_fourier:.2e})')
    axes[1].set_title('Error puntual del acoplamiento $|\\varphi(x,T)+\\mu y(x,T)|$', fontweight='bold')
    axes[1].set_xlabel('$x$'); axes[1].legend(fontsize=10)

    fig.suptitle('Condición de transversalidad HUM: $\\varphi(\\cdot,T) = -\\mu\\,y(\\cdot,T)$',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    pdf.savefig(fig, bbox_inches='tight'); plt.close(fig)

print(f"\nProceso finalizado. PDF guardado como: '{pdf_filename}'")
print(f"\nResumen de errores:")
print(f"  PINN    — Err_y: {err_y_pinn:.3e}  Err_u: {err_u_pinn:.3e}  Coupling: {coup_pinn:.3e}")
print(f"  FourKAN — Err_y: {err_y_fourier:.3e}  Err_u: {err_u_fourier:.3e}  Coupling: {coup_fourier:.3e}")

# Verificación de sanidad numérica
for nombre, arr in [('Y_hum', Y_hum), ('U_hum', U_hum),
                     ('Y_pinn', Y_pinn), ('U_pinn', U_pinn),
                     ('Y_fourier', Y_fourier), ('U_fourier', U_fourier)]:
    n_nan = np.sum(np.isnan(arr))
    n_inf = np.sum(np.isinf(arr))
    if n_nan > 0 or n_inf > 0:
        print(f"  AVISO: {nombre} tiene {n_nan} NaN y {n_inf} Inf")