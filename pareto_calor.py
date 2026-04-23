# heat_control_pareto.py
# ============================================================
# Frontera de Pareto — Control Óptimo HUM (Ecuación del Calor)
# Arquitecturas: PINN primal-adjunto + FourierKAN primal-adjunto
#
# Métrica de evaluación: error L2 relativo vs solución HUM exacta
#   - Err_y : ||y_red - y_HUM|| / ||y_HUM||
#   - Err_u : ||u_red - u_HUM|| / ||u_HUM||
#   - Err_phi: residuo de acoplamiento ||phi(T) + mu*y(T)||
#
# Proceso de entrenamiento idéntico al benchmark v2:
#   - Sistema primal-adjunto completo (u := phi/beta)
#   - Pesos adaptativos EMA; w_coupling FIJO=200
#   - Adam + CosineAnnealingWarmRestarts → L-BFGS strong_wolfe
#   - BC Dirichlet impuestas arquitecturalmente (máscara x*(L-x))
# ============================================================

import os
import copy
import time

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.backends.backend_pdf import PdfPages

torch.set_default_dtype(torch.float64)
os.chdir(os.path.dirname(os.path.abspath(__file__)))

# Compatibilidad NumPy ≥2.0 (trapezoid) y <2.0 (trapz)
np_trapz = getattr(np, 'trapezoid', None) or np.trapz

device = torch.device("cpu")
print(f"Pareto HUM — dispositivo: {device}  (float64)\n")

# ============================================================
# 1. PARÁMETROS FÍSICOS
# ============================================================
D       = 0.1
T_final = 1.0
L       = 1.0
mu      = 10.0   # penalización ||y(·,T)||²
beta    = 0.1    # penalización ||u||²

def y0_func_torch(x):
    return torch.exp(-50.0 * (x - L / 2.0) ** 2)

# ============================================================
# 2. SOLUCIÓN HUM EXACTA (Fourier, referencia)
# ============================================================
N_MODES = 40

def hum_on_grid(x_grid, t_grid, n_modes=N_MODES):
    """
    Solución exacta del sistema primal-adjunto HUM.
    Integral del estado en forma segura (sin overflow para modos altos):
        exp(-lam*(T+t))*(exp(2*lam*t)-1) = exp(-lam*(T-t)) - exp(-lam*(T+t))
    """
    Nx, Nt  = len(x_grid), len(t_grid)
    Y = np.zeros((Nx, Nt))
    U = np.zeros((Nx, Nt))

    x_quad  = np.linspace(0, L, 4000)
    y0_quad = np.exp(-50.0 * (x_quad - L / 2.0) ** 2)

    for k in range(1, n_modes + 1):
        lam_k      = D * (k * np.pi / L) ** 2
        phi_k      = np.sqrt(2.0 / L) * np.sin(k * np.pi * x_grid / L)
        phi_k_quad = np.sqrt(2.0 / L) * np.sin(k * np.pi * x_quad / L)

        yk0 = np_trapz(y0_quad * phi_k_quad, x_quad)

        G_k   = (1.0 - np.exp(-2.0 * lam_k * T_final)) / (2.0 * lam_k)
        denom = 1.0 + (mu / beta) * G_k
        ykT   = yk0 * np.exp(-lam_k * T_final) / denom
        phikT = -mu * ykT

        for j, t in enumerate(t_grid):
            phi_kt = phikT * np.exp(-lam_k * (T_final - t))
            uk_t   = phi_kt / beta
            integral_safe = (np.exp(-lam_k * (T_final - t))
                             - np.exp(-lam_k * (T_final + t))) / (2.0 * lam_k)
            yk_t = yk0 * np.exp(-lam_k * t) + (phikT / beta) * integral_safe
            Y[:, j] += yk_t * phi_k
            U[:, j] += uk_t * phi_k

    return Y, U


# ============================================================
# 3. GRID DE EVALUACIÓN (precalculado una sola vez)
# ============================================================
print("Calculando solución HUM exacta (referencia)...")
Nx_ev, Nt_ev = 100, 200
x_ev = np.linspace(0, L, Nx_ev)
t_ev = np.linspace(0, T_final, Nt_ev)
Y_hum, U_hum = hum_on_grid(x_ev, t_ev)

X_mesh, T_mesh = np.meshgrid(x_ev, t_ev, indexing='ij')
X_flat = torch.tensor(X_mesh.reshape(-1, 1))
T_flat = torch.tensor(T_mesh.reshape(-1, 1))

norm_y = np.sqrt(np.mean(Y_hum ** 2)) + 1e-12
norm_u = np.sqrt(np.mean(U_hum ** 2)) + 1e-12
print(f"  ||Y_hum||={norm_y:.4f}  ||U_hum||={norm_u:.4f}  (sin NaN: {not np.any(np.isnan(Y_hum))})\n")


def eval_errors(model):
    """Devuelve (err_y, err_u, err_coupling) respecto a la solución HUM exacta."""
    with torch.no_grad():
        Y_pred   = model.forward_y(X_flat, T_flat).numpy().reshape(Nx_ev, Nt_ev)
        U_pred   = model.forward_u(X_flat, T_flat).numpy().reshape(Nx_ev, Nt_ev)
        PHI_pred = model.forward_phi(X_flat, T_flat).numpy().reshape(Nx_ev, Nt_ev)

    err_y    = np.sqrt(np.mean((Y_pred - Y_hum) ** 2)) / norm_y
    err_u    = np.sqrt(np.mean((U_pred - U_hum) ** 2)) / norm_u
    # Residuo de acoplamiento: ||phi(T) + mu*y(T)|| / ||mu*y(T)||
    yT_pred  = Y_pred[:, -1]
    phiT_pred= PHI_pred[:, -1]
    norm_coup= np.sqrt(np.mean((mu * Y_hum[:, -1]) ** 2)) + 1e-12
    err_coup = np.sqrt(np.mean((phiT_pred + mu * yT_pred) ** 2)) / norm_coup

    return float(err_y), float(err_u), float(err_coup)


# ============================================================
# 4. PUNTOS DE COLOCACIÓN (fijos para todo el benchmark)
# ============================================================
N_col = 4000
N_ic  = 800
torch.manual_seed(42)

x_col   = torch.rand(N_col, 1, dtype=torch.float64) * L
N_early = int(N_col * 0.20)
N_late  = int(N_col * 0.20)
N_mid   = N_col - N_early - N_late
t_early = torch.rand(N_early, 1, dtype=torch.float64) * (0.05 * T_final)
t_late  = T_final - torch.rand(N_late,  1, dtype=torch.float64) * (0.05 * T_final)
t_mid   = torch.rand(N_mid,   1, dtype=torch.float64) * T_final
t_col   = torch.cat([t_early, t_late, t_mid], dim=0)

x_ic = torch.rand(N_ic, 1, dtype=torch.float64) * L
t_0  = torch.zeros(N_ic, 1, dtype=torch.float64)
t_f  = torch.ones(N_col, 1, dtype=torch.float64) * T_final

TRAIN_DATA = (x_col, t_col, x_ic, t_0, t_f)


# ============================================================
# 5. FUNCIÓN DE PÉRDIDA — SISTEMA PRIMAL-ADJUNTO COMPLETO
# ============================================================
def get_loss_components(model, x_col, t_col, x_ic, t_0, t_f, W, is_train=True):
    """
    Cinco componentes del sistema KKT:
        L_primal_pde : y_t - D*y_xx - u = 0
        L_ic         : y(x,0) = y0(x)
        L_adj_pde    : -phi_t - D*phi_xx = 0
        L_coupling   : phi(x,T) = -mu*y(x,T)   ← condición HUM crítica
        L_ctrl       : regularización ||u||²
    BC Dirichlet impuestas arquitecturalmente (sin término de pérdida propio).
    """
    cg = is_train

    xc = x_col.detach().requires_grad_(True)
    tc = t_col.detach().requires_grad_(True)

    # --- Primal ---
    y    = model.forward_y(xc, tc)
    y_t  = torch.autograd.grad(y,   tc,  grad_outputs=torch.ones_like(y),
                                create_graph=cg, retain_graph=True)[0]
    y_x  = torch.autograd.grad(y,   xc,  grad_outputs=torch.ones_like(y),
                                create_graph=True, retain_graph=True)[0]
    y_xx = torch.autograd.grad(y_x, xc,  grad_outputs=torch.ones_like(y_x),
                                create_graph=cg, retain_graph=True)[0]
    u    = model.forward_u(xc, tc)

    L_primal_pde = torch.mean((y_t - D * y_xx - u) ** 2)
    L_ic         = torch.mean((model.forward_y(x_ic, t_0) - y0_func_torch(x_ic)) ** 2)

    # --- Adjunto ---
    phi    = model.forward_phi(xc, tc)
    phi_t  = torch.autograd.grad(phi,   tc,  grad_outputs=torch.ones_like(phi),
                                  create_graph=cg, retain_graph=True)[0]
    phi_x  = torch.autograd.grad(phi,   xc,  grad_outputs=torch.ones_like(phi),
                                  create_graph=True, retain_graph=True)[0]
    phi_xx = torch.autograd.grad(phi_x, xc,  grad_outputs=torch.ones_like(phi_x),
                                  create_graph=cg, retain_graph=True)[0]

    L_adj_pde  = torch.mean((phi_t + D * phi_xx) ** 2)

    # --- Acoplamiento terminal ---
    y_T   = model.forward_y(x_col,   t_f)
    phi_T = model.forward_phi(x_col, t_f)
    L_coupling = torch.mean((phi_T + mu * y_T) ** 2)

    # --- Regularización control ---
    L_ctrl = torch.mean(u ** 2)

    total = (W['primal']   * L_primal_pde
           + W['ic']       * L_ic
           + W['adj']      * L_adj_pde
           + W['coupling'] * L_coupling
           + W['ctrl']     * L_ctrl)

    return total, L_primal_pde, L_adj_pde, L_coupling


# ============================================================
# 6. ENTRENAMIENTO (idéntico al benchmark v2)
# ============================================================
def train_model(model, tag, epochs_adam=6000, epochs_lbfgs=500, lr=3e-3):
    n_par = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n  [{tag}]  {n_par:,} params")

    W = {'primal': 1.0, 'ic': 50.0, 'adj': 1.0, 'coupling': 200.0, 'ctrl': 0.01}
    alpha_ema = 0.92

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=2000, T_mult=2, eta_min=1e-5)

    t0 = time.time()

    # ── Fase 1: Adam ──
    for ep in range(epochs_adam):
        optimizer.zero_grad()
        total, L_p, L_a, L_c = get_loss_components(model, *TRAIN_DATA, W, is_train=True)

        if ep % 200 == 0 and ep > 0:
            with torch.no_grad():
                ref = L_p.item()
                W['ic']  = alpha_ema * W['ic']  + (1 - alpha_ema) * float(np.clip(ref / (L_p.item() + 1e-10), 1.0, 500.0))
                W['adj'] = alpha_ema * W['adj'] + (1 - alpha_ema) * float(np.clip(ref / (L_a.item() + 1e-10), 0.1, 50.0))

        total.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        if ep % 2000 == 0 or ep == epochs_adam - 1:
            print(f"    Adam {ep:5d} | total={total.item():.2e} "
                  f"pde_y={L_p.item():.2e} pde_phi={L_a.item():.2e} coup={L_c.item():.2e}")

    # ── Fase 2: L-BFGS ──
    opt_lbfgs = torch.optim.LBFGS(
        model.parameters(), lr=1.0, max_iter=25,
        tolerance_grad=1e-9, tolerance_change=1e-11,
        history_size=60, line_search_fn='strong_wolfe')

    _cache = {}
    best_loss  = float('inf')
    best_state = copy.deepcopy(model.state_dict())

    def closure():
        opt_lbfgs.zero_grad()
        tot, L_p, L_a, L_c = get_loss_components(model, *TRAIN_DATA, W, is_train=True)
        tot.backward()
        _cache.update({'loss': tot.item(), 'pde_p': L_p.item(),
                       'pde_a': L_a.item(), 'coup': L_c.item()})
        return tot

    for ep in range(epochs_lbfgs):
        opt_lbfgs.step(closure)
        if np.isnan(_cache.get('loss', float('nan'))):
            print(f"    NaN en L-BFGS ep {ep} — restaurando...")
            model.load_state_dict(best_state)
            break
        if _cache['loss'] < best_loss:
            best_loss  = _cache['loss']
            best_state = copy.deepcopy(model.state_dict())

        if ep % 200 == 0 or ep == epochs_lbfgs - 1:
            print(f"    LBFGS {ep:4d} | total={_cache['loss']:.2e} "
                  f"pde_y={_cache['pde_p']:.2e} coup={_cache['coup']:.2e}")

    model.load_state_dict(best_state)
    elapsed = time.time() - t0

    err_y, err_u, err_coup = eval_errors(model)
    print(f"    → Err_y={err_y:.3e}  Err_u={err_u:.3e}  Coup={err_coup:.3e}  t={elapsed:.0f}s")
    return err_y, err_u, err_coup, elapsed


# ============================================================
# 7. ARQUITECTURAS — PINN Y FOURIERKAN PRIMAL-ADJUNTO
# ============================================================

# ── PINN (Tanh + máscara Dirichlet + normalización entrada) ──
class PINN_Model(nn.Module):
    def __init__(self, n_hidden=4, width=48):
        super().__init__()

        def make_net():
            act = nn.Tanh()
            layers = [nn.Linear(2, width), act]
            for _ in range(n_hidden - 1):
                layers.extend([nn.Linear(width, width), act])
            layers.append(nn.Linear(width, 1))
            return nn.Sequential(*layers)

        self.net_y   = make_net()
        self.net_phi = make_net()

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight, gain=nn.init.calculate_gain('tanh'))
                nn.init.zeros_(m.bias)

    def _inp(self, x, t):
        return torch.cat([x / L * 2.0 - 1.0, t / T_final * 2.0 - 1.0], dim=1)

    def _mask(self, x):
        return x * (L - x) / (L / 2.0) ** 2

    def forward_y(self,   x, t): return self.net_y(self._inp(x, t))   * self._mask(x)
    def forward_phi(self, x, t): return self.net_phi(self._inp(x, t)) * self._mask(x)
    def forward_u(self,   x, t): return self.forward_phi(x, t) / beta


# ── FourierKAN (capa Fourier + gating residual + máscara Dirichlet) ──
class FourierKANLayer(nn.Module):
    def __init__(self, input_dim, output_dim, K=4):
        super().__init__()
        self.K  = K
        std     = np.sqrt(2.0 / (input_dim + output_dim))
        self.base_w = nn.Parameter(torch.empty(output_dim, input_dim))
        self.cos_c  = nn.Parameter(torch.empty(output_dim, input_dim, K))
        self.sin_c  = nn.Parameter(torch.empty(output_dim, input_dim, K))
        nn.init.normal_(self.base_w, 0.0, std)
        with torch.no_grad():
            decay = 1.0 / torch.arange(1, K + 1, dtype=torch.float64)
            nn.init.normal_(self.cos_c, 0.0, std); self.cos_c.data *= decay.view(1, 1, -1)
            nn.init.normal_(self.sin_c, 0.0, std); self.sin_c.data *= decay.view(1, 1, -1)

    def forward(self, x):
        xn   = torch.tanh(x)
        base = torch.einsum('bi,oi->bo', xn, self.base_w)
        xe   = (xn * torch.pi).unsqueeze(-1)
        k    = torch.arange(1, self.K + 1, device=x.device, dtype=x.dtype).view(1, 1, -1)
        return (base
                + torch.einsum('bik,oik->bo', torch.cos(k * xe), self.cos_c)
                + torch.einsum('bik,oik->bo', torch.sin(k * xe), self.sin_c))


class KAN_Net(nn.Module):
    def __init__(self, inp, out, hidden, n_hidden, K=4):
        super().__init__()
        self.layer_in  = FourierKANLayer(inp, hidden, K)
        self.hiddens   = nn.ModuleList(
            [FourierKANLayer(hidden, hidden, K) for _ in range(n_hidden - 1)])
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
        return torch.cat([x / L * 2.0 - 1.0, t / T_final * 2.0 - 1.0], dim=1)

    def _mask(self, x):
        return x * (L - x) / (L / 2.0) ** 2

    def forward_y(self,   x, t): return self.net_y(self._inp(x, t))   * self._mask(x)
    def forward_phi(self, x, t): return self.net_phi(self._inp(x, t)) * self._mask(x)
    def forward_u(self,   x, t): return self.forward_phi(x, t) / beta


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# ============================================================
# 8. CONFIGURACIONES DE PARETO
#    — se barren distintas profundidades/anchuras para trazar
#      la frontera params vs error HUM
# ============================================================
# Épocas reducidas respecto al benchmark v2 para que el barrido
# sea ejecutable; aumentar E_ADAM/E_LBFGS para mayor precisión.
E_ADAM  = 6000
E_LBFGS = 500

# (n_hidden, width, lr)
PINN_CFGS = [
    (2, 16,  4e-3),
    (2, 32,  4e-3),
    (3, 32,  3e-3),
    (3, 48,  3e-3),
    (4, 48,  3e-3),
    (4, 64,  2e-3),
    (5, 80,  2e-3),
]

KAN_CFGS = [
    (2,  8,  8e-3),
    (2, 14,  6e-3),
    (2, 20,  5e-3),
    (3, 20,  5e-3),
    (3, 28,  4e-3),
    (3, 36,  3e-3),
    (4, 36,  3e-3),
]

ARCHS = {
    "PINN": {
        "cfgs"   : PINN_CFGS,
        "builder": lambda nh, w, **_: PINN_Model(n_hidden=nh, width=w),
        "color"  : "#1f77b4",
        "marker" : "o",
    },
    "FourierKAN": {
        "cfgs"   : KAN_CFGS,
        "builder": lambda nh, w, **_: FourierKAN_Model(n_hidden=nh, width=w, K=4),
        "color"  : "#2ca02c",
        "marker" : "s",
    },
}

# ============================================================
# 9. BUCLE PRINCIPAL DEL BENCHMARK
# ============================================================
print("="*65)
print("  BARRIDO PARETO — Parámetros vs Error HUM")
print("="*65)

results = {name: [] for name in ARCHS}  # lista de dicts por configuración

for arch_name, arch in ARCHS.items():
    print(f"\n{'━'*65}")
    print(f"  ARQUITECTURA: {arch_name}")
    print(f"{'━'*65}")

    for (nh, w, lr) in arch["cfgs"]:
        model = arch["builder"](nh, w)
        n_par = count_parameters(model)
        tag   = f"{arch_name} h={nh} w={w}"

        err_y, err_u, err_coup, elapsed = train_model(
            model, tag,
            epochs_adam=E_ADAM,
            epochs_lbfgs=E_LBFGS,
            lr=lr
        )

        results[arch_name].append({
            "n_params" : n_par,
            "err_y"    : err_y,
            "err_u"    : err_u,
            "err_coup" : err_coup,
            "elapsed"  : elapsed,
            "n_hidden" : nh,
            "width"    : w,
            "label"    : f"h{nh}·w{w}",
        })

# ============================================================
# 10. TABLA RESUMEN EN CONSOLA
# ============================================================
print("\n\n" + "="*80)
print(f"{'Arquitectura':<16} {'h':>3} {'w':>4} {'Params':>8} "
      f"{'Err_y':>10} {'Err_u':>10} {'Coup':>10} {'t(s)':>7}")
print("-"*80)
for name, pts in results.items():
    for p in pts:
        print(f"{name:<16} {p['n_hidden']:>3} {p['width']:>4} {p['n_params']:>8,d} "
              f"{p['err_y']:>10.3e} {p['err_u']:>10.3e} {p['err_coup']:>10.3e} {p['elapsed']:>7.0f}")
print("="*80)


# ============================================================
# 11. PDF DE RESULTADOS (3 páginas)
# ============================================================
print("\nGenerando PDF...")
plt.style.use('seaborn-v0_8-whitegrid')
pdf_filename = 'heat_control_pareto_HUM.pdf'

# Paleta consistente
C = {name: arch["color"]  for name, arch in ARCHS.items()}
M = {name: arch["marker"] for name, arch in ARCHS.items()}

def pareto_frontier(pts_sorted_by_x):
    """Extrae la frontera de Pareto (mínimo error para cada nº de params creciente)."""
    front = []
    best  = float('inf')
    for p in pts_sorted_by_x:
        if p[1] < best:
            best = p[1]
            front.append(p)
    return front


with PdfPages(pdf_filename) as pdf:

    # ── PÁG 1: Pareto params vs Err_y  y  params vs Err_u ──
    fig, axes = plt.subplots(1, 2, figsize=(15, 6.5))
    fig.suptitle(
        'Frontera de Pareto — Error L2-rel vs Solución HUM Exacta\n'
        '(Ecuación del Calor, sistema primal-adjunto, u = φ/β)',
        fontsize=13, fontweight='bold')

    for ax, metric, ylabel, title in [
        (axes[0], 'err_y', 'Error L2-rel  $\\|y_{red}-y_{HUM}\\|/\\|y_{HUM}\\|$',
         'Estado $y(x,t)$'),
        (axes[1], 'err_u', 'Error L2-rel  $\\|u_{red}-u_{HUM}\\|/\\|u_{HUM}\\|$',
         'Control óptimo $u(x,t)$'),
    ]:
        for name, pts in results.items():
            xs = [p['n_params'] for p in pts]
            ys = [p[metric]     for p in pts]
            labels = [p['label'] for p in pts]

            ax.scatter(xs, ys, color=C[name], marker=M[name],
                       s=80, zorder=5, label=name)
            ax.plot(xs, ys, color=C[name], alpha=0.35, lw=1.5)

            # Frontera de Pareto resaltada
            sorted_pts = sorted(zip(xs, ys, labels), key=lambda t: t[0])
            front      = pareto_frontier(sorted_pts)
            if front:
                fx, fy, _ = zip(*front)
                ax.plot(fx, fy, color=C[name], lw=2.5, ls='--',
                        alpha=0.85, label=f'{name} (Pareto)')

            # Anotaciones de configuración
            for x, y, lbl in zip(xs, ys, labels):
                ax.annotate(lbl, (x, y),
                            textcoords='offset points', xytext=(4, 6),
                            fontsize=7, color=C[name], alpha=0.9)

        ax.set_xscale('log'); ax.set_yscale('log')
        ax.set_xlabel('Parámetros entrenables', fontsize=11)
        ax.set_ylabel(ylabel, fontsize=10)
        ax.set_title(title, fontweight='bold', fontsize=12)
        ax.legend(fontsize=9, framealpha=0.9)
        ax.grid(True, which='both', ls=':', alpha=0.5)
        ax.xaxis.set_major_formatter(ticker.FuncFormatter(
            lambda v, _: f'{int(v):,}' if v >= 1000 else str(int(v))))

    plt.tight_layout()
    pdf.savefig(fig, bbox_inches='tight'); plt.close(fig)

    # ── PÁG 2: Pareto params vs Err_coupling  +  scatter Err_y vs Err_u ──
    fig, axes = plt.subplots(1, 2, figsize=(15, 6.5))
    fig.suptitle(
        'Calidad del acoplamiento HUM y correlación entre errores',
        fontsize=13, fontweight='bold')

    # Izquierda: params vs err_coupling
    ax = axes[0]
    for name, pts in results.items():
        xs = [p['n_params'] for p in pts]
        ys = [p['err_coup'] for p in pts]
        ax.scatter(xs, ys, color=C[name], marker=M[name], s=80, zorder=5, label=name)
        ax.plot(xs, ys, color=C[name], alpha=0.35, lw=1.5)
    ax.set_xscale('log'); ax.set_yscale('log')
    ax.set_xlabel('Parámetros entrenables', fontsize=11)
    ax.set_ylabel('Error acoplamiento  $\\|\\varphi(T)+\\mu y(T)\\|/\\|\\mu y(T)\\|$', fontsize=10)
    ax.set_title('Residuo de la condición HUM $\\varphi(T)=-\\mu y(T)$',
                 fontweight='bold', fontsize=12)
    ax.legend(fontsize=9); ax.grid(True, which='both', ls=':', alpha=0.5)

    # Derecha: Err_y vs Err_u (scatter por arquitectura)
    ax = axes[1]
    for name, pts in results.items():
        xs = [p['err_y'] for p in pts]
        ys = [p['err_u'] for p in pts]
        ns = [p['n_params'] for p in pts]
        sc = ax.scatter(xs, ys, c=ns, cmap='viridis', marker=M[name],
                        s=100, zorder=5, vmin=0, vmax=max(max(p['n_params'] for p in pts2)
                                                          for pts2 in results.values()))
        # Contorno de color de arquitectura
        ax.scatter(xs, ys, facecolors='none', edgecolors=C[name],
                   marker=M[name], s=120, lw=1.5, label=name, zorder=6)
    plt.colorbar(sc, ax=ax, label='Nº parámetros')
    ax.set_xscale('log'); ax.set_yscale('log')
    ax.set_xlabel('Err_y  (estado)', fontsize=11)
    ax.set_ylabel('Err_u  (control)', fontsize=11)
    ax.set_title('Correlación Err_y vs Err_u\n(color = nº parámetros)', fontweight='bold', fontsize=12)
    ax.legend(fontsize=9); ax.grid(True, which='both', ls=':', alpha=0.5)

    plt.tight_layout()
    pdf.savefig(fig, bbox_inches='tight'); plt.close(fig)

    # ── PÁG 3: Tabla visual completa ──
    fig, ax = plt.subplots(figsize=(14, 0.45 * (sum(len(v) for v in results.values()) + 3) + 1.5))
    ax.axis('off')

    col_labels = ['Arquitectura', 'h', 'w', 'Params', 'Err_y', 'Err_u', 'Coup.', 't (s)']
    rows = []
    row_colors = []
    color_map  = {'PINN': '#dce9f7', 'FourierKAN': '#d9f0da'}

    for name, pts in results.items():
        for p in sorted(pts, key=lambda x: x['n_params']):
            rows.append([
                name,
                str(p['n_hidden']),
                str(p['width']),
                f"{p['n_params']:,}",
                f"{p['err_y']:.3e}",
                f"{p['err_u']:.3e}",
                f"{p['err_coup']:.3e}",
                f"{p['elapsed']:.0f}",
            ])
            row_colors.append([color_map.get(name, '#f5f5f5')] * len(col_labels))

    tbl = ax.table(
        cellText=rows,
        colLabels=col_labels,
        cellColours=row_colors,
        cellLoc='center',
        loc='center',
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    tbl.scale(1.0, 1.5)

    # Cabecera en negrita
    for j in range(len(col_labels)):
        tbl[(0, j)].set_facecolor('#2c3e50')
        tbl[(0, j)].set_text_props(color='white', fontweight='bold')

    ax.set_title('Resumen completo del barrido de Pareto\n'
                 'Errores L2-rel vs solución HUM exacta (Fourier, 40 modos)',
                 fontsize=12, fontweight='bold', pad=20)
    plt.tight_layout()
    pdf.savefig(fig, bbox_inches='tight'); plt.close(fig)

print(f"\nPDF guardado como: '{pdf_filename}'")
print("Proceso finalizado.")