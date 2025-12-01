import torch
import torch.nn as nn
from torch.autograd import grad
import math

torch.set_default_dtype(torch.float64)
device = "cuda" if torch.cuda.is_available() else "cpu"

class HestonPINN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(3, 64), nn.Tanh(),
            nn.Linear(64, 64), nn.Tanh(),
            nn.Linear(64, 64), nn.Tanh(),
            nn.Linear(64, 1)
        )
    def forward(self, x):  # x = [s_tilde, v_tilde, t_tilde]
        return self.net(x).squeeze(-1)  # V normalizzato (qui lo teniamo "in euro" per semplicità)

# ---- helpers: scaling ----
def scale_inputs(S, v, t, K=100.0, T=1.0, v_max=0.8):
    s_t = torch.log(torch.clamp(S, min=1e-12) / K)                 # \tilde s
    v_t = torch.log1p(torch.clamp(v, min=0.0)) / math.log1p(v_max) # \tilde v in [0,1] circa
    t_t = t / T                                                    # \tilde t
    return torch.stack([s_t, v_t, t_t], dim=1)

# ---- PDE loss (corretta con (r-q) e mix-derivative) ----
def heston_pde_loss(model, S, v, t, r=0.05, q=0.0, rho=-0.7, kappa=2.0, theta=0.04, sigma=0.3,
                    K=100.0, T=1.0, v_max=0.8):
    S = S.detach().clone().requires_grad_(True).to(device)
    v = v.detach().clone().requires_grad_(True).to(device)
    t = t.detach().clone().requires_grad_(True).to(device)

    x = scale_inputs(S, v, t, K=K, T=T, v_max=v_max)
    V = model(x)

    dV_dS = grad(V, S, torch.ones_like(V), create_graph=True)[0]
    dV_dv = grad(V, v, torch.ones_like(V), create_graph=True)[0]
    dV_dt = grad(V, t, torch.ones_like(V), create_graph=True)[0]

    d2V_dS2  = grad(dV_dS, S, torch.ones_like(dV_dS), create_graph=True)[0]
    d2V_dv2  = grad(dV_dv, v, torch.ones_like(dV_dv), create_graph=True)[0]
    d2V_dSdv = grad(dV_dS, v, torch.ones_like(dV_dS), create_graph=True)[0]

    pde = (dV_dt
           + 0.5 * v * S**2 * d2V_dS2
           + rho * sigma * v * S * d2V_dSdv
           + 0.5 * sigma**2 * v * d2V_dv2
           + (r - q) * S * dV_dS
           + kappa * (theta - v) * dV_dv
           - r * V)

    return (pde**2).mean()

# ---- Terminal loss (t = T) ----
def heston_terminal_loss(model, S, v, K=100.0, T=1.0, v_max=0.8):
    t1 = torch.full_like(S, T)
    xT = scale_inputs(S, v, t1, K=K, T=T, v_max=v_max)
    V_pred = model(xT)
    V_true = torch.clamp(S - K, min=0.0)
    return ((V_pred - V_true)**2).mean()

# ---- Boundary losses ----
def boundary_S0_loss(model, t, v, K=100.0, T=1.0, v_max=0.8):
    S0 = torch.zeros_like(t)
    x = scale_inputs(S0, v, t, K=K, T=T, v_max=v_max)
    V = model(x)
    return (V**2).mean()

def boundary_Smax_loss(model, t, v, S_max, r=0.05, q=0.0, K=100.0, T=1.0, v_max=0.8):
    S = torch.full_like(t, S_max)
    x = scale_inputs(S, v, t, K=K, T=T, v_max=v_max)
    V = model(x)
    target = S*torch.exp(-q*(T - t)) - K*torch.exp(-r*(T - t))
    return ((V - target)**2).mean()

def boundary_v_neumann_loss(model, S, t, v_val, K=100.0, T=1.0, v_max=0.8):
    # penalizza V_v ~ 0 su v=0 o v=v_max
    v = torch.full_like(S, v_val, requires_grad=True)
    x = scale_inputs(S, v, t, K=K, T=T, v_max=v_max)
    V = model(x)
    dV_dv = grad(V, v, torch.ones_like(V), create_graph=True)[0]
    return (dV_dv**2).mean()

# ---- Training loop minimale con sampling decente ----
def train_heston(model, epochs=15000, lr=1e-3,
                 K=100.0, T=1.0, S_max=400.0, v_max=0.8,
                 r=0.05, q=0.0, rho=-0.7, kappa=2.0, theta=0.04, sigma=0.3,
                 batch_pde=8192, seed=0):
    torch.manual_seed(seed)
    model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    for ep in range(epochs):
        # --- PDE collocation (curriculum leggero: più denso vicino all'ATM) ---
        t = torch.rand(batch_pde, device=device) * T
        # log-uniform su S per coprire code
        uS = torch.rand(batch_pde, device=device)
        S = K * torch.exp((2*uS - 1.0) * 1.5)  # ~ e^{Uniform[-1.5,1.5]}*K
        # bias v verso valori medi
        u = torch.rand(batch_pde, device=device)
        v = (u**2) * v_max

        loss_pde = heston_pde_loss(model, S, v, t, r=r, q=q, rho=rho, kappa=kappa, theta=theta, sigma=sigma,
                                   K=K, T=T, v_max=v_max)

        # --- Terminale (t=T): S log-uniform, v random (il payoff non dipende da v) ---
        S_T = K * torch.exp((2*torch.rand(4096, device=device) - 1.0) * 1.5)
        v_T = torch.rand_like(S_T) * v_max
        loss_T = heston_terminal_loss(model, S_T, v_T, K=K, T=T, v_max=v_max)

        # --- Bordi S ---
        tb = torch.rand(2048, device=device) * T
        vb = torch.rand_like(tb) * v_max
        loss_S0   = boundary_S0_loss(model, tb, vb, K=K, T=T, v_max=v_max)
        loss_Smax = boundary_Smax_loss(model, tb, vb, S_max, r=r, q=q, K=K, T=T, v_max=v_max)

        # --- Bordi v (Neumann soft) ---
        Sb = K * torch.exp((2*torch.rand(2048, device=device) - 1.0) * 1.5)
        tb2 = torch.rand_like(Sb) * T
        loss_v0   = boundary_v_neumann_loss(model, Sb, tb2, v_val=0.0, K=K, T=T, v_max=v_max)
        loss_vmax = boundary_v_neumann_loss(model, Sb, tb2, v_val=v_max, K=K, T=T, v_max=v_max)

        # --- Pesi (parsimoniosi) ---
        loss = (1.0*loss_pde + 10.0*loss_T + 1.0*loss_S0 + 1.0*loss_Smax + 0.1*loss_v0 + 0.1*loss_vmax)

        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        if ep % 500 == 0:
            print(f"[ep {ep}] total={loss.item():.4e} | pde={loss_pde.item():.4e} T={loss_T.item():.4e} S0={loss_S0.item():.4e} Smax={loss_Smax.item():.4e}")

    return model