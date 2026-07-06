import io
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from scipy.sparse import lil_matrix, csr_matrix
from scipy.sparse.linalg import splu
import matplotlib.pyplot as plt
from dataclasses import dataclass
import time
import json
import sys

# ═══════════════════════════════════════════════════════════════════════
# 1. Config
# ═══════════════════════════════════════════════════════════════════════
@dataclass
class Config:
    # Physics / geometry
    Lx: float = 1.0
    Ly: float = 1.0
    k:  float = 1.0       # thermal conductivity
    h:  float = 5.0       # effective convection coefficient
    T_amb: float = 25.0
    nx: int = 40
    ny: int = 40
    # Dataset
    n_train: int = 4096
    n_val:   int = 32
    k_min:   int = 3
    k_max:   int = 12
    n_query: int = 384
    # Model
    d_model:   int = 128
    n_heads:   int = 4
    n_self:    int = 2
    n_cross:   int = 3
    n_fourier: int = 32
    fourier_sigma: float = 5.0
    # Training
    epochs: int = 200
    batch_size: int = 32
    lr: float = 1e-3
    weight_decay: float = 1e-5
    # PDE residual loss
    n_colloc: int = 256             # collocation points per sample
    lambda_pde_max: float = 0.005    # max PDE-loss weight
    pde_warmup_epochs: int = 10     # epochs of pure data loss before ramp
    pde_ramp_epochs: int = 20       # epochs to linearly ramp lambda 0 -> max

CFG = Config()

# ═══════════════════════════════════════════════════════════════════════
# 2. FDM ground-truth solver (linear system; LU-factor once, solve many)
# ═══════════════════════════════════════════════════════════════════════
def build_fdm_system(cfg):
    nx, ny = cfg.nx, cfg.ny
    dx = cfg.Lx / (nx - 1)
    dy = cfg.Ly / (ny - 1)
    N = nx * ny
    A = lil_matrix((N, N))
    boundary = np.zeros(N, dtype=bool)
    for j in range(ny):
        for i in range(nx):
            idx = j * nx + i
            if i == 0 or i == nx - 1 or j == 0 or j == ny - 1:
                A[idx, idx] = 1.0
                boundary[idx] = True
            else:
                A[idx, idx]      = -2*cfg.k/dx**2 - 2*cfg.k/dy**2 - cfg.h
                A[idx, idx - 1]  = cfg.k / dx**2
                A[idx, idx + 1]  = cfg.k / dx**2
                A[idx, idx - nx] = cfg.k / dy**2
                A[idx, idx + nx] = cfg.k / dy**2
    return splu(csr_matrix(A).tocsc()), boundary

def solve_fdm(lu, boundary, Q, cfg):
    """k∇²T - h(T-T_amb) + Q = 0, Dirichlet T=T_amb on boundary."""
    N = cfg.nx * cfg.ny
    b = np.empty(N)
    Qf = Q.flatten()
    b[boundary] = cfg.T_amb
    b[~boundary] = -cfg.h * cfg.T_amb - Qf[~boundary]
    return lu.solve(b).reshape(cfg.ny, cfg.nx)

def random_heat_source(cfg):
    n_spots = np.random.randint(1, 5)
    x = np.linspace(0, cfg.Lx, cfg.nx)
    y = np.linspace(0, cfg.Ly, cfg.ny)
    X, Y = np.meshgrid(x, y)
    Q = np.zeros_like(X)
    for _ in range(n_spots):
        cx  = np.random.uniform(0.15, cfg.Lx - 0.15)
        cy  = np.random.uniform(0.15, cfg.Ly - 0.15)
        sig = np.random.uniform(0.05, 0.15)
        amp = np.random.uniform(100, 400)
        Q += amp * np.exp(-((X-cx)**2 + (Y-cy)**2) / (2*sig**2))
    return Q

def generate_dataset(n, lu, boundary, cfg, tag=''):
    Ts, Qs = [], []
    t0 = time.time()
    for i in range(n):
        Q = random_heat_source(cfg)
        T = solve_fdm(lu, boundary, Q, cfg)
        Ts.append(T.astype(np.float32))
        Qs.append(Q.astype(np.float32))
    print(f"  [{tag}] generated {n} scenarios in {time.time()-t0:.1f}s")
    return np.stack(Ts), np.stack(Qs)

# ═══════════════════════════════════════════════════════════════════════
# 3. Model
# ═══════════════════════════════════════════════════════════════════════
class FourierFeatures(nn.Module):
    """Random Fourier features: (x, y) → [sin(2π Bx), cos(2π Bx)]"""
    def __init__(self, input_dim=2, n_features=32, sigma=5.0):
        super().__init__()
        self.register_buffer('B', torch.randn(input_dim, n_features) * sigma)
    def forward(self, x):
        xB = 2 * np.pi * (x @ self.B)
        return torch.cat([torch.sin(xB), torch.cos(xB)], dim=-1)

class SensorEncoder(nn.Module):
    def __init__(self, d_model, n_fourier, sigma):
        super().__init__()
        self.ff = FourierFeatures(2, n_fourier, sigma)
        self.mlp = nn.Sequential(
            nn.Linear(2*n_fourier + 1, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )
    def forward(self, sensors):
        # sensors: (B, K, 3) = (x, y, T)
        xy = sensors[..., :2]
        T  = sensors[..., 2:3]
        return self.mlp(torch.cat([self.ff(xy), T], dim=-1))

class QueryEncoder(nn.Module):
    def __init__(self, d_model, n_fourier, sigma):
        super().__init__()
        self.ff = FourierFeatures(2, n_fourier, sigma)
        self.mlp = nn.Sequential(
            nn.Linear(2*n_fourier, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )
    def forward(self, q):
        return self.mlp(self.ff(q))

class AttentionBlock(nn.Module):
    """Pre-norm transformer block; self-attn if cross=False else cross-attn."""
    def __init__(self, d_model, n_heads, ff_mult=4, cross=False):
        super().__init__()
        self.cross = cross
        self.ln_q = nn.LayerNorm(d_model)
        self.ln_kv = nn.LayerNorm(d_model) if cross else None
        self.attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.ln2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, ff_mult*d_model),
            nn.GELU(),
            nn.Linear(ff_mult*d_model, d_model),
        )
    def forward(self, x, ctx=None, mask=None):
        q = self.ln_q(x)
        if self.cross:
            kv = self.ln_kv(ctx)
            a, _ = self.attn(q, kv, kv, key_padding_mask=mask)
        else:
            a, _ = self.attn(q, q, q, key_padding_mask=mask)
        x = x + a
        return x + self.ffn(self.ln2(x))

class ThermalOperator(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.sens_enc = SensorEncoder(cfg.d_model, cfg.n_fourier, cfg.fourier_sigma)
        self.query_enc = QueryEncoder(cfg.d_model, cfg.n_fourier, cfg.fourier_sigma)
        self.self_blocks  = nn.ModuleList([
            AttentionBlock(cfg.d_model, cfg.n_heads, cross=False)
            for _ in range(cfg.n_self)
        ])
        self.cross_blocks = nn.ModuleList([
            AttentionBlock(cfg.d_model, cfg.n_heads, cross=True)
            for _ in range(cfg.n_cross)
        ])
        self.head = nn.Sequential(
            nn.LayerNorm(cfg.d_model),
            nn.Linear(cfg.d_model, cfg.d_model),
            nn.GELU(),
            nn.Linear(cfg.d_model, 1),
        )
    def forward(self, sensors, queries, sensor_mask=None):
        s = self.sens_enc(sensors)
        for blk in self.self_blocks:
            s = blk(s, mask=sensor_mask)
        q = self.query_enc(queries)
        for blk in self.cross_blocks:
            q = blk(q, ctx=s, mask=sensor_mask)
        return self.head(q).squeeze(-1)   # (B, Q)

device = torch.device("mps" if torch.mps.is_available() else "cpu")
ckpt = torch.load("thermal_operator.pt", map_location=device, weights_only=False)
CFG_loaded = Config(**ckpt['config'])
T_mean = ckpt['T_mean']
T_std  = ckpt['T_std']

model = ThermalOperator(CFG_loaded).to(device)
model.load_state_dict(ckpt['model_state'])
model.eval()
print('Loaded.')

lu, boundary = build_fdm_system(CFG)

# ═══════════════════════════════════════════════════════════════════════
# Test: 4 sensors at rectangle corners
# ═══════════════════════════════════════════════════════════════════════
@torch.no_grad()
def test_corner_sensors(model, lu, boundary, cfg, T_mean, T_std, device,
                        inset=0.2, seed=11111, save_path='corner_test.png'):
    """
    Test the model with exactly 4 sensors at the rectangle corners.

    NOTE: sensors placed exactly at (0,0), (1,0), (0,1), (1,1) would lie on
    the Dirichlet boundary (T=T_amb by construction), giving trivial readings.
    Default inset=0.05 samples the interior. Pass inset=0.0 for literal corners.
    """
    # Fresh test scenario (seeded so it's reproducible but not in training set)
    np.random.seed(seed)
    Q_test = random_heat_source(cfg)
    T_test = solve_fdm(lu, boundary, Q_test, cfg).astype(np.float32)

    # 4 corner positions
    corners = np.array([
        [0.0 + inset,       0.0 + inset      ],  # bottom-left
        [cfg.Lx - inset,    0.0 + inset      ],  # bottom-right
        [0.0 + inset,       cfg.Ly - inset   ],  # top-left
        [cfg.Lx - inset,    cfg.Ly - inset   ],  # top-right
        [0.5, cfg.Ly - 0.25],
        [0.5, 0.3]
        ]
    , dtype=np.float32)

    # Read corner temps from ground truth (nearest grid point)
    xg = np.linspace(0, cfg.Lx, cfg.nx)
    yg = np.linspace(0, cfg.Ly, cfg.ny)
    sT = np.array([
        T_test[int(np.argmin(np.abs(yg - sy))),
               int(np.argmin(np.abs(xg - sx)))]
        for sx, sy in corners
    ], dtype=np.float32)

    # Pack input for model
    sT_norm = (sT - T_mean) / T_std
    sensors = torch.from_numpy(
        np.concatenate([corners, sT_norm[:, None]], axis=-1)
    ).unsqueeze(0).to(device)                                  # (1, 4, 3)

    X, Y = np.meshgrid(xg.astype(np.float32), yg.astype(np.float32))
    queries = torch.from_numpy(
        np.stack([X.flatten(), Y.flatten()], axis=-1)
    ).unsqueeze(0).to(device)                                  # (1, nx*ny, 2)

    model.eval()
    pred = (model(sensors, queries).cpu().numpy()[0] * T_std + T_mean).reshape(cfg.ny, cfg.nx)
    err = np.abs(pred - T_test)
    rmse = float(np.sqrt(((pred - T_test) ** 2).mean()))

    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.3))
    vmin, vmax = T_test.min(), T_test.max()

    ax = axes[0]
    im = ax.imshow(T_test, extent=[0, cfg.Lx, 0, cfg.Ly], origin='lower',
                   cmap='hot', vmin=vmin, vmax=vmax)
    ax.scatter(corners[:, 0], corners[:, 1], c='cyan', s=90,
               edgecolors='white', linewidths=1.5, zorder=5)
    for (sx, sy), t in zip(corners, sT):
        ax.annotate(f'{t:.1f}°C', (sx, sy), textcoords='offset points',
                    xytext=(8, 6), fontsize=9, color='cyan', fontweight='bold')
    ax.set_title('FDM truth + 4 corner sensors'); ax.set_xlabel('x'); ax.set_ylabel('y')
    plt.colorbar(im, ax=ax, fraction=0.046, label='T (°C)')

    ax = axes[1]
    im = ax.imshow(pred, extent=[0, cfg.Lx, 0, cfg.Ly], origin='lower',
                   cmap='hot', vmin=vmin, vmax=vmax)
    ax.scatter(corners[:, 0], corners[:, 1], c='cyan', s=90,
               edgecolors='white', linewidths=1.5, zorder=5)
    ax.set_title(f'Prediction  RMSE={rmse:.2f}°C'); ax.set_xlabel('x')
    plt.colorbar(im, ax=ax, fraction=0.046, label='T (°C)')

    ax = axes[2]
    im = ax.imshow(err, extent=[0, cfg.Lx, 0, cfg.Ly], origin='lower', cmap='YlOrRd')
    ax.scatter(corners[:, 0], corners[:, 1], c='blue', s=60, marker='^',
               edgecolors='white', linewidths=1, zorder=5)
    ax.set_title(f'|Error|  MAE={err.mean():.2f}°C  Max={err.max():.2f}°C')
    ax.set_xlabel('x')
    plt.colorbar(im, ax=ax, fraction=0.046, label='|ΔT| (°C)')

    plt.suptitle(f'4-corner-sensor test  (inset={inset}, seed={seed})', fontsize=13, y=1.02)
    plt.tight_layout()

    print(f'Corner readings      : {sT.round(2).tolist()} °C')
    print(f'Truth field range    : [{T_test.min():.2f}, {T_test.max():.2f}] °C')
    print(f'Prediction RMSE/MAE  : {rmse:.3f} / {err.mean():.3f} °C    Max|err|: {err.max():.3f} °C')

    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)

    return buf

def main():
    raw = sys.stdin.read()
    payload = json.loads(raw)

    seed = payload.get("seed", 11111)
    inset = payload.get("inset", 0.2)

    # For seed 12345, there will be huge errors.
    buf = test_corner_sensors(model, lu, boundary, CFG, T_mean, T_std, device, seed=seed, inset=inset)
    sys.stdout.buffer.write(buf)

if __name__ == "__main__" :
    main()