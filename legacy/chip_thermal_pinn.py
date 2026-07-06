"""
芯片热分布预测 —— 基于 PINN (Physics-Informed Neural Network)
=============================================================

思路：
  1. 用有限差分法 (FDM) 生成一块芯片的"真实"稳态温度场（模拟红外热像仪数据）
  2. 从中采样 N 个传感器读数（模拟 2-3 个或更多温度传感器）
  3. 训练 PINN：loss = 物理残差(PDE) + 传感器数据误差 + 边界条件误差
  4. 对比 PINN 预测 vs 真实温度场

物理方程（二维稳态热传导）：
  k (∂²T/∂x² + ∂²T/∂y²) + Q(x,y) = 0

边界条件（对流冷却，Robin BC 简化为固定温度）：
  T|boundary = T_ambient

依赖：  pip install torch numpy matplotlib
运行：  python chip_thermal_pinn.py
"""

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import StepLR

# ── 全局参数 ─────────────────────────────────────────────
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 芯片尺寸 (归一化到 [0,1] x [0,1])
LX, LY = 1.0, 1.0
NX, NY = 64, 64  # FDM 网格分辨率

# 物理参数
K_THERMAL = 150.0   # 导热系数 W/(m·K)，硅的量级
T_AMBIENT = 25.0     # 环境温度 °C

# 热源定义：每个热源 = (x_center, y_center, intensity, sigma)
# 模拟芯片上 CPU核心、GPU区域、I/O 控制器等不同发热单元
# intensity 已校准，使峰值温度落在 ~35-40°C（环境 25°C，ΔT ≈ 10-15°C）
HEAT_SOURCES = [
    (0.3, 0.6, 1.8e5, 0.08),   # 主处理核心（高功率）
    (0.7, 0.7, 1.1e5, 0.06),   # GPU 区域
    (0.5, 0.3, 6.5e4, 0.07),   # 内存控制器
    (0.2, 0.2, 2.0e4, 0.05),   # I/O 模块（低功率）
]

# 传感器数量（核心变量——你可以改成 2、3、5 看效果差异）
N_SENSORS = 5

# PINN 训练参数
N_COLLOCATION = 2000   # PDE 残差采样点数
N_BOUNDARY = 200       # 边界采样点数
EPOCHS = 5000          # 正式训练建议 5000+，你可以先试 2000 看趋势
LR = 1e-3

# loss 权重（PDE 已归一化到 O(1)，Data/BC 的 MSE 量级 ~O(10)）
W_PDE = 1.0
W_DATA = 10.0     # 传感器数据权重要大，因为点少但很重要
W_BC = 5.0


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 第一步：生成"真实"温度场（有限差分法求解）
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def heat_source(x, y):
    """计算位置 (x,y) 处的总热源强度 Q(x,y)"""
    Q = np.zeros_like(x)
    for cx, cy, intensity, sigma in HEAT_SOURCES:
        Q += intensity * np.exp(-((x - cx)**2 + (y - cy)**2) / (2 * sigma**2))
    return Q


def solve_fdm():
    """用 Jacobi 迭代（向量化）求解二维稳态热传导方程"""
    dx = LX / (NX - 1)
    dy = LY / (NY - 1)
    x = np.linspace(0, LX, NX)
    y = np.linspace(0, LY, NY)
    X, Y = np.meshgrid(x, y)

    T = np.full((NY, NX), T_AMBIENT, dtype=np.float64)
    Q = heat_source(X, Y)
    src = (dx**2 / K_THERMAL) * Q

    for it in range(20000):
        T_old = T.copy()
        T[1:-1, 1:-1] = 0.25 * (
            T_old[2:, 1:-1] + T_old[:-2, 1:-1] +
            T_old[1:-1, 2:] + T_old[1:-1, :-2] +
            src[1:-1, 1:-1]
        )
        T[0, :] = T_AMBIENT
        T[-1, :] = T_AMBIENT
        T[:, 0] = T_AMBIENT
        T[:, -1] = T_AMBIENT

        if np.max(np.abs(T - T_old)) < 1e-6:
            print(f"  FDM 收敛于第 {it+1} 次迭代")
            break

    return x, y, X, Y, T


def sample_sensors(x, y, T_field, n_sensors):
    """从真实温度场中均匀网格采样传感器读数（避开边界）"""
    rng = np.random.RandomState(42)
    margin = 0.1

    # 均匀网格布局：找到最接近正方形的 nx × ny >= n_sensors
    ny = int(np.round(np.sqrt(n_sensors)))
    nx = int(np.ceil(n_sensors / ny))

    gx = np.linspace(margin, LX - margin, nx)
    gy = np.linspace(margin, LY - margin, ny)
    Gx, Gy = np.meshgrid(gx, gy)
    sx = Gx.ravel()[:n_sensors]
    sy = Gy.ravel()[:n_sensors]

    # 双线性插值获取温度
    from scipy.interpolate import RegularGridInterpolator
    interp = RegularGridInterpolator((y, x), T_field)
    st = interp(np.column_stack([sy, sx]))

    # 加一点噪声模拟真实传感器
    st += rng.normal(0, 0.2, n_sensors)

    return sx, sy, st


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 第二步：定义 PINN 网络
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class PINN(nn.Module):
    """
    输入: (x, y)  ∈ [0,1]²
    输出: T(x, y) 预测温度
    """
    def __init__(self, layers=None):
        super().__init__()
        if layers is None:
            layers = [2, 64, 64, 64, 64, 1]

        modules = []
        for i in range(len(layers) - 1):
            modules.append(nn.Linear(layers[i], layers[i+1]))
            if i < len(layers) - 2:
                modules.append(nn.Tanh())  # Tanh 对 PINN 很重要，因为需要高阶导数
        self.net = nn.Sequential(*modules)

        # Xavier 初始化
        for m in self.net:
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        return self.net(x)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 第三步：定义 Loss 函数
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def heat_source_torch(x, y):
    """PyTorch 版热源函数"""
    Q = torch.zeros_like(x)
    for cx, cy, intensity, sigma in HEAT_SOURCES:
        Q += intensity * torch.exp(-((x - cx)**2 + (y - cy)**2) / (2 * sigma**2))
    return Q


def compute_pde_residual(model, xy_colloc):
    """
    计算 PDE 残差: k(∂²T/∂x² + ∂²T/∂y²) + Q = 0
    利用 autograd 求二阶偏导
    """
    xy_colloc.requires_grad_(True)
    T_pred = model(xy_colloc)

    # 一阶导数
    grad_T = torch.autograd.grad(
        T_pred, xy_colloc,
        grad_outputs=torch.ones_like(T_pred),
        create_graph=True
    )[0]
    dT_dx = grad_T[:, 0:1]
    dT_dy = grad_T[:, 1:2]

    # 二阶导数
    dT_dxx = torch.autograd.grad(
        dT_dx, xy_colloc,
        grad_outputs=torch.ones_like(dT_dx),
        create_graph=True
    )[0][:, 0:1]

    dT_dyy = torch.autograd.grad(
        dT_dy, xy_colloc,
        grad_outputs=torch.ones_like(dT_dy),
        create_graph=True
    )[0][:, 1:2]

    Q = heat_source_torch(xy_colloc[:, 0:1], xy_colloc[:, 1:2])

    # 残差: k * (T_xx + T_yy) + Q = 0
    # 除以 Q_max 归一化，使残差量级 ~O(1)，避免 loss 爆炸
    Q_max = max(s[2] for s in HEAT_SOURCES)
    residual = (K_THERMAL * (dT_dxx + dT_dyy) + Q) / Q_max
    return residual


def compute_loss(model, xy_colloc, xy_bc, T_bc, xy_sensor, T_sensor):
    """总 loss = PDE 残差 + 边界条件 + 传感器数据"""
    # 1) PDE 残差
    res = compute_pde_residual(model, xy_colloc)
    loss_pde = torch.mean(res ** 2)

    # 2) 边界条件 loss
    T_bc_pred = model(xy_bc)
    loss_bc = torch.mean((T_bc_pred - T_bc) ** 2)

    # 3) 传感器数据 loss
    T_sensor_pred = model(xy_sensor)
    loss_data = torch.mean((T_sensor_pred - T_sensor) ** 2)

    total = W_PDE * loss_pde + W_BC * loss_bc + W_DATA * loss_data
    return total, loss_pde.item(), loss_bc.item(), loss_data.item()


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 第四步：训练
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def train():
    print("=" * 60)
    print("  芯片热分布 PINN 预测")
    print("=" * 60)

    # ── 生成真实数据 ──
    print("\n[1/4] 用 FDM 生成真实温度场 ...")
    x, y, X, Y, T_true = solve_fdm()
    T_max = T_true.max()
    T_min = T_true.min()
    print(f"  温度范围: {T_min:.1f}°C ~ {T_max:.1f}°C")

    # ── 采样传感器 ──
    print(f"\n[2/4] 采样 {N_SENSORS} 个传感器 ...")
    sx, sy, st = sample_sensors(x, y, T_true, N_SENSORS)
    for i in range(N_SENSORS):
        print(f"  传感器 {i+1}: ({sx[i]:.3f}, {sy[i]:.3f}) → {st[i]:.2f}°C")

    # ── 准备训练数据 ──
    print(f"\n[3/4] 训练 PINN (epochs={EPOCHS}) ...")

    # 传感器数据 → tensor
    xy_sensor = torch.tensor(np.column_stack([sx, sy]), dtype=torch.float32).to(DEVICE)
    T_sensor = torch.tensor(st.reshape(-1, 1), dtype=torch.float32).to(DEVICE)

    # 边界点
    rng = np.random.RandomState(0)
    n_per_side = N_BOUNDARY // 4
    bc_pts = []
    # 下边
    bc_pts.append(np.column_stack([rng.uniform(0, LX, n_per_side), np.zeros(n_per_side)]))
    # 上边
    bc_pts.append(np.column_stack([rng.uniform(0, LX, n_per_side), np.full(n_per_side, LY)]))
    # 左边
    bc_pts.append(np.column_stack([np.zeros(n_per_side), rng.uniform(0, LY, n_per_side)]))
    # 右边
    bc_pts.append(np.column_stack([np.full(n_per_side, LX), rng.uniform(0, LY, n_per_side)]))
    bc_pts = np.vstack(bc_pts)

    xy_bc = torch.tensor(bc_pts, dtype=torch.float32).to(DEVICE)
    T_bc = torch.full((xy_bc.shape[0], 1), T_AMBIENT, dtype=torch.float32).to(DEVICE)

    # 模型
    model = PINN().to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = StepLR(optimizer, step_size=2000, gamma=0.5)

    # ── 训练循环 ──
    history = {"total": [], "pde": [], "bc": [], "data": []}

    for epoch in range(1, EPOCHS + 1):
        model.train()
        optimizer.zero_grad()

        # 每轮重新采样 collocation 点（帮助覆盖更多区域）
        xy_c = torch.rand(N_COLLOCATION, 2, device=DEVICE)
        xy_c[:, 0] *= LX
        xy_c[:, 1] *= LY

        loss, l_pde, l_bc, l_data = compute_loss(
            model, xy_c, xy_bc, T_bc, xy_sensor, T_sensor
        )

        loss.backward()
        optimizer.step()
        scheduler.step()

        history["total"].append(loss.item())
        history["pde"].append(l_pde)
        history["bc"].append(l_bc)
        history["data"].append(l_data)

        if epoch % 500 == 0 or epoch == 1:
            print(f"  Epoch {epoch:5d} | Total: {loss.item():.4e} | "
                  f"PDE: {l_pde:.4e} | BC: {l_bc:.4e} | Data: {l_data:.4e}")

    # ── 预测 ──
    print("\n[4/4] 生成预测温度场 ...")
    model.eval()
    with torch.no_grad():
        xg = np.linspace(0, LX, NX)
        yg = np.linspace(0, LY, NY)
        Xg, Yg = np.meshgrid(xg, yg)
        xy_grid = torch.tensor(
            np.column_stack([Xg.ravel(), Yg.ravel()]),
            dtype=torch.float32
        ).to(DEVICE)
        T_pred = model(xy_grid).cpu().numpy().reshape(NY, NX)

    # 误差统计
    mae = np.mean(np.abs(T_pred - T_true))
    rmse = np.sqrt(np.mean((T_pred - T_true) ** 2))
    max_err = np.max(np.abs(T_pred - T_true))
    print(f"\n  MAE  = {mae:.3f}°C")
    print(f"  RMSE = {rmse:.3f}°C")
    print(f"  最大误差 = {max_err:.3f}°C")

    return x, y, X, Y, T_true, T_pred, sx, sy, st, history


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 第五步：可视化
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def visualize(x, y, X, Y, T_true, T_pred, sx, sy, st, history):
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    vmin = min(T_true.min(), T_pred.min())
    vmax = max(T_true.max(), T_pred.max())

    # ── (a) 真实温度场 ──
    ax = axes[0, 0]
    im = ax.contourf(X, Y, T_true, levels=50, cmap="hot", vmin=vmin, vmax=vmax)
    ax.set_title("(a) 真实温度场 (FDM ground truth)", fontsize=13)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    plt.colorbar(im, ax=ax, label="T (°C)")

    # ── (b) PINN 预测温度场 + 传感器位置 ──
    ax = axes[0, 1]
    im = ax.contourf(X, Y, T_pred, levels=50, cmap="hot", vmin=vmin, vmax=vmax)
    ax.scatter(sx, sy, c="cyan", s=80, edgecolors="white", linewidths=1.5,
               zorder=5, label=f"传感器 (n={len(sx)})")
    for i in range(len(sx)):
        ax.annotate(f"{st[i]:.1f}°C", (sx[i], sy[i]),
                    textcoords="offset points", xytext=(8, 8),
                    fontsize=8, color="cyan", fontweight="bold")
    ax.legend(loc="lower right", fontsize=9)
    ax.set_title("(b) PINN 预测温度场", fontsize=13)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    plt.colorbar(im, ax=ax, label="T (°C)")

    # ── (c) 绝对误差图 ──
    ax = axes[1, 0]
    err = np.abs(T_pred - T_true)
    im = ax.contourf(X, Y, err, levels=50, cmap="YlOrRd")
    ax.scatter(sx, sy, c="blue", s=60, edgecolors="white", linewidths=1,
               zorder=5, marker="^")
    ax.set_title(f"(c) 绝对误差 (MAE={np.mean(err):.2f}°C, Max={err.max():.2f}°C)", fontsize=13)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    plt.colorbar(im, ax=ax, label="|Error| (°C)")

    # ── (d) 训练曲线 ──
    ax = axes[1, 1]
    ax.semilogy(history["total"], label="Total Loss", linewidth=2, alpha=0.8)
    ax.semilogy(history["pde"], label="PDE Loss", linewidth=1, alpha=0.6)
    ax.semilogy(history["bc"], label="BC Loss", linewidth=1, alpha=0.6)
    ax.semilogy(history["data"], label="Data Loss", linewidth=1, alpha=0.6)
    ax.set_title("(d) 训练收敛曲线", fontsize=13)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss (log)")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    plt.suptitle(
        f"PINN 芯片热分布预测  |  传感器数: {len(sx)}  |  "
        f"RMSE: {np.sqrt(np.mean((T_pred - T_true)**2)):.2f}°C",
        fontsize=15, fontweight="bold", y=1.01
    )
    plt.tight_layout()
    plt.savefig("chip_thermal_pinn_result.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("\n图片已保存: chip_thermal_pinn_result.png")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

if __name__ == "__main__":
    results = train()
    visualize(*results)
