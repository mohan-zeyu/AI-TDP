"""
芯片热分布预测 —— 基于 PINN (Physics-Informed Neural Network) + DeepXDE
=====================================================================

思路：
  1. 用有限差分法 (FDM) 生成一块芯片的"真实"稳态温度场（模拟红外热像仪数据）
  2. 从中采样 N 个传感器读数（模拟 2-3 个或更多温度传感器）
  3. 用 DeepXDE 训练 PINN：loss = PDE残差 + 边界条件 + 传感器数据
  4. 对比 PINN 预测 vs 真实温度场

物理方程（二维稳态热传导）：
  k (∂²T/∂x² + ∂²T/∂y²) + Q(x,y) = 0

边界条件（Dirichlet）：
  T|boundary = T_ambient

依赖：  pip install deepxde torch numpy matplotlib scipy
        （DeepXDE 后端使用 PyTorch，运行前设置环境变量）
运行：  DDE_BACKEND=pytorch python chip_thermal_pinn.py
"""

import os
os.environ["DDE_BACKEND"] = "pytorch"   # 必须在 import dde 之前

import numpy as np
import matplotlib.pyplot as plt
import deepxde as dde

# ── 全局参数 ─────────────────────────────────────────────

# 芯片尺寸 (归一化到 [0,1] x [0,1])
LX, LY = 1.0, 1.0
NX, NY = 64, 64  # FDM 网格分辨率

# 物理参数
K_THERMAL = 150.0    # 导热系数 W/(m·K)，硅的量级
T_AMBIENT = 25.0     # 环境温度 °C

# 热源定义：每个热源 = (x_center, y_center, intensity, sigma)
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
ADAM_EPOCHS = 100     # Adam 粗调轮数
LR = 1e-3

# loss 权重: [PDE, BC, sensor_data]
LOSS_WEIGHTS = [1, 5, 10]


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 第一步：生成"真实"温度场（有限差分法求解）
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def heat_source(x, y):
    """计算位置 (x,y) 处的总热源强度 Q(x,y) —— NumPy 版"""
    Q = np.zeros_like(x)
    for cx, cy, intensity, sigma in HEAT_SOURCES:
        Q += intensity * np.exp(-((x - cx)**2 + (y - cy)**2) / (2 * sigma**2))
    return Q


def solve_fdm():
    """用 Jacobi 迭代（向量化）求解二维稳态热传导方程"""
    dx = LX / (NX - 1)
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
# 第二步：DeepXDE —— PDE 定义 + 网络 + 训练
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def heat_source_backend(x, y):
    """热源函数 —— PyTorch 后端版本，供 DeepXDE 的 PDE 残差使用"""
    import torch
    Q = torch.zeros_like(x)
    for cx, cy, intensity, sigma in HEAT_SOURCES:
        Q += intensity * torch.exp(-((x - cx)**2 + (y - cy)**2) / (2 * sigma**2))
    return Q


def pde(x, y):
    """
    二维稳态热传导 PDE 残差: k(∂²T/∂x² + ∂²T/∂y²) + Q = 0

    DeepXDE 自动用 autograd 计算二阶导数，
    替代了手写版本中 ~40 行的 torch.autograd.grad 调用。

    参数:
        x: 输入坐标 (N, 2)，x[:,0]=x坐标, x[:,1]=y坐标
        y: 网络输出 T(x,y) (N, 1)
    返回:
        PDE 残差 (N, 1)，训练目标是让它趋近于 0
    """
    dy_xx = dde.grad.hessian(y, x, i=0, j=0)   # ∂²T/∂x²
    dy_yy = dde.grad.hessian(y, x, i=1, j=1)   # ∂²T/∂y²
    Q = heat_source_backend(x[:, 0:1], x[:, 1:2])

    # 归一化残差，避免 loss 量级过大
    Q_max = max(s[2] for s in HEAT_SOURCES)
    return (K_THERMAL * (dy_xx + dy_yy) + Q) / Q_max


def train():
    print("=" * 60)
    print("  芯片热分布 PINN 预测 (DeepXDE)")
    print("=" * 60)

    # ── 1. 生成真实数据 ──
    print("\n[1/4] 用 FDM 生成真实温度场 ...")
    x, y, X, Y, T_true = solve_fdm()
    print(f"  温度范围: {T_true.min():.1f}°C ~ {T_true.max():.1f}°C")

    # ── 2. 采样传感器 ──
    print(f"\n[2/4] 采样 {N_SENSORS} 个传感器 (均匀网格) ...")
    sx, sy, st = sample_sensors(x, y, T_true, N_SENSORS)
    for i in range(N_SENSORS):
        print(f"  传感器 {i+1}: ({sx[i]:.3f}, {sy[i]:.3f}) → {st[i]:.2f}°C")

    # ── 3. 构建 DeepXDE 问题 ──
    print(f"\n[3/4] 构建 DeepXDE 模型 & 训练 ...")

    # 3a. 几何域: [0,1] × [0,1] 矩形
    geom = dde.geometry.Rectangle([0, 0], [LX, LY])

    # 3b. 边界条件 (Dirichlet: T = T_AMBIENT on all boundaries)
    bc = dde.icbc.DirichletBC(
        geom,
        lambda x: T_AMBIENT,
        lambda _, on_boundary: on_boundary,
    )

    # 3c. 传感器观测数据 (PointSetBC: 作为 pseudo-BC 加入 loss)
    sensor_xy = np.column_stack([sx, sy])
    sensor_T = st.reshape(-1, 1)
    observe = dde.icbc.PointSetBC(sensor_xy, sensor_T, component=0)

    # 3d. 组装 PDE 问题
    data = dde.data.PDE(
        geom,
        pde,
        [bc, observe],              # loss 项: [PDE残差, BC, 传感器]
        num_domain=N_COLLOCATION,    # 域内 collocation 点
        num_boundary=N_BOUNDARY,     # 边界点
        num_test=1000,               # 测试点
        anchors=sensor_xy,           # 确保传感器位置被包含在训练集中
    )

    # 3e. 神经网络: [2] → [64]*4 → [1], Tanh, Glorot 初始化
    net = dde.nn.FNN(
        [2] + [64] * 4 + [1],
        "tanh",
        "Glorot normal",
    )

    # 3f. 构建模型
    model = dde.Model(data, net)

    # ── 4. 训练 ──
    # 阶段 1: Adam 粗调
    model.compile("adam", lr=LR, loss_weights=LOSS_WEIGHTS)
    losshistory, train_state = model.train(
        iterations=ADAM_EPOCHS,
        display_every=500,
    )

    # 阶段 2: L-BFGS-B 精调（比纯 Adam 收敛更精确）
    print("\n  切换到 L-BFGS-B 精调 ...")
    model.compile("L-BFGS-B", loss_weights=LOSS_WEIGHTS)
    losshistory, train_state = model.train()

    # ── 5. 预测 ──
    print("\n[4/4] 生成预测温度场 ...")
    xg = np.linspace(0, LX, NX)
    yg = np.linspace(0, LY, NY)
    Xg, Yg = np.meshgrid(xg, yg)
    xy_grid = np.column_stack([Xg.ravel(), Yg.ravel()])
    T_pred = model.predict(xy_grid).reshape(NY, NX)

    # 误差统计
    mae = np.mean(np.abs(T_pred - T_true))
    rmse = np.sqrt(np.mean((T_pred - T_true) ** 2))
    max_err = np.max(np.abs(T_pred - T_true))
    print(f"\n  MAE  = {mae:.3f}°C")
    print(f"  RMSE = {rmse:.3f}°C")
    print(f"  最大误差 = {max_err:.3f}°C")

    # 提取 loss 历史用于可视化
    loss_arr = np.array(losshistory.loss_train)
    history = {
        "total": loss_arr.sum(axis=1).tolist(),
        "pde":   loss_arr[:, 0].tolist(),
        "bc":    loss_arr[:, 1].tolist(),
        "data":  loss_arr[:, 2].tolist(),
    }

    return x, y, X, Y, T_true, T_pred, sx, sy, st, history


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 第三步：可视化
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def visualize(x, y, X, Y, T_true, T_pred, sx, sy, st, history):
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    vmin = min(T_true.min(), T_pred.min())
    vmax = max(T_true.max(), T_pred.max())

    # ── (a) 真实温度场 ──
    ax = axes[0, 0]
    im = ax.contourf(X, Y, T_true, levels=50, cmap="hot", vmin=vmin, vmax=vmax)
    ax.set_title("(a) Ground truth (FDM)", fontsize=13)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    plt.colorbar(im, ax=ax, label="T (°C)")

    # ── (b) PINN 预测温度场 + 传感器位置 ──
    ax = axes[0, 1]
    im = ax.contourf(X, Y, T_pred, levels=50, cmap="hot", vmin=vmin, vmax=vmax)
    ax.scatter(sx, sy, c="cyan", s=80, edgecolors="white", linewidths=1.5,
               zorder=5, label=f"Sensors (n={len(sx)})")
    for i in range(len(sx)):
        ax.annotate(f"{st[i]:.1f}°C", (sx[i], sy[i]),
                    textcoords="offset points", xytext=(8, 8),
                    fontsize=8, color="cyan", fontweight="bold")
    ax.legend(loc="lower right", fontsize=9)
    ax.set_title("(b) PINN prediction (DeepXDE)", fontsize=13)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    plt.colorbar(im, ax=ax, label="T (°C)")

    # ── (c) 绝对误差图 ──
    ax = axes[1, 0]
    err = np.abs(T_pred - T_true)
    im = ax.contourf(X, Y, err, levels=50, cmap="YlOrRd")
    ax.scatter(sx, sy, c="blue", s=60, edgecolors="white", linewidths=1,
               zorder=5, marker="^")
    ax.set_title(f"(c) Absolute error (MAE={np.mean(err):.2f}°C, Max={err.max():.2f}°C)", fontsize=13)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    plt.colorbar(im, ax=ax, label="|Error| (°C)")

    # ── (d) 训练曲线 ──
    ax = axes[1, 1]
    ax.semilogy(history["total"], label="Total Loss", linewidth=2, alpha=0.8)
    ax.semilogy(history["pde"], label="PDE Loss", linewidth=1, alpha=0.6)
    ax.semilogy(history["bc"], label="BC Loss", linewidth=1, alpha=0.6)
    ax.semilogy(history["data"], label="Data Loss", linewidth=1, alpha=0.6)
    ax.set_title("(d) Training convergence", fontsize=13)
    ax.set_xlabel("Step")
    ax.set_ylabel("Loss (log)")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    plt.suptitle(
        f"PINN Chip Thermal Prediction (DeepXDE)  |  Sensors: {len(sx)}  |  "
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
