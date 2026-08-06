# TDP v3 材料/热源几何模型发布说明

本分支整理了显式材料几何、分区热传导、界面热流与 Robin 边界条件相关的核心代码、代表性模型和效果图。它不是把某块树莓派的器件坐标写进通用骨干，而是将能力拆成：

- 通用骨干：从随机仿真板学习非均匀薄板稳态热方程；
- 共享 Adapter：编码材料区域、热源区域和分段边界；
- Board Card：保存具体板卡的几何、少量物理参数和 24 个 tokens。

## 当前应使用哪个模型

### 树莓派 4B：v3.1 推荐版

- 通用骨干：`models/v3/pretrain_v31_material_physics_512x12.pt`
- 推荐 Board Card：`benchmark/results/v31_shape_ablation_full_pi4b_k3_t24_600/pi4b_paint_card_k3.pt`
- 效果图：`benchmark/results/v31_shape_ablation_full_pi4b_k3_t24_600/test_preview.png`
- 形状消融图：`reports/model_comparison/v31_shape_loss_ablation/shape_ablation_comparison.png`

该组合使用 case16 半载训练 Board Card，以完全留出的 case17 满载测试，K=3、24 tokens、600 步。主要测试结果：SoC RMSE 0.588 °C、峰值误差 0.237 °C、90% 热区 IoU 0.791、95% 热区 IoU 0.806；95% 高温核心预测尺寸为 24×25 px，真实为 23×24 px。

### Orange Pi / 跨板：v3.2 实验版

- 全骨干续训：`models/v3/pretrain_v32_source_geometry_256x12_final.pt`
- 仅训练 Source Adapter：`models/v3/pretrain_v32_source_adapter_only_256x12.pt`
- 相应 Board Card、指标和预览图位于 `benchmark/results/cross_board_opi5pro_v32_*`。

v3.2 已将 `MaterialRegion` 与 `SourceRegion` 的语义和输入分离，但在严格的 case14 训练、case15 测试协议下尚未全面超过旧 v2。它用于保存新架构和跨板实验，不作为当前 Orange Pi 部署主模型。详见 `reports/v32_source_geometry_crossboard_result.md`。

## 物理与训练设计

稳态仿真求解：

```text
div(K(x,y) grad(theta)) - H(x,y) theta + Q(x,y) = 0
```

训练中随机化材料区和热源区的数量、位置、尺寸、旋转、矩形/椭圆形状、相对热导率 `K_rel`、垂向散热 `H_rel`、接触热阻 `Rc`、分段 Robin 边界、K=3 测点和噪声。材料界面使用谐均值通量与接触热阻，保留 25% 均匀材料回放，并随机丢弃部分几何描述。

真实板阶段冻结通用骨干和共享 Adapter，只优化 Board Card 中的少量 tokens、区域参数、热源幅值先验和边界参数。

## 复现实验入口

v3.1 512-board 预训练：

```powershell
python scripts/run_heterogeneous_pretrain.py `
  --config configs/pretrain_heterogeneous.yaml `
  --init-checkpoint models/v2/pretrain_v2.pt `
  --train-boards 512 --val-boards 64 --epochs 12 --batch-size 4 `
  --tag pretrain_v31_material_physics_512x12
```

树莓派推荐 Board Card：

```powershell
python scripts/run_v3_transfer.py `
  --protocol pi4b_paint `
  --board-geometry configs/board_material_geometry_v1.json `
  --checkpoint models/v3/pretrain_v31_material_physics_512x12.pt `
  --tokens 24 --steps 600 --shape-loss full `
  --output-dir benchmark/results/v31_shape_ablation_full_pi4b_k3_t24_600
```

v3.2 Source Adapter 消融：

```powershell
python scripts/run_heterogeneous_pretrain.py `
  --config configs/pretrain_source_geometry_v32.yaml `
  --init-checkpoint models/v3/pretrain_v31_material_physics_512x12.pt `
  --source-adapter-only --tag pretrain_v32_source_adapter_only_256x12
```

真实测量数据本身不在本发布包中重复提交；脚本读取 `configs/measurements_v3.yaml` 所指向的现有 `data/processed/real_v3`。

## 权重校验

| 文件 | 大小（字节） | SHA-256 |
|---|---:|---|
| `pretrain_v31_material_physics_512x12.pt` | 4,338,828 | `bff779c77f42b2d1034ead69c180ea63d3e7e8374557fee3b7d898d9aec2b71f` |
| `pretrain_v32_source_geometry_256x12_final.pt` | 4,380,526 | `b4c48036582d8fdd8279c00b4187cb9fa444e6f5d426af61c6c8fce45d6d7b75` |
| `pretrain_v32_source_adapter_only_256x12.pt` | 4,380,320 | `fe45b4db6b0cd37c9bd5e2a2b7a87f7a8918d9a4219c4c456b247a6295ae1c02` |

Board Card 不能脱离对应通用骨干单独部署。完整指标以各结果目录的 `metrics.json` 和 `reports/model_comparison/*/shape_ablation_metrics.json` 为准。

## 本次发布核验

- Python 语法编译检查通过；
- `tests/test_fdm.py` 中 10 项不加载 PyTorch 的数值测试通过；
- 当前机器导入 CUDA 版 PyTorch 时受到 Windows 页面文件限制（`WinError 1455`），其余依赖 PyTorch 的测试本次未能重新执行；训练完成时的 v3.2 回归记录为 38 项通过。
