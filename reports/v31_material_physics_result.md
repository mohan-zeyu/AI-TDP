# TDP v3.1：显式材料几何与物理约束训练结果

## 架构边界

本轮严格把通用规律与具体板卡分离：

- 通用骨干约 1.08M 参数，和共享的 Material/Boundary Adapter 一起只在随机仿真板上训练。
- Material Adapter 接收任意位置、尺寸、角度的矩形或椭圆区域，区域属性为相对面内热导率 `K_rel`、相对垂向散热系数 `H_rel`、界面接触热阻 `Rc`。
- Boundary Adapter 接收直线边界段 `[p0,p1,n,log(gamma),type]`；矩形板和多边形板均可用若干线段表达。
- 真实树莓派阶段冻结骨干和两个共享 Adapter，只训练 24 个 board tokens、每区 3 个材料参数、分段边界系数及两个全局物理标量。
- 树莓派器件坐标只存在 Board Card 配置中，不进入通用骨干权重。

## 物理模型

稳态仿真满足：

`div(K(x,y) grad(theta)) - H(x,y) theta + Q(x,y) = 0`

并显式使用：

- 材料界面谐均值有限体积通量；
- 有限接触热阻 `Rc` 的串联面热阻；
- 理想界面的热流连续损失，以及 `Rc > 0` 时的温度跳跃条件；
- `-n·K grad(theta) = gamma theta` Robin 外边界；
- 四边独立 `gamma` 与随机局部接触段；
- 75% 异质材料板、25% 均匀材料回放；
- 材料数量、位置、尺寸、角度、矩形/椭圆形状、热源相对位置、K/H/Rc、K=3 测点与噪声随机化；
- 15% 材料区域描述随机丢弃。

## 训练产物

- 通用检查点：`models/v3/pretrain_v31_material_physics_512x12.pt`
- 规模：512 个训练板、64 个验证板、每板 3 个状态、12 epoch、batch size 4。
- 最佳检查点：epoch 8；末轮没有覆盖最佳权重。
- 树莓派 Board Card：`benchmark/results/v31_material_physics512_pi4b_k3_t24_600/pi4b_paint_card_k3.pt`
- Board Card：24×128 tokens、3 个材料区、6 个边界段，总文件约 15.5 KB。
- 真实训练：case16 半载；测试：训练中未使用的 case17 满载；K=3。

## 与上一版 512-board 异质先导模型比较

| 指标 | 上一版 | v3.1 | 变化 |
|---|---:|---:|---:|
| SoC RMSE | 1.287 °C | 1.195 °C | -7.1% |
| SoC MAE | 1.026 °C | 0.844 °C | -17.8% |
| 峰值误差 | 0.735 °C | 0.353 °C | -51.9% |
| 热点位置误差 | 2.553 px | 2.007 px | -21.4% |
| 90% 热区 IoU | 0.754 | 0.793 | +0.039 |
| 95% 热区 IoU | 0.637 | 0.680 | +0.043 |
| 全裁剪 RMSE | 3.269 °C | 2.642 °C | -19.2% |
| 可信区 RMSE | 2.012 °C | 2.097 °C | +4.2% |

本项目以热斑重建为主要目标，因此 v3.1 在 SoC、峰值、定位和热区 IoU 上的同步改善比可信区 RMSE 的小幅退化更重要。95% 热区预测尺寸从 21×20 px 改善为 21×21 px，真实为 23×24 px；轮廓已更接近方形，但仍略小、略圆。

## 后续训练建议

下一轮不应直接扩大到 4096 板。512×12 的第 10 轮后出现少数高数据损失批次，应先把界面损失最大权重从 `3e-4` 降到约 `1e-4`，把物理 ramp 延长到 25–30 epoch，并缓存随机仿真板。稳定后再运行 1024–2048 板消融；选模继续以 SoC RMSE、峰值误差、热点位置和 90/95% IoU 为主。

## 第二轮：1024 板稳定性消融

第二轮从 v3.1 512-board 最佳检查点热启动，使用 1024 个训练板、128 个验证板、30 epoch。界面损失上限降为 `1e-4`，边界损失上限降为 `5e-5`，PDE 损失上限降为 `3e-4`，并用 25 epoch ramp。训练耗时约 57.7 分钟，过程没有出现上一轮式的持续失稳；最佳综合验证 checkpoint 位于 epoch 3。

真实板仍使用 case16 训练、case17 测试、K=3、24 tokens。结果与 v3.1 512-board 主模型比较如下：

| 指标 | v3.1 512-board | 第二轮 1024-board |
|---|---:|---:|
| SoC RMSE | **1.195 °C** | 1.712 °C |
| 峰值误差 | **0.353 °C** | 0.541 °C |
| 热点位置误差 | **2.007 px** | 3.040 px |
| 90% 热区 IoU | **0.793** | 0.788 |
| 95% 热区 IoU | 0.680 | **0.694** |
| 可信区 RMSE | **2.097 °C** | 2.170 °C |

1024-board 版本只在 95% 极高温核心 IoU 上小幅提高，但 SoC 温度精度、峰值、定位、90% 轮廓和视觉方形度均退化，因此不替换 512-board 主模型。该结果再次说明扩大随机分布并不自动提高特定真实板能力。

第二轮产物：

- 通用检查点：`models/v3/pretrain_v31_material_physics_round2_1024x30.pt`
- Board Card：`benchmark/results/v31_round2_1024x30_pi4b_k3_t24_600/pi4b_paint_card_k3.pt`
- 对比图：`reports/model_comparison/v31_round2_1024_vs_512/hotspot_shape_comparison.png`

同时修正了训练脚本的一处评估一致性问题：后续 K-scaling 将重新加载最佳 checkpoint 后计算，不再误报末轮内存模型的曲线。本轮已生成的 K-scaling 属于末轮模型，不能当作 epoch 3 checkpoint 的曲线；真实板结果则确实来自保存的 epoch 3 checkpoint。

## 显式热斑形状损失消融

固定当前最佳 `pretrain_v31_material_physics_512x12.pt`，冻结骨干和共享 Adapter，保持 case16 训练、case17 测试、K=3、24 tokens 和相同随机种子，仅改变 Board Card 损失：

- A：原始数据、PDE、边界和界面损失；
- B：A + `2.0 * L_soc`；
- C：B + `0.1 * L_edge + 0.1 * L_plateau + 0.05 * L_peak`。

`L_soc` 使用显式 SoC 材料区内且不属于验证 patch 的像素；`L_edge` 比较材料四边内外约 2 px 成对点的温度跳变；`L_plateau` 比较区域内去均值后的温度形状；`L_peak` 使用可微平滑峰值。所有真实形状监督均来自 case16，未从 case17 提取边缘。

| 指标 | A 基线 | B 仅 SoC | C 完整形状 |
|---|---:|---:|---:|
| SoC RMSE | 1.195 °C | **0.564 °C** | 0.588 °C |
| 峰值误差 | 0.353 °C | 0.858 °C | **0.237 °C** |
| 热点位置误差 | **2.007 px** | 3.828 px | 3.030 px |
| 90% 热区 IoU | **0.793** | 0.731 | 0.791 |
| 95% 热区 IoU | 0.680 | 0.778 | **0.806** |
| 95% 预测尺寸 | 21×21 px | 26×26 px | **24×25 px** |
| 95% 真实尺寸 | 23×24 px | 23×24 px | 23×24 px |

B 说明只优化区域平均误差会扩大并平均化热点，虽然 SoC RMSE 最低，但峰值和定位明显退化。C 的边缘、平台和峰值项纠正了大部分副作用：95% 核心轮廓明显更方、更接近真实尺寸，同时获得最低峰值误差。其不足是 90% 外层轮廓仍偏大，热点位置误差高于 A。

在“热斑形状优先”的项目目标下，C 作为新的推荐 Board Card；A 继续作为热点定位更准的保守备份。

- C 组 Board Card：`benchmark/results/v31_shape_ablation_full_pi4b_k3_t24_600/pi4b_paint_card_k3.pt`
- A/B/C 对比图：`reports/model_comparison/v31_shape_loss_ablation/shape_ablation_comparison.png`
- 完整指标：`reports/model_comparison/v31_shape_loss_ablation/shape_ablation_metrics.json`
