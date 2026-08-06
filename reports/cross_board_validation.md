# TDP 跨板验证：Orange Pi 5 Pro

## 验证目的

验证在树莓派数据上完成设计和选择的通用骨干、共享材料适配器与 Board Card 机制，能否迁移到布局不同的 Orange Pi 5 Pro。该实验不使用 Orange Pi 测试集调参，也不与 IPTR 比较。

## 严格协议

- 目标板：Orange Pi 5 Pro（黑胶带版）。
- 训练集：`case14_half_a`，48 帧，仅用于拟合 Orange Pi Board Card。
- 测试集：`case15_full`，29 帧，全程不参与训练、损失选择或几何参数选择。
- 输入：K=3，分别为 `soc`、`warm_secondary`、`corner_cold`。
- 两组实验均使用当前统一的 178×112 裁切；旧的 191×120 结果不纳入像素级比较。
- 共享骨干和 Adapter 在真实板阶段冻结，只训练 24 个 board tokens、区域参数、边界参数和全局物理标量。
- 主要评价 SoC 热区 RMSE、峰值误差、热点位置误差和 90%/95% 热区 IoU；可信区域 RMSE 作为辅助指标。

## 比较方案

1. **旧 v2 基线**：`pretrain_v2.pt`，没有可用的显式材料几何 Adapter。
2. **单层封装几何**：v3.1 512-board 骨干，显式输入旋转后的 SoC 外封装以及其他材料/边界区域。
3. **外封装＋内核几何**：在外封装内增加一个小型发热内核区域；内核尺寸只从 case14 训练数据估计。

## case15 完全留出测试结果

| 指标 | 旧 v2 基线 | 单层封装几何 | 外封装＋内核几何 |
|---|---:|---:|---:|
| SoC RMSE (°C) | **1.129** | 1.476 | 1.426 |
| SoC MAE (°C) | **0.955** | 1.259 | 1.219 |
| 峰值误差 (°C) | 0.528 | 0.627 | **0.396** |
| 热点位置误差 (px) | 1.414 | **1.371** | 1.414 |
| 90% 热区 IoU | **0.786** | 0.700 | 0.681 |
| 95% 热区 IoU | **0.688** | 0.647 | 0.647 |
| 可信区域 RMSE (°C) | **2.863** | 3.193 | 3.200 |
| 全裁切 RMSE (°C) | **3.238** | 4.052 | 4.069 |

嵌套几何将峰值误差降低约 25.0%，证明材料/局部核心描述存在可迁移信号；但是相对旧 v2，SoC RMSE 增加约 26.3%，可信区域 RMSE 增加约 11.8%，90% 和 95% 热区 IoU 均下降。因此它不能取代旧基线，也不能据此宣称当前 v3.1 已获得完整跨板优势。

另外进行了“直接迁移树莓派完整热斑形状损失”的压力测试。它在 case15 上得到 SoC RMSE 1.550°C、峰值误差 2.816°C、热点位置误差 2.828 px、90% IoU 0.374 和 95% IoU 0.333，明显失败。树莓派上有效的形状损失权重不能原样迁移到 Orange Pi。

## 原因判断

从留出真值可见，Orange Pi 的高温核心只占旋转外封装的一小部分；外封装是较大的温暖材料区，发热内核则是较小的热源区。当前模型只接收材料几何描述，没有单独接收热源几何，容易把“高导热/高散热材料区域”和“发热功率支撑区域”混为一体。对大外封装施加形状约束会把高温平台错误扩张，这正是单层和嵌套方案 90%/95% 轮廓变大的原因。

这不是继续增加 board tokens 或把预训练板数从 512 直接扩大到 4096 就能可靠解决的问题。首先需要消除输入语义缺失。

## 下一版架构要求

- `MaterialRegion` 只描述形状、`K_rel`、`H_rel` 和 `Rc`。
- 新增独立的 `SourceRegion`，描述热源形状、位置、方向、相对功率及可学习幅值。
- 使用共享 `SourceGeometryAdapter` 编码任意矩形、椭圆和多边形热源；它与 Material Adapter 一起进入通用骨干。
- Board Card 分别保存材料区域参数、热源区域参数、分段边界参数和少量 tokens。
- 仿真预训练继续随机化材料区和热源区的重合、偏移、包含与完全分离关系，并保留均匀材料回放。
- 跨板选择必须继续使用 case14 训练、case15 留出的协议；只有新架构在未见 case15 的情况下超过旧 v2，才算真正改进。

## 产物

- 统一比较图：`reports/model_comparison/cross_board_opi5pro_final/shape_ablation_comparison.png`
- 完整指标：`reports/model_comparison/cross_board_opi5pro_final/shape_ablation_metrics.json`
- Orange Pi 几何质检：`reports/qc_new_sessions/case14_opi5pro_material_geometry.png`
- 旧 v2 基线：`benchmark/results/cross_board_opi5pro_legacy_v2_currentcrop_k3_t24_600/`
- 单层封装：`benchmark/results/cross_board_opi5pro_geometry_a_k3_t24_600/`
- 嵌套几何：`benchmark/results/cross_board_opi5pro_nested_geometry_k3_t24_600/`

结论：本轮完成了严格跨板验证，并识别出一个明确的架构瓶颈。当前 Orange Pi 推荐保留旧 v2 基线；v3.1 材料几何方案保留为实验分支，不升级为跨板主模型。
