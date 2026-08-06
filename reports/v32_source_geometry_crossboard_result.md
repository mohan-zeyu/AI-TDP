# TDP v3.2：独立热源几何与跨板验证

## 本轮改动

v3.2 将材料区域和发热源区域从输入语义上彻底分开：

- `MaterialRegion = [位置, 尺寸, 角度, K_rel, H_rel, Rc, 形状]`；
- `SourceRegion = [位置, 尺寸, 角度, 相对功率, 形状]`；
- 新增所有板共享的 `SourceGeometryAdapter`，不在骨干中写死树莓派或 Orange Pi 坐标；
- Board Card 固定保存源区几何，只学习每个源区的少量 `log(relative amplitude)`；
- 仿真预训练显式随机化源区数量、形状、旋转、关断，以及源区与材料区的重合、偏移和分离；
- 旧 v3.1 检查点加载时 Source Adapter 零初始化，因此没有源描述时预测保持兼容；
- 真实板训练仍冻结共享模型，只训练 24 个 tokens、材料参数、源强先验和分段边界参数。

模型总参数约 1.08M；Source Adapter 约 9.0K 参数。回归测试为 38 项全部通过。

## 预训练

主消融使用 256 块随机仿真板、每板 3 个状态、12 轮、batch size 4，并从 `pretrain_v31_material_physics_512x12.pt` 热启动。

比较两种训练策略：

1. **全骨干续训**：允许通用骨干和三个 Adapter 一起更新；
2. **冻结骨干**：只训练新增的 9.0K Source Adapter 参数，防止已有跨板能力漂移。

冻结骨干版本的最佳检查点出现在 epoch 2，之后继续训练收益很小。说明当前瓶颈不只是训练轮数不足。

## 严格真实板协议

- 板卡：Orange Pi 5 Pro 黑胶带版；
- 训练：`case14_half_a`，48 帧；
- 测试：`case15_full`，29 帧，完全不参与训练、早停或参数选择；
- K=3：`soc`、`warm_secondary`、`corner_cold`；
- 裁切：统一 178×112；
- Board Card：24 tokens、600 步。

## case15 结果

| 指标 | 旧 v2 | v3.1 嵌套材料 | v3.2 全续训 | v3.2 冻结骨干 | v3.2 源区形状损失 |
|---|---:|---:|---:|---:|---:|
| SoC RMSE (°C) | **1.129** | 1.426 | 1.609 | 1.311 | 1.486 |
| 峰值误差 (°C) | 0.528 | **0.396** | 0.485 | 0.710 | 0.631 |
| 热点位置误差 (px) | 1.414 | 1.414 | 1.584 | 1.286 | **1.000** |
| 90% 热区 IoU | **0.786** | 0.681 | 0.726 | 0.706 | 0.579 |
| 95% 热区 IoU | **0.688** | 0.647 | 0.673 | 0.647 | 0.516 |
| 可信区 RMSE (°C) | **2.863** | 3.200 | 3.521 | 3.319 | 4.218 |

### 可确认的进步

- 相比 v3.1 嵌套材料，独立源输入将 90% IoU 从 0.681 提高到 0.726，95% IoU 从 0.647 提高到 0.673；高温核心面积从 113 像素收缩到 106 像素，说明分离材料和热源的输入语义是有效的。
- 冻结骨干将 SoC RMSE 从全续训的 1.609°C 恢复到 1.311°C，热点位置误差改善到 1.286 px，证明全骨干续训确实造成了部分已有能力漂移。

### 尚未通过的门槛

- 所有 v3.2 分支仍未同时超过旧 v2 的 SoC RMSE、峰值误差和 90%/95% IoU。
- 将树莓派式完整形状损失改为小型 SourceRegion 后，热点定位提高到 1 px，但预测高温区域反而扩大到 133/64 像素，IoU 明显下降，因此该损失分支被否决。
- 256 板规模已经足以让 Source Adapter 权重从零开始学习，但仅增加轮数不会自动解决真实板迁移问题。

## 架构瓶颈

当前 Source Adapter 只把“查询点相对各源区的位置”编码后加到 query embedding。真实板的每个源区功率则是 Board Card 中固定的先验。这个设计无法充分表达：

- 不同工作负载下各热源的相对功率变化；
- K=3 实时读数应如何反推出每个源区当前激活程度；
- 多个源区之间的全局组合关系。

下一版应把 SourceRegion 同时编码为两类信息：

1. **局部查询特征**：描述查询点是否位于源区内部或边缘；
2. **Source tokens**：与传感器 tokens 一起进入注意力，由一个共享 `SourceAmplitudeHead` 根据实时 K=3 读数预测每个源区的当前幅值。

Board Card 中只保存源区几何和幅值先验，不保存某个固定工况的最终幅值。预训练还应增加 teacher-preservation loss：在源描述丢弃或均匀材料回放样本上，使新模型保持 v3.1 的原预测，防止通用能力漂移。

## 决策

- 保留 v3.2 独立材料/热源的数据结构、Source Adapter、Board Card 和训练链路；
- Orange Pi 当前部署基线仍使用旧 v2；
- v3.2 全续训、源区形状损失均不升级为主模型；
- 暂不启动 1024/4096 板大训练；先实现“Source tokens + 动态幅值头 + teacher preservation”，再用相同 case14/case15 协议做小规模门控验证。

## 产物

- v3.2 全续训检查点：`models/v3/pretrain_v32_source_geometry_256x12_final.pt`
- v3.2 冻结骨干检查点：`models/v3/pretrain_v32_source_adapter_only_256x12.pt`
- Orange Pi 显式材料/热源配置：`configs/board_geometry_source_v3.json`
- 预训练配置：`configs/pretrain_source_geometry_v32.yaml`
- 全续训 Board Card：`benchmark/results/cross_board_opi5pro_v32_source_geometry_k3_t24_600/`
- 冻结骨干 Board Card：`benchmark/results/cross_board_opi5pro_v32_source_adapter_only_k3_t24_600/`
- 最终训练策略比较：`reports/model_comparison/cross_board_opi5pro_v32_final/`
- 源区形状损失消融：`reports/model_comparison/cross_board_opi5pro_v32_source_ablation/`
