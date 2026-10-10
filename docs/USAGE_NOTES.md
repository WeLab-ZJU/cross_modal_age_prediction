# 使用说明

## 环境与眼球模型

- 依赖版本未锁定。ViT 实现继承 timm，请确认当前环境中
  `forward_features`、继承的 `forward` 和分类头相互兼容。
  Global pooling 返回 `[B,1,C]`，分类损失要求 `[B,K]`；
  请结合安装的 timm 版本检查输出形状。
- 输入尺寸需要与 patch size 和 checkpoint 匹配。部分历史配置使用
  Base/16 + 196 或 Huge/14 + 256，尺寸不是对应 patch size 的整数倍。
  当前启动脚本默认 Base/16 + 256。
- 眼球训练使用 CUDA AMP，并无条件调用 `torch.cuda.synchronize()`，
  不支持 CPU/MPS 训练。
- 默认池化配置为 `global_pool=True`。
- 仅加载可信来源的 checkpoint 和 joblib 文件，并确认当前 PyTorch
  版本的反序列化兼容性。

## 训练与评测

- `--eval` 仍构建 `train/val/test`，三个目录均需存在。
  评测权重通过 `--resume` 加载。
- 最佳权重按 `(F1 + ROC_AUC + Kappa)/3` 选取，
  对应训练变量名为 `max_auc`。
- 验证集需要覆盖各类别。缺失类别可能导致 AUC 为 NaN；
  若没有分数超过初始值 0，可能不会保存最佳权重，而最终评测仍会尝试加载它。
- Best checkpoint 不包含完整优化器状态，用它恢复不等于完整续跑训练。
- 分布式评测同步计数，但未汇总所有 rank 的预测后再计算全部指标；
  多个 rank 还可能写入同一 CSV 或混淆矩阵文件。启动脚本默认单 GPU。
  多卡使用时需检查这些行为，尤其是 `--dist_eval`。

## 小鼠融合与年龄分数

- 自动选择设备：无 CUDA 时用 CPU，一块 GPU 时两路共用，
  多块 GPU 时分别使用 `cuda:0` 和 `cuda:1`。
- 两路眼球数据必须按个体对齐。第二路按第一路的 ID 列表重排，
  文件数相同不能证明个体对应。
- 眼球文件排序后通过 `zip` 配对，列表长度不等时会截断。
  请核对两路文件列表、个体 ID、标签和顺序。
- 步态汇总后保留个体在 CSV 中首次出现的顺序，不会显式按
  `--subject_order` 重排。两者顺序必须一致。
  仅比较标签无法发现同年龄组内的个体错配。
- “Eye-A-Single” 和 “Eye-B-Single” 输出使用
  `head(latent1 + latent2)`，不是隔离另一模态后的独立单模态推理。
- 合成步态与行为演示使用不同的 ID，不能直接组合用于三模态融合。
- 连续年龄评分使用真实年龄组 `label` 选择年龄锚点，
  不是无需年龄组信息的独立年龄回归。

## 历史配置

可运行入口见 `scripts/` 和 `mouse/demo/`。
`legacy/` 下的 `.sh.txt` 文件仅作为实验配置参考。

- 部分 Huge 配置只暴露两块 GPU，却启动四个进程。
- 历史 FP 单模态测试传入了 `train_eye.py` 不支持的 `--class_order`。
- 历史 `vis.sh` 引用未提供的 `vis_vit_cam.py`；
  `test_eye_huge.sh` 为空。可视化请使用 `human/visualize_all_best_models.py`。
- 同一变量重复赋值时，以最后一次赋值为准。
