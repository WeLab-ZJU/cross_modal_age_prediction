# 合并记录与已知限制

整理日期：2026-10-10。

## 迁移原则

- 原始 `codes/`、`codes_mouse/` 不修改；全部工作在新目录内完成。
- 原 Python 文件全部保留。逐文件来源、原 SHA-256 和打包后的 SHA-256
  见 `source_manifest.json`，路径不包含个人机器根目录。
- 两套训练、模型和 `util/` 原本相同，暂保留双份以降低重构风险。
- 原 `codes/A_train_eye.py` 与 `train_eye.py` 相同，继续保留并标为历史重复入口。
- 原 `codes/fuse_modalities.py` 及 `_generate_demo_data.py` 实际与小鼠版本相同，
  为不丢文件保留在人类目录，但不注册为人类的统一命令。
- 补齐 `mouse/engine_finetune.py`，来源是原人类目录的同名文件。
- 将原 shell 脚本、README 移至 `legacy/<species>/`；脚本保存为
  `.sh.txt`，防止被误认为可直接执行。原服务器路径改为尖括号占位符。
  原 README 中的外部来源、性能、环境与授权文字**未经本次核实**，
  仅用于历史追溯，以新 README 为准。
- 不复制 `.DS_Store`、`__pycache__`、`.pyc` 和旧 `.gitignore`。
- 新增根 README、数据约定、依赖清单、统一入口、可配置 shell 及标准库检查。

## 对 Python 代码的有限修改

1. `human/train_eye.py`、`human/A_train_eye.py`、`mouse/train_eye.py`：
   打开 argparse 的 `--help`；不改变训练参数默认值。
2. 两份 `_generate_demo_data.py`：输出目录由脚本所在目录改为
   该目录内的 `data_examples/`，匹配演示脚本。
3. `human/visualize_all_best_models.py`：将旧服务器默认路径换为
   `outputs/human/...` 和 `data/human`。
4. `mouse/engine_finetune.py`：仅补齐缺失依赖，内容与原人类引擎一致。

没有修改损失函数、模型结构、checkpoint 格式、类别重排、融合算法、
训练超参数默认值或指标公式。

## 原代码中发现、但本次未擅自更改的问题

这些是源代码审查发现的风险；不等同于已经在原实验环境复现了错误。

### 眼球训练 / 依赖

- 缺少原始环境锁文件。模型继承 timm 的 ViT，必须在实际版本下验证
  `forward_features` 与继承的 `forward` / 分类头兼容性。
  尤其 global pooling 返回 `[B,1,C]`，而分类损失一般接收 `[B,K]`；
  需核验原环境是否已有相应处理，本次没有改变张量形状。
- 一些历史模板中 Base/16 输入为 196、Huge/14 输入为 256；
  这些不是 patch size 的整数倍。不要不加核对地复用，
  请验证 patch embedding 和权重兼容性。新模板默认 Base/16 + 256。
- 训练默认 `global_pool=True`。两套代码都使用 CUDA AMP，
  `train_one_epoch()` 无条件调用 `torch.cuda.synchronize()`；
  CPU/MPS 训练并未适配。
- `evaluate()` 返回的选优值是 `(F1 + ROC_AUC + Kappa)/3`，
  训练变量却叫 `max_auc`；本次仅修正文档描述。
- 若验证集缺少类别造成 AUC 为 NaN，或所有 score 都不大于初始 0，
  可能未保存最佳权重，但训练结束仍尝试加载 `checkpoint-best.pth`。
- 分布式评测只同步指标计数，未汇总各 rank 的完整预测再算所有指标；
  多个 rank 还可能同时写 CSV/混淆矩阵。新模板默认单 GPU，
  正式多卡评测需另行审查，尤其 `--dist_eval`。
- `--eval` 仍构建 train/val/test，不是仅有 test 目录即可运行。
- best checkpoint 未包含完整优化器状态；用它恢复不等同于完整训练续跑。
- 不同 PyTorch 版本的 checkpoint 反序列化行为需在实际环境验证；
  不要为了兼容而无条件信任来路不明的序列化文件。

### 历史 shell 模板

- 某些 Huge 模板只暴露两块 GPU，却启动四个进程。
- 单模态 FP 测试模板传入了训练入口不支持的 `--class_order`。
- 旧 `vis.sh` 引用了未提供的 `vis_vit_cam.py`，另有空的 `test_eye_huge.sh`。
- 多个变量重复赋值，后一个覆盖前一个。为保留实验记录而未重写历史文件。
  新增 `scripts/` 才是可配置的运行入口。

### 小鼠融合 / 数据语义

- 原代码自动选择设备：无 CUDA 时用 CPU，一块 GPU 时共用，
  多块 GPU 时用 `cuda:0` 与 `cuda:1`；未增加设备 CLI。
- 眼球第二路按第一路 ID 列表重排；两路源文件必须逐个体严格对齐，
  不能仅因为文件数量相同就认为对应。
- 步态汇总后保持 CSV 内个体首次出现顺序，没有显式按
  `--subject_order` 重排。调用前必须确保该顺序一致。
  原检查仅比较标签，不能发现同年龄组内部错配。
- 眼球文件通过排序后 `zip` 配对，可能截断长度不等的列表；
  正式评测应核查文件列表、ID、标签和顺序。
- 名称为 “Eye-A-Single” / “Eye-B-Single” 的输出在原代码中来自
  `head(latent1 + latent2)`，不是隔离另一模态后的纯单模态推理。
- 合成演示的行为/步态 ID 不一致，因此不提供“无需真实数据”的三模态演示。
  旧融合模板保留在 `legacy/mouse/demo/`，正式命令见根 README。
- 连续年龄评分需要真实年龄组 `label`，不能描述为完全独立的年龄预测。

## 发布状态

本次完成的是代码打包、路径整理和静态检查；没有实际训练、比较模型性能或
验证真实数据配对。请在原训练环境逐项确认以上限制后再进行科学结果复现。
未添加 LICENSE：应先确认作者授权、第三方代码来源与兼容的许可条款。
