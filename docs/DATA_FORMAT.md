# 数据与标签约定

数据目录、类别映射和个体对齐要求。

## 眼球图像

```text
data/<species>/<modality>/
├── train/<class_name>/*.png
├── val/<class_name>/*.png
└── test/<class_name>/*.png
```

- 单模态训练与 `--eval` 使用 ImageFolder 的类别目录排序。三个 split
  必须包含相同类别目录，目录到整数标签的对应关系必须一致。
- 人类示例目录为 `1-30`、`30-60`、`60-80`、`80+`；
  边界年龄如何划分由数据集定义。
- 小鼠眼球分类头输出顺序在融合代码中固定假设为
  `12M, 18M, 3M, 8W`，融合时重排至 `8W, 3M, 12M, 18M`。
- 不要把人类 checkpoint、类别或标签语义混用到小鼠流程。
- 直接测试命令使用 `--resume`；`--finetune` 不是 `--eval` 的权重加载入口。

## 人类 OCT / OCTA 融合

`fuse_oct_octa_human.py` / `fuse_oct_octa_human2.py` 按类别扫描图像，
通过 `split_subject_side()`、`group_lr_pairs()` 推断个体和左右眼。
请按已有解析逻辑核对自己的文件名；不要假设任意文件名都可以正确配对。
新版的 `extract_real_age_from_path()` / `extract_real_age_from_files()`
从路径或文件名提取年龄；年龄缺失或命名不同会影响 MAE 的可用性。
`--class_age_values` 可显式设置与 `--class_order` 对齐的类别代表年龄。
所有配对和年龄字段应在输出中抽样核查。

## 小鼠表格

步态 CSV：

```csv
id,feature1,feature2,label
S001,0.1,0.2,0
S001,0.3,0.4,0
S002,0.5,0.6,1
```

行为 CSV：

```csv
id,feature1,feature2,label
S001,0.1,0.2,0
S002,0.5,0.6,1
```

- 上面仅展示格式，不是足够训练四分类器的数据集。
- 表格标签固定约定为 `0=8W, 1=3M, 2=12M, 3=18M`。
- 步态允许同一个体多行序列，按 ID 汇总概率；同一个体所有行标签应一致。
- 行为通常每个体一行；默认 `--n_features 28`，使用两列示例时设为 2。
- train/test 必须保持数值列名称和顺序一致；当前 scaler 会处理
  删除 `id` 和 `label` 后的整个表格，再选定特征，不要增加字符串元数据列。
- 真正实验的训练/验证/测试个体应互斥。合成数据生成器仅演示格式和流程，
  不能用其性能证明泛化，生成器在 train/test 中会复用演示 ID 字符串。
- 三模态融合要求步态、行为和两路眼球的 ID 一致；
  `--subject_order` 必须包含所有测试 ID 且没有重复。
  步态及眼球第二路的对齐限制见 `USAGE_NOTES.md`。

## 连续年龄分数

输入至少包含 `class_0,class_1,class_2,class_3,label`。
列顺序对应 `8W,3M,12M,18M`。`label` 是真实类别索引，
原函数用它选择年龄锚点；`max_index` 不是该步骤的必要输入。
分数先按行和归一化；请排查零行和、缺失值和不符合预期的负分数。
