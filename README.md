# 面向 FPC 缺陷检测的半监督语义增强归一化流模型

用于论文《基于归一化流的柔性电路板缺陷半监督检测方法》。

主要修改包括：半监督原型监督、层次语义一致性约束、温度退火机制，以及 FPCSDM2026 数据集支持。

## 方法简介

沿用归一化流异常检测框架，并引入层次向量量化结构：

- **CPC**：概念原型码本，用于图像级正常/缺陷语义分组。
- **CSPC**：概念特定模式码本，用于多尺度局部模式建模。
- **SSEF**：半监督语义增强归一化流模型，引入图像级标签进行原型约束。
- **HSCF**：层次语义协同优化方法，引入语义一致性约束和温度退火机制。

训练目标主要包括：

```text
L = L_flow + L_q + L_sup + L_sep + L_foc + L_bal + L_con
```

其中，`L_con` 用于约束 CPC 组概率与 CSPC 组概率之间的一致性。

## 环境配置

常用配置如下：

- Python 3.8 / 3.9
- PyTorch 1.9.1 或更高版本
- CUDA 11.x
- FrEIA
- torchvision, numpy, scipy, scikit-learn, opencv-python, tqdm

也可以使用：

```shell
conda env create -f environment.yaml
```

## 数据集结构

### FPCSDM2026

推荐的数据集目录结构如下：

```shell
FPCSDM2026
├── train
│   ├── good
│   └── defect
├── test
│   ├── good
│   └── defect
└── ground_truth
    └── defect
```

`train/defect` 只需要图像级标签；像素级掩码只在测试阶段用于定位评价。

### MVTec AD / VisA

MVTec AD 和 VisA 数据集结构保持原 VQ-Flow 项目格式即可。

## 训练

FPCSDM2026 训练示例：

```shell
CUDA_VISIBLE_DEVICES=0 python main.py \
  --mode train \
  --dataset fpcsdm2026 \
  --data-path ./data/FPCSDM2026 \
  --class-names all \
  --extractor resnet18 \
  --input-size 256 \
  --batch-size 4 \
  --meta-epochs 100 \
  --quantize-enable \
  --quantize-type residual \
  --concat-pos
```

MVTec AD 训练示例：

```shell
CUDA_VISIBLE_DEVICES=0 python main.py \
  --mode train \
  --dataset mvtec \
  --class-names all \
  --multi-class \
  --extractor resnet18 \
  --input-size 256 \
  --quantize-enable \
  --quantize-type residual \
  --concat-pos
```

## 测试

```shell
CUDA_VISIBLE_DEVICES=0 python main.py \
  --mode test \
  --dataset fpcsdm2026 \
  --data-path ./data/FPCSDM2026 \
  --class-names all \
  --extractor resnet18 \
  --input-size 256 \
  --quantize-enable \
  --quantize-type residual \
  --concat-pos \
  --eval_ckpt ./checkpoints/model.pt
```

## 主要参数

| 参数 | 说明 |
| --- | --- |
| `--dataset` | 数据集名称，可选 `fpcsdm2026`、`mvtec`、`visa`。 |
| `--data-path` | 数据集根目录。 |
| `--extractor` | 特征提取器。论文设置为 `resnet18`。 |
| `--input-size` | 输入图像尺寸。论文设置为 `256`。 |
| `--quantize-enable` | 启用层次向量量化。 |
| `--k-cpc` | CPC 码本大小，默认 `32`。 |
| `--k-cspc` | CSPC 码本大小，默认 `512`。 |
| `--lambda-cpc-proto` | CPC 原型监督损失权重。 |
| `--lambda-cspc-proto` | CSPC 原型监督损失权重。 |
| `--lambda-con` | 层次语义一致性损失权重。 |
| `--cpc-tau-min` | CPC 最小温度。 |
| `--cspc-tau-min` | CSPC 最小温度。 |

## 消融实验

常用消融设置：

```shell
# 去除 CPC 监督
--lambda-cpc-proto 0

# 去除 CSPC 监督
--lambda-cspc-proto 0

# 去除层次语义一致性约束
--lambda-con 0

# 固定温度，关闭温度退火效果
--proto-tau 0.2 --cpc-tau-min 0.2 --cspc-tau-min 0.2
```

## 说明

- 模型使用图像级标签进行半监督原型学习。
- 像素级掩码仅用于测试阶段的定位评价。
- 论文默认设置为 `ResNet18 + 256×256 + CPC 32 + CSPC 512 + batch size 4 + 100 epochs`。
- 当前补丁版本重点接入 SSEF 与 HSCF 的损失和训练逻辑，未强行接入归一化流缺陷检测解码器。

