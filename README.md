# 通用物体语义分割训练项目

这是一个从零准备数据、训练、评估和推理语义分割模型的轻量项目骨架。默认使用 PyTorch + torchvision 的 DeepLabV3 预训练模型做迁移学习，适合 6-20 类通用物体分割的第一版验证。

## 1. 环境准备

建议使用 Python 3.9+ 和 NVIDIA GPU。

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## 2. 数据格式

原始数据放在：

```text
data/raw/images/
data/raw/masks/
```

要求：

- 每张图片对应一个 mask，文件名主干一致，例如 `001.jpg` 对应 `001.png`。
- mask 必须是单通道 PNG。
- mask 像素值是类别 id，例如 `0=background, 1=person, 2=car`。
- 类别定义在 `configs/classes.json`。

处理后数据会放在：

```text
data/processed/train/images/
data/processed/train/masks/
data/processed/val/images/
data/processed/val/masks/
data/processed/test/images/
data/processed/test/masks/
```

## 3. 切分数据

如果你想直接准备一个猫/狗语义分割小数据集，可以使用 Oxford-IIIT Pet 官方数据源：

```bash
python scripts/prepare_oxford_pet_subset.py --max-per-class 80
```

这会生成：

- `data/processed/train/images/` 和 `data/processed/train/masks/`
- `data/processed/val/images/` 和 `data/processed/val/masks/`
- `data/processed/test/images/` 和 `data/processed/test/masks/`

默认类别是 `0=background, 1=cat, 2=dog`。

如果你已经有自己的原始图片和 mask，可以使用下面的脚本切分数据：

```bash
python scripts/split_dataset.py \
  --images data/raw/images \
  --masks data/raw/masks \
  --output data/processed \
  --train-ratio 0.70 \
  --val-ratio 0.15 \
  --test-ratio 0.15
```

## 4. 检查数据

```bash
python scripts/validate_dataset.py --data-root data/processed --classes configs/classes.json
```

## 5. 训练

```bash
python -m src.segmentation.train --config configs/train.yaml
```

训练输出：

- `outputs/checkpoints/best.pt`
- `outputs/checkpoints/last.pt`
- `outputs/reports/train_metrics.json`

## 6. 评估

```bash
python -m src.segmentation.evaluate \
  --config configs/train.yaml \
  --checkpoint outputs/checkpoints/best.pt \
  --split test
```

## 7. 单图推理

```bash
python -m src.segmentation.infer \
  --config configs/train.yaml \
  --checkpoint outputs/checkpoints/best.pt \
  --image path/to/image.jpg \
  --output outputs/predictions/result.png
```

## 8. 目标检测：猫狗分类 + 红框

如果目标是“判断图片里是猫还是狗，并用框框出来”，建议使用目标检测模型，而不是语义分割模型。项目已提供 Oxford-IIIT Pet 到 YOLO 格式的转换脚本。

准备 YOLO 检测数据：

```bash
python scripts/prepare_oxford_pet_detection.py --max-per-class 300
```

这会生成：

```text
data/detection/images/train
data/detection/images/val
data/detection/images/test
data/detection/labels/train
data/detection/labels/val
data/detection/labels/test
```

类别定义：

```text
0 = cat
1 = dog
```

训练 YOLOv8 nano 检测模型：

```bash
yolo detect train \
  model=yolov8n.pt \
  data=configs/detection.yaml \
  epochs=30 \
  imgsz=640 \
  batch=8 \
  project=outputs/detection \
  name=cat_dog_yolov8n
```

如果没有 GPU，可以先把 `epochs` 调小，例如 `epochs=3`，用于确认流程。

评估检测模型：

```bash
yolo detect val \
  model=outputs/detection/cat_dog_yolov8n/weights/best.pt \
  data=configs/detection.yaml
```

单图检测：

```bash
yolo detect predict \
  model=outputs/detection/cat_dog_yolov8n/weights/best.pt \
  source=data/detection/images/test \
  project=outputs/detection \
  name=predictions
```

## 推荐数据量

- 最低可跑通：每类 100-300 张有效标注图。
- 较可靠：每类 500-1,000 张。
- 更接近生产：每类 1,000+ 张，并覆盖真实场景。

语义分割最重要的是 mask 质量。类别定义、边界标准、遮挡标准必须在标注前固定。
