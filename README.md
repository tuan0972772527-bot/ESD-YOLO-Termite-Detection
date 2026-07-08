# ESD-YOLO: A Method for Termite Detection in Complex Backgrounds with Small Targets

## Research Background

Timely and accurate termite detection is important for termite monitoring, prevention, and control. In real activity environments, termites are often mixed with wood chips, soil, stones, and termite galleries. Their bodies are small, low-contrast, and easily affected by background texture interference, which makes precise object detection and localization difficult.

Existing termite vision studies have mainly focused on species classification, single-class termite detection, or damage-area monitoring. Systematic multi-class termite individual detection under complex backgrounds is still limited. To address this problem, this study constructs a termite image dataset with complex backgrounds and proposes **ESD-YOLO**, a fine-grained feature-enhanced object detection model based on YOLO11n.

The main challenges addressed in this work are:

1. Termite individuals are small and often densely distributed;
2. Body colors can be similar to soil, wood chips, and termite galleries;
3. Inter-class visual differences are subtle;
4. Occlusion, overlap, and weak object boundaries affect localization accuracy;
5. Background textures can easily cause false positives or bounding-box misalignment.

---

## Model Architecture / ESD-YOLO

ESD-YOLO follows the design logic:

`Feature Enhancement -> Deep Context Aggregation -> Dynamic Multi-scale Fusion -> Detection`

Using YOLO11n as the baseline, ESD-YOLO introduces three core improvements:

1. **C3k2-EMA**
   The Efficient Multi-scale Attention (EMA) mechanism is introduced into the C3k2 backbone module to enhance termite-region feature responses and suppress background texture interference.

2. **SPPF-DGP**
   The original SPPF module is replaced with Spatial Pyramid Pooling-Fast with Dual Global Pooling (SPPF-DGP). Global average pooling and global max pooling are added to combine global contextual information with salient response information.

3. **DySample**
   The nearest-neighbor upsampling operation in the neck is replaced with DySample dynamic upsampling to improve spatial alignment during multi-scale feature fusion and strengthen boundary representation for small termite targets.

The overall model is defined in:

```text
models/ESD-YOLO.yaml
```

Core modules are provided in:

```text
models/modules/EMA.py
models/modules/SPPF_DGP.py
models/modules/DySample.py
```

---

## Dataset

The dataset used in this study contains five termite species collected from Zhuji City, Zhejiang Province, China:

| Abbreviation | Species |
| --- | --- |
| CF | Coptotermes formosanus |
| MB | Macrotermes barneyi |
| OF | Odontotermes formosanus |
| RF | Reticulitermes flaviceps |
| RC | Reticulitermes chinensis |

Images were captured under controlled acquisition conditions while retaining complex background elements such as wood chips, soil, stones, and termite galleries. All termite instances were manually annotated and converted into YOLO format.

This repository only provides a lightweight sample dataset for project display:

```text
sample_data/images/
sample_data/labels/
```

For each termite category, several non-consecutive example images and corresponding YOLO-format labels are included.

---

## Installation

Clone the repository and install dependencies:

```bash
git clone https://github.com/your-name/ESD-YOLO-Termite-Detection.git
cd ESD-YOLO-Termite-Detection
pip install -r requirements.txt
```

This project is based on the Ultralytics YOLO framework. The custom modules should be registered in the corresponding Ultralytics model parser before training with `ESD-YOLO.yaml`.

---

## Usage

### Train

```bash
python train.py --data path/to/data.yaml --epochs 300 --batch 48 --imgsz 640 --device 0
```

### Validate

```bash
python val.py --weights path/to/best.pt --data path/to/data.yaml --split test
```

### Predict

```bash
python predict.py --weights path/to/best.pt --source sample_data/images
```

---

## Project Structure

```text
ESD-YOLO-Termite-Detection/
|-- README.md
|-- requirements.txt
|-- train.py
|-- val.py
|-- predict.py
|-- models/
|   |-- ESD-YOLO.yaml
|   `-- modules/
|       |-- EMA.py
|       |-- SPPF_DGP.py
|       `-- DySample.py
|-- sample_data/
|   |-- images/
|   `-- labels/
`-- docs/
    `-- data_description.md
```

---

## Funding & Data Statement

This research was supported by Zhejiang Provincial Natural Science Foundation of China under Grant No. LY23C140004. In collaboration with local government departments, this project has signed relevant confidentiality agreements. For this reason, we only display partial data samples instead of sharing the complete raw data.

---

## Citation

If this project is useful for your research, please cite this work after the paper is formally published.
