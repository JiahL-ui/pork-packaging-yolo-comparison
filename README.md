# MMA3001 Project
**Performance comparison of YOLOv5s and YOLOv8s for packaging defect detection on pork rasher trays**

> University project for MMA3001 — Numerical Methods and Machine Learning  
> Individual project assessing the performance difference between YOLOv5s and YOLOv8s on the pork-rasher packaging defect detection task.

---

## Project Overview

This project compares engineering-oriented performance metrics including detection accuracy, inference runtime and model size between YOLOv5s and YOLOv8s on a pork-rasher packaging defect detection task.

### Dataset
Official MMA3001 Dataset 3: packaged pork-rasher RGB images with bounding-box annotations (Roboflow, CC BY 4.0).

The original dataset defines **5 classes**: `loose-meat`, `packaging-error`, `twisted-meat`, `unsealed`, `wrinkle`.

### Data Audit Findings

A detailed audit of the dataset revealed severe class imbalance and missing validation labels:

| Class ID | Class Name | Train Instances | Valid Instances | Test Instances | Evaluation Status |
|:---:|:---|:---:|:---:|:---:|:---|
| 0 | loose-meat | 15 | 1 | 1 | Evaluated |
| 1 | packaging-error | 207 | 14 | 9 | Evaluated |
| 2 | twisted-meat | 9 | 0 | 0 | **Not Evaluated** |
| 3 | unsealed | 28 | 1 | 2 | Evaluated |
| 4 | wrinkle | 18 | 0 | 0 | **Not Evaluated** |

**Key limitations:**
- `wrinkle` (class 4) has 18 training instances but **zero annotations in the validation and test sets**. Its AP cannot be computed. It is retained in `dataset.yaml` for index consistency but excluded from final evaluation metrics.
- `twisted-meat` (class 2) has 9 training instances but **zero annotations in the validation and test sets**. Its AP cannot be computed either.
- `loose-meat` (class 0) and `unsealed` (class 3) have very few validation instances (1 each), so their validation metrics have high variance.
- The validation set contains 120 images, of which 14 have empty label files (no defect objects).

**Conclusion:** Final evaluation focuses on `packaging-error` and `unsealed`, which have sufficient validation instances.

---

## Repository Structure

```
pork-packaging-yolo-comparison/
├── src/ # Source python code
├── tests/ # pytest unit tests
├── notebooks/ # Exploratory data analysis notebooks
├── docs/html/ # Auto-generated code documentation from pdoc
├── data/ # Dataset yaml + annotation txt labels (images are git-ignored)
├── runs/ # Native YOLO training outputs: csv logs, training curves, confusion matrices
├── results/ # Experiment summary CSV, selected result figures
├── report/ # Final project report PDF
├── venv/ # Python virtual environment (ignored by git)
├── README.md
├── LICENSE
├── .gitignore
├── dataset.yaml
└── yolov8s.pt # Pretrained weight for reference
```

## Environment setup
```bash
# create virtual environment
python -m venv venv

# activate venv (Git-Bash / MINGW64 on Windows)
source venv/Scripts/activate

# install dependencies
pip install torch ultralytics pytest pdoc pandas matplotlib
```
## Dataset instruction
Raw dataset images are not stored in this repository (too large for GitHub).
Download MMA3001 Dataset 3 from course resources and place images and annotation files into the local `data/` folder:
```
data/
├── train/
│   ├── images/
│   └── labels/
├── valid/
│   ├── images/
│   └── labels/
└── test/
    ├── images/
    └── labels/
```
Only YOLO-format `.txt` annotation labels are tracked in this repository. Images are git-ignored.

## Experiment Status

✅ Baseline finished: YOLOv5s & YOLOv8s trained for 30 epochs under identical  hyper‑parameters (input size 640).

✅ Generated mAP, precision, recall metrics, training curves and confusion matrices.

🔜 Next: Ablation study of input resolution; failure‑case analysis.

## How to Run

```bash
# 1. Validate dataset paths and pair image-label files
python src/data_preprocess.py

# 2. Check all annotation files for format errors and invalid class IDs
python src/check_all_labels.py

# 3. Train a YOLO model (CLI, matches actual experiments)
yolo detect train \
  data=data/dataset.yaml \
  model=yolov8s.pt \
  epochs=30 \
  imgsz=640 \
  batch=8 \
  name=yolov8s_baseline_640_batch8 \
  project=runs/detect

# 4. Run unit tests
pytest tests/ -v

# 5. Generate source-code HTML documentation
pdoc src -o docs/html
```

## Experiment Status & Results

### Completed experiments (30 epochs each)

| # | Model | imgsz | batch | mAP@0.5 | mAP@0.5:0.95 | Model Size |
|:---:|:---|:---:|:---:|:---:|:---:|:---:|
| 1 | YOLOv5s | 640 | 16 | 0.334 | 0.195 | 18.5 MB |
| 2 | YOLOv8s | 640 | 16 | 0.187 | 0.0924 | 22.5 MB |
| 3 | YOLOv5s | 640 | 8 | **0.478** | **0.280** | 18.5 MB |
| 4 | YOLOv8s | 640 | 8 | **0.587** | **0.354** | 22.5 MB |

### Key Findings

1. **YOLOv8s outperforms YOLOv5s** at both batch settings.
2. **batch=8 significantly outperforms batch=16** for both models (mAP@0.5: 0.587 vs 0.187 for YOLOv8s). This is attributed to the regularisation effect of larger gradient noise in small batches, which helps the model escape poor local minima.
3. **Severe class imbalance** limits the reliability of minority-class metrics (`loose-meat`, `twisted-meat`).


### Planned / In Progress

- 🔜 Resolution ablation: YOLOv8s @ 1280 (batch=8), pending due to 8 GB GPU memory limit.
- 🔜 Failure-case analysis on `packaging-error` (most frequent but hardest class).


## Deliverables
1. Project written report (5‑10 pages max, inside /report)
2. GitHub repository with reproducible workflow
3. 5‑min presentation + 3‑min Q&A in week‑14 examination period


## AI usage note

All AI tool usage (ChatGPT) is documented in the final project report's AI-reflection chapter, as required by the MMA3001 project brief. In summary:

1. AI was used to assist with understanding YOLO training parameters and experiment design.
2. All AI-generated suggestions were verified through empirical testing (e.g. the batch=16 @ 1280 suggestion caused GPU memory overflow and was rejected based on nvidia-smi monitoring).
3. All engineering decisions (batch size, resolution, control variables) were made independently by the student.


## License

Dataset: CC BY 4.0 (Roboflow Pork Rasher Error Packaging v4).  
Code: MIT License (see LICENSE file).
