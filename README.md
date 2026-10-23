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

### Data Audit and Validation Split

The dataset is strongly imbalanced, and the supplied test labels do not cover all classes. To ensure rigorous evaluation, a reproducible 20% disjoint holdout was created using `src/split_custom_data.py`. 

The splitter balances image-level class presence where possible, excludes all validation images from the training list, and leaves the original training files untouched. **Note:** An earlier draft of this split had data-leakage risks (images remained in the training directory). That protocol was audited and discarded. All final metrics reported below are based on this corrected, strictly disjoint validation set.

With the current dataset, the generated split contains **2,679 training images** and **670 validation images**. The validation set contains the following classes (image counts): `loose-meat`: 17, `packaging-error`: 389, `twisted-meat`: 18, `unsealed`: 273, and `wrinkle`: 4.
---

## Repository Structure

```
pork-packaging-yolo-comparison/
├── src/                     # Source python code
├── tests/                   # pytest unit tests
├── notebooks/               # Exploratory data analysis notebooks
├── docs/html/               # Generate source documentation with pdoc
├── data/                    # Dataset yaml + annotation txt labels (images are git-ignored)
├── runs/                    # Native YOLO training outputs: csv logs, training curves, confusion matrices
├── results/                 # Experiment summary CSV, selected result figures
├── report/                  # Written report, test report, and evaluation figures
├── venv/                    # Python virtual environment (ignored by git)
├── README.md
├── LICENSE
├── .gitignore
├── dataset.yaml
└── yolov8s.pt               # Pretrained weight for reference
```

## Environment Setup
```bash
# create virtual environment
python -m venv venv

# activate venv (Windows PowerShell)
.\venv\Scripts\Activate.ps1

# install dependencies
pip install torch ultralytics pytest pytest-html pdoc pandas matplotlib
```
## Dataset Instruction
Raw dataset images are not tracked in this repository (too large for GitHub).
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

## Numerical Methods Integration
Custom trapezoidal and left-Riemann integration functions are implemented in `src/custom_map.py` and covered by unit tests. They integrate supplied precision-recall points; they do not replace Ultralytics' mAP calculation. To present them as an independent verification of model AP, compare both methods on the same exported precision-recall points and discuss their different interpolation assumptions.


## How to Run

```bash
# 1. Validate dataset paths and pair image-label files
python src/data_preprocess.py

# 2. Create a disjoint, reproducible train/validation split
python src/split_custom_data.py

# 3. Train a YOLO model on the training list (example)
yolo detect train data=data/dataset.yaml model=yolov8s.pt epochs=30 imgsz=640 batch=8

# 4. (Optional) Train using the corrected split
yolo detect train data=data/dataset.yaml model=yolov8s.pt epochs=30 imgsz=640 batch=8 seed=42 device=0

# 5. Evaluate an existing checkpoint on the independent validation split (Windows Git Bash / PowerShell)
./venv/Scripts/yolo.exe detect val data=data/dataset.yaml model=runs/detect/yolov8s_baseline_640_batch8/weights/best.pt imgsz=640 batch=8

# 6. Evaluate the existing 640/batch-8 checkpoints on the held-out test split
yolo detect val data=data/dataset.yaml split=test model=runs/detect/yolov5s_baseline_640_batch8/weights/best.pt imgsz=640 batch=8 device=0
yolo detect val data=data/dataset.yaml split=test model=runs/detect/yolov8s_baseline_640_batch8/weights/best.pt imgsz=640 batch=8 device=0

# 7. Run unit tests and generate HTML report
pytest tests/ -v --html=report/test_report.html --self-contained-html

# 8. Generate source-code HTML documentation
pdoc src/data_preprocess.py src/model_runner.py src/custom_map.py src/split_custom_data.py src/check_all_labels.py src/train_ablation_416.py -o docs/html
```
## Experiment Status & Results

### Final Validation Results (Custom Disjoint Split)
To eliminate data leakage and ensure reliable evaluation, the pre-trained `best.pt` weights were evaluated on the strictly disjoint custom validation set (670 images, 747 instances). No retraining was required for this out-of-sample evaluation.

| Model | mAP@0.5 | mAP@0.5:0.95 | Precision | Recall | Inference (ms/img) | Model Size |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| YOLOv5s (Baseline) | 0.504 | 0.267 | 0.812 | 0.395 | 3.7 | 18.5 MB |
| YOLOv8s (Final) | **0.673** | **0.402** | 0.863 | 0.571 | 4.5 | 22.5 MB |

### Per-Class Performance (YOLOv8s)
- **unsealed**: 0.950 (Excellent)
- **packaging-error**: 0.905 (Excellent)
- **loose-meat**: 0.831 (Good)
- **twisted-meat**: 0.619 (Moderate)
- **wrinkle**: 0.059, Recall=0 (Failure case: Only 4 validation images, severe overfitting due to data scarcity).

### Key Findings
1. **Architecture:** YOLOv8s (C2f) significantly outperforms YOLOv5s (C3) on this dataset (+16.9% mAP).
2. **Data Imbalance:** The complete failure of the `wrinkle` class proves that extreme data scarcity leads to background overfitting, validating the Week 3 theoretical concepts.
3. **Trade-off:** The 4.5 ms inference time of YOLOv8s is well within real-time industrial requirements, justifying the slight size increase.

The comprehensive 8-page report is available as [`report/project_report.pdf`](report/project_report.pdf), with source files in [`report/`](report/). All test plots (Confusion Matrix, PR Curve, Loss curves) and `test_report.html` are included in the repository.


## Deliverables
1. Written project report (5–10 pages maximum; source and PDF in `/report`)
2. GitHub repository with reproducible workflow, source HTML documentation and tests
3. Individual 5-minute presentation plus 3-minute Q&A during the Week 14 examination period (book and attend a session)

Before submission, review the changed files, commit and push the final project files, submit the repository link and a zipped clone through Moodle, and book/attend the Week 14 presentation and Q&A. These external submission and attendance steps cannot be completed from this local workspace.


## AI Usage Note

The existing project record reports using DeepSeek V4 to assist with understanding YOLO training parameters and experiment design. Copilot was also used for this revision to compare the project against the brief, identify the train/validation overlap, and assist with changes to the split workflow, metrics, tests, and documentation. The student remains responsible for reviewing the code and results and for all final engineering decisions.

The project report records the tools used, their approximate contribution, how suggestions were checked, a methodological error found during validation review, which decisions remained the student's, and the student's learning reflection.


## License

Dataset: CC BY 4.0 (Roboflow Pork Rasher Error Packaging v4).  
Code: MIT License (see LICENSE file).