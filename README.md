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

The dataset is strongly imbalanced, and the supplied validation/test labels do not cover every class. The project therefore creates a reproducible 20% holdout from the training data. The splitter balances image-level class presence where possible, writes validation images and labels to `data/independent_split/valid/`, and writes the remaining training images to `data/independent_split/train.txt`.

The training list excludes every validation image; the original training files are left untouched. This split must be generated before training. The earlier `data/custom_valid/` split copied images without excluding them from training, so metrics obtained from that split are **not independent validation results** and must not be used as final evidence.

With the dataset currently in the workspace, the generated split contains 2,679 training images and 670 validation images. The validation images contain the following classes (image counts, not object-instance counts): `loose-meat`: 17, `packaging-error`: 389, `twisted-meat`: 18, `unsealed`: 273, `wrinkle`: 4. Recheck these counts if the source dataset changes.

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

## Experiment Status

- **Historical training runs:** Six 30-epoch experiments and a 640/1280 resolution comparison are present.
- **Validation status:** Runs completed before the disjoint holdout was introduced are preliminary. Re-run training and evaluation against the generated split before treating their metrics or failure-case analysis as final.
- **Tests:** Run `pytest tests/ -v` to reproduce the current unit-test results.

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

# 5. Evaluate an existing checkpoint on the independent validation split
yolo detect val data=data/dataset.yaml model=runs/detect/yolov8s_independent_640/weights/best.pt imgsz=640 batch=8 device=0

# 6. Evaluate the existing 640/batch-8 checkpoints on the held-out test split
yolo detect val data=data/dataset.yaml split=test model=runs/detect/yolov5s_baseline_640_batch8/weights/best.pt imgsz=640 batch=8 device=0
yolo detect val data=data/dataset.yaml split=test model=runs/detect/yolov8s_baseline_640_batch8/weights/best.pt imgsz=640 batch=8 device=0

# 7. Run unit tests and generate HTML report
pytest tests/ -v --html=report/test_report.html --self-contained-html

# 8. Generate source-code HTML documentation
pdoc src/data_preprocess.py src/model_runner.py src/custom_map.py src/split_custom_data.py src/check_all_labels.py src/train_ablation_416.py -o docs/html
```

## Experiment Status & Results

### Held-out test evaluation (640, batch 8)

The existing `best.pt` checkpoints were evaluated on the supplied test split with the same settings and GPU (RTX 4060 Laptop, Ultralytics 8.4.144, CUDA). No training was performed for this evaluation.

| Model | Checkpoint | Images | Instances | Precision | Recall | mAP@0.5 | mAP@0.5:0.95 | Inference (ms/image) |
|:---|:---|---:|---:|---:|---:|---:|---:|---:|
| YOLOv5s | `yolov5s_baseline_640_batch8/weights/best.pt` | 80 | 79 | 0.812 | 0.395 | 0.464 | 0.210 | 3.7 |
| YOLOv8s | `yolov8s_baseline_640_batch8/weights/best.pt` | 80 | 79 | 0.756 | 0.434 | 0.502 | 0.266 | 4.1 |

On this test set, YOLOv8s has higher mAP (0.038 higher mAP@0.5 and 0.056 higher mAP@0.5:0.95) and recall, while YOLOv5s has higher precision and a lower reported inference time. These are descriptive results from one evaluation, not evidence that the differences are statistically significant or that either model is universally better.

The test set contains only 79 labelled objects across 80 images. It has no `wrinkle` instances and only four `twisted-meat` instances, so it cannot establish performance on all five classes; per-class results for rare classes are particularly uncertain. Inference timing is the single-run Ultralytics timing on the stated machine and should not be treated as a controlled benchmark. Evaluation artifacts are saved under `runs/detect/runs/test_eval/`.

The machine-readable summary is in [`results/test_evaluation_640.csv`](results/test_evaluation_640.csv), and selected plots are included under [`report/figures/test_eval/`](report/figures/test_eval/). The 8-page report is available as [`report/project_report.pdf`](report/project_report.pdf), with editable source in [`report/project_report.html`](report/project_report.html). Review and confirm the AI-reflection paragraph before submitting.

### Previously recorded experiment results (30 epochs each)

The values below are retained from the earlier project record, not presented as final independent-validation results. The old custom validation split overlapped with training, and the run metadata only records the mutable dataset YAML path rather than a snapshot of its contents. Re-run all comparisons with the disjoint split above and update this table before drawing final conclusions. Do not combine old and new metrics as if they used the same validation protocol.

| # | Model | imgsz | batch | mAP@0.5 | mAP@0.5:0.95 | Inference (ms) | Model Size |
|:---:|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| 1 | YOLOv5s | 640 | 16 | 0.334 | 0.195 | — | 18.5 MB |
| 2 | YOLOv8s | 640 | 16 | 0.187 | 0.0924 | — | 22.5 MB |
| 3 | YOLOv5s | 640 | 8 | 0.478 | 0.280 | 2.1 | 18.5 MB |
| 4 | YOLOv8s | 640 | 8 | **0.636*** | **0.421*** | 4.1 | 22.5 MB |
| 5 | YOLOv5s | 1280 | 8 | 0.499 | 0.295 | 9.4 | 18.5 MB |
| 6 | YOLOv8s | 1280 | 8 | **0.502** | **0.311** | 10.3 | 22.5 MB |

*Previously recorded values; custom-validation results are not independent because those images were also used for training.*

### Previously reported per-class results

These values came from the overlapping custom validation set and are not evidence of performance on unseen data. Recalculate them after retraining and evaluation with the disjoint split.

- **unsealed**: mAP50 = **0.972**
- **packaging-error**: mAP50 = **0.877**
- **loose-meat**: mAP50 = **0.737**
- **twisted-meat**: mAP50 = **0.540**
- **wrinkle**: mAP50 = **0.052**, Recall = **0** (previously reported; not independently validated).

### Previously reported observations (to be re-evaluated)

The following interpretations were based on the old runs. Treat them as hypotheses until the runs are repeated with a disjoint validation split and consistent measurement conditions.

1. **Batch size impact:** `batch=8` significantly outperforms `batch=16` (mAP improves from 0.334 to 0.478 for YOLOv5s; 0.187 to 0.636 for YOLOv8s).
2. **Resolution trade-off:** 1280 resolution increases inference time by 2.6–4.5x but does not consistently improve accuracy (YOLOv8s degraded by 14.5%). 640 resolution is recommended for deployment.
3. **Class imbalance limitations:** The failure of the `wrinkle` class aligns with the theoretical concepts of overfitting and data noise. Visual evidence shows all `wrinkle` instances were classified as background due to insufficient training data.

## Deliverables
1. Written project report (5–10 pages maximum; source and PDF in `/report`)
2. GitHub repository with reproducible workflow, source HTML documentation and tests
3. Individual 5-minute presentation plus 3-minute Q&A during the Week 14 examination period (book and attend a session)

Before submission, review the changed files, commit and push the final project files, submit the repository link and a zipped clone through Moodle, and book/attend the Week 14 presentation and Q&A. These external submission and attendance steps cannot be completed from this local workspace.


## AI Usage Note

The existing project record reports using DeepSeek V4 to assist with understanding YOLO training parameters and experiment design. Copilot was also used for this revision to compare the project against the brief, identify the train/validation overlap, and assist with changes to the split workflow, metrics, tests, and documentation. The student remains responsible for reviewing the code and results and for all final engineering decisions.

The project report records the tools used, their approximate contribution, how suggestions were checked, a methodological error found during validation review, which decisions remained the student's, and the student's learning reflection. Review that account for accuracy before submission.


## License

Dataset: CC BY 4.0 (Roboflow Pork Rasher Error Packaging v4).  
Code: MIT License (see LICENSE file).