# MMA3001 Project
**Performance comparison of YOLOv5s and YOLOv8s for packaging defect detection on pork rasher trays**

> University project for MMA3001‑ Numerical Methods and Machine Learning
> Individual project assessing performance difference between YOLOv5s(baseline) and YOLOv8s on pork‑rasher packaging defect detection task.

## Project Overview
This project compares engineering‑oriented performance metrics including detection accuracy, inference runtime and GPU memory consumption.

Four defect categories: `loose‑meat`, `packaging‑error`, `twisted‑meat`, `unsealed`.
The dataset suffers severe class imbalance: two defect classes only contain 3 annotated samples.

Dataset: Official MMA3001 Dataset3: packaged pork‑rasher RGB images with bounding‑box annotations.

Project scope limitation:
- Only use the provided course dataset; no new real‑world factory data collected.
- No GUI or hardware deployment implemented.
- Focus on defect detection model comparison, not meat internal quality evaluation.

## Repository structure
```
pork‑packaging‑yolo‑comparison/
├── src/                     # Source python code
├── tests/                   # pytest unit tests
├── notebooks/               # Exploratory data analysis notebooks
├── docs/html/               # Auto‑generated code documentation from pdoc
├── data/                    # Dataset yaml + annotation txt labels (images are git‑ignored)
├── runs/                    # Native YOLO training outputs: csv logs, training curves, confusion matrix
├── results/                 # Experiment outputs, metrics csv, selected result figures
├── report/                  # Final project report PDF
├── venv/                    # Python virtual environment (ignored by git)
```

## Environment setup
```bash
# create virtual environment
python -m venv venv

# activate venv (Git‑Bash / MINGW64 windows)
source venv/Scripts/activate

# install dependencies
pip install torch ultralytics pytest pdoc pandas matplotlib
```
## Dataset instruction
Raw dataset images are not stored in this repository.
Download MMA3001 Dataset‑3 from course resources, place images and annotation files into local data/ folder.
Only YOLO‑format `.txt` annotation labels are tracked in this repository.

## Experiment Status
✅ Baseline finished: YOLOv5s & YOLOv8s trained for 30 epochs under identical hyper‑parameters (input size 640).
✅ Generated mAP, precision, recall metrics, training curves and confusion matrices.
🔜 Next: Ablation study of input resolution; failure‑case analysis.


## How to run

To be filled after code implementation.
```bash
# Run data pre‑processing
python src/data_preprocess.py

# Run unit tests
pytest tests/

# Generate source‑code HTML documentation
pdoc src -o docs/html
```

## Deliverables
1. Project written report (5‑10 pages max, inside /report)
2. GitHub repository with reproducible workflow
3. 5‑min presentation + 3‑min Q&A in week‑14 examination period

## AI usage note
All AI tool usage is documented inside the final project report’s AI‑reflection chapter, as required by MMA3001 project brief.