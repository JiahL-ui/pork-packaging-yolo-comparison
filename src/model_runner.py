"""
model_runner.py
Train and evaluate YOLOv5 and YOLOv8 for packaging defect detection,
save comparison metrics to results folder.
1.加载 YOLOv5、YOLOv8 模型（ultralytics 库）
2.用我们data_preprocess读取的数据集，分别训练 / 评估两个模型
3.自动对比指标：mAP、推理耗时 FPS
4.把评估结果保存成 csv，输出到results/文件夹（匹配项目结构）
"""
from pathlib import Path
import csv
from ultralytics import YOLO
from src.data_preprocess import validate_dataset_paths


def train_and_evaluate(model_name: str, data_yaml_path: Path, epochs: int = 10):
    """
    Train YOLO model and run evaluation.
    Args:
        model_name: 'yolov5s' or 'yolov8s'
        data_yaml_path: path to YOLO dataset yaml config
        epochs: training epochs
    Returns:
        dict: evaluation metrics
    """
    model = YOLO(f"{model_name}.pt")
    # Train
    train_result = model.train(
        data=str(data_yaml_path),
        epochs=epochs,
        project="results",
        name=model_name,
        exist_ok=True
    )
    # Evaluate on validation set
    val_result = model.val()

    metrics = {
        "model": model_name,
        "mAP50": val_result.box.map50,
        "mAP50-95": val_result.box.map,
        "inference_fps": val_result.speed["inference"]
    }
    return metrics


def save_metric_csv(metrics_list: list[dict], out_csv: Path):
    """Save model comparison metrics into CSV file"""
    fieldnames = ["model", "mAP50", "mAP50-95", "inference_fps"]
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(metrics_list)


if __name__ == "__main__":
    # Project root
    root = Path(__file__).parents[1]
    data_yaml = root / "data" / "dataset.yaml"
    img_dir = root / "data" / "images"
    lbl_dir = root / "data" / "labels"

    # Validate dataset paths
    valid, msg = validate_dataset_paths(img_dir, lbl_dir)
    if not valid:
        for m in msg:
            print(f"[ERROR] {m}")
            exit(1)

    # Compare yolov5s and yolov8s
    model_list = ["yolov5s", "yolov8s"]
    all_metrics = []
    for m_name in model_list:
        print(f"\n===== Starting run for {m_name} =====")
        metric = train_and_evaluate(m_name, data_yaml, epochs=10)
        all_metrics.append(metric)

    # Save results
    csv_out = root / "results" / "model_comparison.csv"
    csv_out.parent.mkdir(exist_ok=True)
    save_metric_csv(all_metrics, csv_out)
    print(f"\nComparison results saved to: {csv_out}")