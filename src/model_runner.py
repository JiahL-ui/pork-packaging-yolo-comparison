"""
model_runner.py
Train and compare YOLOv5 and YOLOv8 on pork packaging defect dataset
MMA3001 Project
1.加载 YOLOv5、YOLOv8 模型（ultralytics 库）
2.用我们data_preprocess读取的数据集，分别训练 / 评估两个模型
3.自动对比指标：mAP、每张图的推理耗时
4.把评估结果保存成 csv，输出到results/文件夹（匹配项目结构）
"""
from pathlib import Path

import pandas as pd
from ultralytics import YOLO


def train_and_evaluate(model_name: str, yaml_path: str, epochs: int = 30, device="cpu"):
    print(f"\n========== Start training {model_name} ==========")

    if not (model_name.endswith('.pt') or model_name.endswith('.yaml')):
        model_name += '.pt'

    model = YOLO(model_name)
    # 训练
    model.train(
        data=yaml_path,
        epochs=epochs,
        imgsz=640,
        project="runs/train",
        name=model_name,
        device=device,  # 修复：使用传入的device参数，不再硬编码0
        verbose=True
    )
    best_weights = Path(model.trainer.best)
    best_model = YOLO(str(best_weights))
    val_result = best_model.val(data=yaml_path, device=device)
    metrics = {
        "model": model_name,
        "weights": str(best_weights),
        "mAP50": val_result.box.map50,
        "mAP50-95": val_result.box.map,
        "precision": val_result.box.mp,
        "recall": val_result.box.mr,
        "inference_ms_per_image": val_result.speed["inference"],
    }
    return metrics

def save_metric_csv(metrics_list: list, save_path: str):
    """
    将评估指标列表保存为 CSV 文件。

    Args:
        metrics_list (list): 包含每个模型 metrics 字典的列表。
        save_path (str): CSV 文件的保存路径。
    """
    df = pd.DataFrame(metrics_list)
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(save_path, index=False)
    print(f"Saved comparison result to {save_path}")

if __name__ == "__main__":
    DATA_YAML = "data/dataset.yaml"
    # 选择模型：yolov5s, yolov8s
    model_list = [
        "yolov5s.yaml",
        "yolov8s.yaml"
    ]
    all_metrics = []
    # ===== 这里改device！有NVIDIA显卡写 device=0，无显卡写 device="cpu" =====
    device = 0

    # 创建results文件夹，把csv保存到results/下，匹配项目注释
    Path("results").mkdir(parents=True, exist_ok=True)
    save_csv_path = "results/model_comparison_result.csv"

    for m in model_list:
        metric = train_and_evaluate(m, DATA_YAML, epochs=30, device=device)
        all_metrics.append(metric)

    # 输出对比表格并保存csv，方便写报告

    df = pd.DataFrame(all_metrics)
    print("\n==== Model Comparison Summary ====")
    print(df)
    save_metric_csv(all_metrics, save_csv_path)  # 调用刚才写的函数