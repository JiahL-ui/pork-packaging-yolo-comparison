"""
Unit tests for model_runner.py, mocked so no real training / dataset required
"""
from pathlib import Path
import csv
from unittest.mock import patch, MagicMock
from src.model_runner import train_and_evaluate, save_metric_csv


@patch("src.model_runner.YOLO")
def test_train_and_evaluate(mock_yolo_cls):
    # Mock YOLO model and results
    mock_model = MagicMock()
    mock_yolo_cls.return_value = mock_model

    mock_val_result = MagicMock()
    mock_val_result.box.map50 = 0.85
    mock_val_result.box.map = 0.72
    mock_val_result.speed = {"inference": 25.0}
    mock_model.val.return_value = mock_val_result

    # Call function
    fake_yaml = Path("data/dataset.yaml")
    metric = train_and_evaluate("yolov8s", fake_yaml, epochs=5)

    # Assertions
    mock_yolo_cls.assert_called_once_with("yolov8s.pt")
    mock_model.train.assert_called_once()
    assert metric["model"] == "yolov8s"
    assert metric["mAP50"] == 0.85
    assert metric["mAP50-95"] == 0.72
    assert metric["inference_fps"] == 25.0


def test_save_metric_csv(tmp_path):
    # Test saving metrics to CSV
    metrics = [
        {"model": "yolov5s", "mAP50": 0.80, "mAP50-95": 0.68, "inference_fps": 30},
        {"model": "yolov8s", "mAP50": 0.85, "mAP50-95": 0.72, "inference_fps": 25}
    ]
    out_file = tmp_path / "output.csv"
    save_metric_csv(metrics, out_file)

    # Read back and verify
    with open(out_file, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    assert len(rows) == 2
    assert rows[0]["model"] == "yolov5s"
    assert rows[1]["model"] == "yolov8s"