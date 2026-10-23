"""
Unit tests for model_runner.py, mocked so no real training / dataset required
"""
from pathlib import Path
import csv
from types import SimpleNamespace
from unittest.mock import patch, MagicMock
from src.model_runner import train_and_evaluate, save_metric_csv


@patch("src.model_runner.YOLO")
def test_train_and_evaluate(mock_yolo_cls, tmp_path):
    mock_model = MagicMock()
    best_weights = tmp_path / "best.pt"
    mock_model.trainer = SimpleNamespace(best=best_weights)

    mock_val_result = MagicMock()
    mock_val_result.box.map50 = 0.85
    mock_val_result.box.map = 0.72
    mock_val_result.box.mp = 0.81
    mock_val_result.box.mr = 0.76
    mock_val_result.speed = {"inference": 25.0}
    mock_best_model = MagicMock()
    mock_best_model.val.return_value = mock_val_result
    mock_yolo_cls.side_effect = [mock_model, mock_best_model]

    fake_yaml = Path("data/dataset.yaml")
    metric = train_and_evaluate("yolov8s", fake_yaml, epochs=5)

    assert mock_yolo_cls.call_args_list[0].args == ("yolov8s.pt",)
    assert mock_yolo_cls.call_args_list[1].args == (str(best_weights),)
    mock_model.train.assert_called_once()
    mock_best_model.val.assert_called_once_with(data=fake_yaml, device="cpu")
    assert metric["model"] == "yolov8s.pt"
    assert metric["weights"] == str(best_weights)
    assert metric["mAP50"] == 0.85
    assert metric["mAP50-95"] == 0.72
    assert metric["precision"] == 0.81
    assert metric["recall"] == 0.76
    assert metric["inference_ms_per_image"] == 25.0


def test_save_metric_csv(tmp_path):
    # Test saving metrics to CSV
    metrics = [
        {"model": "yolov5s", "mAP50": 0.80, "mAP50-95": 0.68, "inference_ms_per_image": 30},
        {"model": "yolov8s", "mAP50": 0.85, "mAP50-95": 0.72, "inference_ms_per_image": 25}
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