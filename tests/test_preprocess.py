"""
Unit tests for data_preprocess.py using pytest
用 pytest 测试data_preprocess.py里的两个函数，验证逻辑是否正确，匹配你项目tests/目录规划
"""
from pathlib import Path
from src.data_preprocess import validate_dataset_paths, get_image_label_pairs


def test_validate_paths_not_exist():
    # 测试：文件夹不存在的情况，应该返回False+警告信息
    fake_img = Path("data/nonexist_images")
    fake_lbl = Path("data/nonexist_labels")
    is_valid, msg_list = validate_dataset_paths(fake_img, fake_lbl)
    assert is_valid is False
    assert len(msg_list) == 2


def test_validate_paths_exist(tmp_path):
    # pytest内置临时目录，自动创建真实文件夹
    img_dir = tmp_path / "images"
    lbl_dir = tmp_path / "labels"
    img_dir.mkdir()
    lbl_dir.mkdir()
    is_valid, msg_list = validate_dataset_paths(img_dir, lbl_dir)
    assert is_valid is True
    assert len(msg_list) == 0


def test_get_image_label_pairs(tmp_path):
    # 测试图片与标注配对逻辑
    img_dir = tmp_path / "images"
    lbl_dir = tmp_path / "labels"
    img_dir.mkdir()
    lbl_dir.mkdir()

    # 创建测试样本：1张图片+同名label
    (img_dir / "img01.jpg").write_text("")
    (lbl_dir / "img01.txt").write_text("")
    # 创建图片，没有对应的label（不会被配对）
    (img_dir / "img02.png").write_text("")

    pairs = get_image_label_pairs(img_dir, lbl_dir)
    assert len(pairs) == 2
    # 检查配对路径
    img_path, label_path = pairs[0]
    assert img_path.name == "img01.jpg"
    assert label_path.name == "img01.txt"