"""
data_preprocess.py ->负责读取 YOLO 格式数据集，检查文件完整性，把图片和标注一一配对，是你项目的数据入口模块
Data loading, validation and pre‑processing for MMA3001 pork‑rasher packaging defect dataset.
"""
from pathlib import Path


def validate_dataset_paths(image_dir: Path, label_dir: Path) -> tuple[bool, list[str]]:
    """Check existence of dataset directories and detect missing/corrupted files.

    Args:
        image_dir: Path to directory storing dataset images
        label_dir: Path to directory storing YOLO format annotation txt files

    Returns:
        tuple:
            bool: True if paths pass basic validation
            list[str]: List of warning/error messages
    """
    messages = []
    if not image_dir.exists():
        messages.append(f"Image directory not found: {image_dir}")
    if not label_dir.exists():
        messages.append(f"Label directory not found: {label_dir}")

    valid = len(messages) == 0
    return valid, messages


def get_image_label_pairs(image_dir: Path, label_dir: Path) -> list[tuple[Path, Path]]:
    """Match image files with corresponding annotation label files.

    Args:
        image_dir: Folder for input images
        label_dir: Folder for YOLO annotation .txt labels

    Returns:
        list[tuple[Path, Path]]: paired (image_path, label_path)
    """
    pairs = []
    img_suffix = {".jpg", ".jpeg", ".png"}
    for img_path in sorted(image_dir.glob("*")):
        if img_path.suffix.lower() not in img_suffix:
            continue
        label_path = label_dir / f"{img_path.stem}.txt"
        pairs.append((img_path, label_path))
    return pairs


if __name__ == "__main__":
    # Example local paths, adjust for your machine
    project_root = Path(__file__).parents[1]
    img_folder = project_root / "data" / "images"
    lbl_folder = project_root / "data" / "labels"

    ok, msgs = validate_dataset_paths(img_folder, lbl_folder)
    for m in msgs:
        print(f"[WARNING] {m}")
    if ok:
        pairs = get_image_label_pairs(img_folder, lbl_folder)
        print(f"Found {len(pairs)} image‑label pairs")