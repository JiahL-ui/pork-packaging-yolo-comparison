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
    # 适配Roboflow导出数据集结构：data/train、data/valid
    project_root = Path(__file__).parents[1]
    subsets = ["train", "valid"]

    total_pairs = 0
    for subset in subsets:
        print(f"\n==== Checking subset: {subset} ====")
        img_folder = project_root / "data" / subset / "images"
        lbl_folder = project_root / "data" / subset / "labels"

        ok, msgs = validate_dataset_paths(img_folder, lbl_folder)
        for m in msgs:
            print(f"[WARNING] {m}")

        if ok:
            pairs = get_image_label_pairs(img_folder, lbl_folder)
            # 过滤：只保留标签文件真实存在的配对
            exist_pairs = [(img, lbl) for img,lbl in pairs if lbl.exists()]
            print(f"Total found image candidates: {len(pairs)}")
            print(f"Valid image-label pairs (label exists): {len(exist_pairs)}")
            total_pairs += len(exist_pairs)

    print(f"\n==== Total valid pairs in train + valid: {total_pairs} ====")