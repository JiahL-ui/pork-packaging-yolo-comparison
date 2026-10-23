"""Create a reproducible, image-level multilabel train/validation split."""

import os
import random
import shutil
from pathlib import Path

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}


def _read_classes(label_path: Path) -> set[int]:
    classes = set()
    for line_number, line in enumerate(label_path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        fields = line.split()
        try:
            classes.add(int(fields[0]))
        except (IndexError, ValueError) as exc:
            raise ValueError(f"Invalid class id in {label_path}:{line_number}") from exc
    return classes


def _select_validation_images(records, validation_fraction: float, seed: int):
    if not 0 < validation_fraction < 1:
        raise ValueError("validation_fraction must be between 0 and 1.")
    if len(records) < 2:
        raise ValueError("At least two labelled images are required for a holdout split.")

    target_size = round(len(records) * validation_fraction)
    target_size = min(max(target_size, 1), len(records) - 1)
    rng = random.Random(seed)
    shuffled = list(records)
    rng.shuffle(shuffled)

    class_totals = {}
    for record in records:
        for class_id in record[2]:
            class_totals[class_id] = class_totals.get(class_id, 0) + 1

    desired_counts = {
        class_id: count * validation_fraction
        for class_id, count in class_totals.items()
    }
    selected = []
    selected_counts = {class_id: 0 for class_id in class_totals}
    remaining = list(shuffled)

    while len(selected) < target_size:
        deficits = {
            class_id: desired - selected_counts[class_id]
            for class_id, desired in desired_counts.items()
        }
        available_classes = [
            class_id
            for class_id, deficit in deficits.items()
            if deficit > 0 and any(class_id in record[2] for record in remaining)
        ]
        if not available_classes:
            break

        class_id = max(available_classes, key=lambda item: deficits[item])
        candidates = [record for record in remaining if class_id in record[2]]
        best_score = max(
            sum(max(deficits.get(label, 0), 0) for label in record[2])
            for record in candidates
        )
        best_candidates = [
            record
            for record in candidates
            if sum(max(deficits.get(label, 0), 0) for label in record[2]) == best_score
        ]
        chosen = rng.choice(best_candidates)
        selected.append(chosen)
        remaining.remove(chosen)
        for label in chosen[2]:
            selected_counts[label] += 1

    if len(selected) < target_size:
        selected.extend(rng.sample(remaining, target_size - len(selected)))

    return selected


def create_stratified_split(
    train_images_dir: Path = Path("data/train/images"),
    train_labels_dir: Path = Path("data/train/labels"),
    output_dir: Path = Path("data/independent_split"),
    validation_fraction: float = 0.2,
    seed: int = 42,
):
    """Create a holdout validation set and a training image list without overlap.

    Validation images and labels are copied into ``output_dir/valid``. The
    training set is written as an image-list file so the source training data
    can remain untouched while excluding validation images.
    """
    train_images_dir = Path(train_images_dir)
    train_labels_dir = Path(train_labels_dir)
    output_dir = Path(output_dir)

    if not train_images_dir.is_dir() or not train_labels_dir.is_dir():
        raise FileNotFoundError(
            f"Training image and label directories must exist: "
            f"{train_images_dir}, {train_labels_dir}"
        )
    image_files = sorted(
        path for path in train_images_dir.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )
    records = []
    missing_labels = []
    for image_path in image_files:
        label_path = train_labels_dir / f"{image_path.stem}.txt"
        if not label_path.is_file():
            missing_labels.append(image_path.name)
            continue
        records.append((image_path, label_path, _read_classes(label_path)))

    if missing_labels:
        raise ValueError(
            f"Found {len(missing_labels)} training images without labels; "
            f"first missing label: {missing_labels[0]}"
        )

    validation_records = _select_validation_images(records, validation_fraction, seed)
    validation_stems = {image_path.stem for image_path, _, _ in validation_records}
    training_records = [
        record for record in records if record[0].stem not in validation_stems
    ]

    expected_files = {Path("train.txt")}
    for image_path, label_path, _ in validation_records:
        expected_files.add(Path("valid") / "images" / image_path.name)
        expected_files.add(Path("valid") / "labels" / label_path.name)
    if output_dir.exists():
        existing_files = {
            path.relative_to(output_dir)
            for path in output_dir.rglob("*")
            if path.is_file()
        }
        if existing_files and existing_files != expected_files:
            raise FileExistsError(
                f"Split output directory contains files not generated by this "
                f"split: {output_dir}. Choose a new output directory."
            )

    validation_images_dir = output_dir / "valid" / "images"
    validation_labels_dir = output_dir / "valid" / "labels"
    validation_images_dir.mkdir(parents=True, exist_ok=True)
    validation_labels_dir.mkdir(parents=True, exist_ok=True)

    for image_path, label_path, _ in validation_records:
        shutil.copy2(image_path, validation_images_dir / image_path.name)
        shutil.copy2(label_path, validation_labels_dir / label_path.name)

    train_list_path = output_dir / "train.txt"
    train_list_path.write_text(
        "".join(
            f"{Path(os.path.relpath(image_path, output_dir)).as_posix()}\n"
            for image_path, _, _ in training_records
        ),
        encoding="utf-8",
    )

    class_counts = {}
    for _, _, class_ids in validation_records:
        for class_id in class_ids:
            class_counts[class_id] = class_counts.get(class_id, 0) + 1

    summary = {
        "training_images": len(training_records),
        "validation_images": len(validation_records),
        "validation_class_images": class_counts,
    }
    print(
        f"Created disjoint split: {summary['training_images']} training images, "
        f"{summary['validation_images']} validation images."
    )
    print(f"Training image list: {train_list_path}")
    print(f"Validation class image counts: {class_counts}")
    return summary


if __name__ == "__main__":
    create_stratified_split()
