from pathlib import Path

import pytest

from src.split_custom_data import create_stratified_split


def _make_training_data(root: Path):
    images_dir = root / "train" / "images"
    labels_dir = root / "train" / "labels"
    images_dir.mkdir(parents=True)
    labels_dir.mkdir()
    for index in range(30):
        (images_dir / f"image_{index:02}.jpg").write_bytes(b"image")
        class_id = 0 if index < 20 else 1
        (labels_dir / f"image_{index:02}.txt").write_text(
            f"{class_id} 0.5 0.5 0.2 0.2\n",
            encoding="utf-8",
        )
    return images_dir, labels_dir


def test_create_stratified_split_is_disjoint_and_reproducible(tmp_path):
    images_dir, labels_dir = _make_training_data(tmp_path)
    first_output = tmp_path / "first_split"
    second_output = tmp_path / "second_split"

    first_summary = create_stratified_split(images_dir, labels_dir, first_output)
    second_summary = create_stratified_split(images_dir, labels_dir, second_output)
    repeated_summary = create_stratified_split(images_dir, labels_dir, first_output)

    first_validation = {
        path.stem for path in (first_output / "valid" / "images").glob("*.jpg")
    }
    first_training_paths = [
        (first_output / "train.txt").parent / line
        for line in (first_output / "train.txt").read_text(encoding="utf-8").splitlines()
    ]
    first_training = {path.stem for path in first_training_paths}
    second_validation = {
        path.stem for path in (second_output / "valid" / "images").glob("*.jpg")
    }

    assert len(first_validation) == 6
    assert len(first_training) == 24
    assert first_training.isdisjoint(first_validation)
    assert second_validation == first_validation
    assert first_summary == second_summary
    assert first_summary == repeated_summary
    assert first_summary["validation_class_images"] == {0: 4, 1: 2}
    assert all(path.is_file() for path in first_training_paths)


def test_create_stratified_split_rejects_existing_output(tmp_path):
    images_dir, labels_dir = _make_training_data(tmp_path)
    output_dir = tmp_path / "existing"
    output_dir.mkdir()
    (output_dir / "keep.txt").write_text("keep", encoding="utf-8")

    with pytest.raises(FileExistsError):
        create_stratified_split(images_dir, labels_dir, output_dir)

    assert (output_dir / "keep.txt").read_text(encoding="utf-8") == "keep"


def test_create_stratified_split_rejects_invalid_fraction(tmp_path):
    images_dir, labels_dir = _make_training_data(tmp_path)

    with pytest.raises(ValueError, match="validation_fraction"):
        create_stratified_split(
            images_dir,
            labels_dir,
            tmp_path / "split",
            validation_fraction=1.0,
        )
