
"""Тестирование подготовки датасета."""

import csv
from pathlib import Path

import pytest

from app.document_classes import CLASSES
from scripts.prepare_dataset import build_dataset


def make_data(root: Path, groups: int = 4):
    for label in CLASSES:
        directory = root / label
        directory.mkdir(parents=True)

        for group_index in range(groups):
            for item_index in range(2):
                name = (
                    f"{label}-tpl{group_index}"
                    f"__doc{item_index}.txt"
                )

                text = (
                    f"Тип документа {label}. "
                    f"Уникальная форма {group_index}, "
                    f"экземпляр {item_index}. "
                    f"Номер {label}-{group_index}-{item_index}; "
                    f"описание операции и реквизитов."
                )

                (directory / name).write_text(
                    text, encoding="utf-8"
                )


def load_csv(path: Path):
    with path.open(
        encoding="utf-8", newline=""
    ) as stream:
        return list(csv.DictReader(stream))


def test_prepare_dataset(tmp_path):
    raw = tmp_path / "raw"
    processed = tmp_path / "processed"

    make_data(raw)
    stats = build_dataset(raw, processed, seed=42)

    assert stats["total"] == len(CLASSES) * 4 * 2
    assert (
        stats["train"] + stats["val"] + stats["test"]
        == stats["total"]
    )

    rows = load_csv(processed / "dataset.csv")

    for label in CLASSES:
        samples = [
            r for r in rows if r["label"] == label
        ]
        assert {r["split"] for r in samples} == {
            "train", "val", "test"
        }

    group_splits = {}

    for row in rows:
        group_splits.setdefault(
            row["group_id"], set()
        ).add(row["split"])

    assert all(
        len(splits) == 1
        for splits in group_splits.values()
    )

    assert len({
        r["text_sha256"] for r in rows
    }) == len(rows)

    original = (
        processed / "dataset.csv"
    ).read_bytes()

    build_dataset(raw, processed, seed=42)

    assert (
        processed / "dataset.csv"
    ).read_bytes() == original


def test_refuses_duplicate_text(tmp_path):
    raw = tmp_path / "raw"
    make_data(raw)

    source = (
        raw / "contract" /
        "contract-tpl0__doc0.txt"
    )
    duplicate = (
        raw / "contract" /
        "contract-tpl1__duplicate.txt"
    )

    duplicate.write_text(
        source.read_text(encoding="utf-8"),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="Одинаковые файлы"):
        build_dataset(raw, tmp_path / "out")


def test_refuses_short_document(tmp_path):
    raw = tmp_path / "raw"
    make_data(raw)

    path = (
        raw / "invoice" /
        "invoice-tpl0__doc0.txt"
    )
    path.write_text("Счёт", encoding="utf-8")

    with pytest.raises(ValueError, match="Слишком короткий"):
        build_dataset(raw, tmp_path / "out")


def test_refuses_single_family(tmp_path):
    raw = tmp_path / "raw"
    make_data(raw, groups=1)

    with pytest.raises(
        ValueError,
        match="минимум 3 независимые группы",
    ):
        build_dataset(raw, tmp_path / "out")


def test_refuses_bad_filename(tmp_path):
    raw = tmp_path / "raw"
    make_data(raw)

    path = raw / "act" / "without_group.txt"
    path.write_text(
        "Достаточно длинный текст с описанием "
        "акта и условий приемки работ.",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="Нужно имя"):
        build_dataset(raw, tmp_path / "out")
