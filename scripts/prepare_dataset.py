
"""Подготовка датасета для классификации документов."""

import argparse
import csv
from collections import Counter
from hashlib import sha256
import json
from pathlib import Path
import random
import re

from app.document_classes import CLASSES
from app.loaders import SUPPORTED_EXTENSIONS, clean_text, load_any


ALLOWED_EXTENSIONS = SUPPORTED_EXTENSIONS | {".txt"}

CSV_FIELDS = [
    "document_id",
    "label",
    "group_id",
    "source_path",
    "source_sha256",
    "text_sha256",
    "text",
    "n_chars",
    "has_ocr",
    "ocr_confidence",
    "split",
]


def extract_text(path: Path) -> tuple[str, bool, float | None]:
    """Извлекает текст с использованием существующих модулей."""

    if path.suffix.lower() == ".txt":
        text = path.read_text(encoding="utf-8-sig")
        return clean_text(text), False, None

    sections = load_any(str(path))

    text = clean_text(
        "\n\n".join(
            section["text"]
            for section in sections
            if section["text"]
        )
    )

    scores = [
        section["ocr_confidence"]
        for section in sections
        if section["ocr_confidence"] is not None
    ]

    confidence = (
        round(sum(scores) / len(scores), 2)
        if scores
        else None
    )

    has_ocr = any(section["has_ocr"] for section in sections)

    return text, has_ocr, confidence


def collect_documents(raw_dir: Path, min_chars: int) -> list[dict]:
    """Собирает документы и проверяет качество разметки."""

    problems = []
    rows = []

    seen_files = {}
    seen_texts = {}
    group_owner = {}

    for folder in sorted(raw_dir.iterdir()):
        if folder.is_dir() and folder.name not in CLASSES:
            problems.append(
                f"Неизвестный класс (папка): {folder.name}"
            )

    for label in CLASSES:
        folder = raw_dir / label

        if not folder.is_dir():
            problems.append(f"Отсутствует папка: {folder}")
            continue

        files = sorted(
            p for p in folder.rglob("*")
            if p.is_file() or p.is_symlink()
        )

        if not files:
            problems.append(f"Нет документов класса: {label}")

        for path in files:
            relative = path.as_posix()

            if path.is_symlink():
                problems.append(
                    f"Недопустимая символическая ссылка: {relative}"
                )
                continue

            if path.suffix.lower() not in ALLOWED_EXTENSIONS:
                problems.append(
                    f"Неподдерживаемый формат: {relative}"
                )
                continue

            # group_id__имя_документа.pdf
            group_id, separator, unique_part = (
                path.stem.partition("__")
            )

            if not separator or not group_id or not unique_part:
                problems.append(
                    f"Нужно имя <group_id>__<имя>: {relative}"
                )
                continue

            old_label = group_owner.setdefault(group_id, label)

            if old_label != label:
                problems.append(
                    f"Группа '{group_id}' относится к двум классам: "
                    f"{old_label}, {label}"
                )
                continue

            try:
                raw_hash = sha256(path.read_bytes()).hexdigest()

                text, has_ocr, ocr_confidence = extract_text(path)

            except Exception as exc:
                problems.append(
                    f"Ошибка чтения {relative}: {exc}"
                )
                continue

            if len(text.strip()) < min_chars:
                problems.append(
                    f"Слишком короткий текст ({len(text)} символов): "
                    f"{relative}"
                )
                continue

            normalized = re.sub(
                r"\s+", " ", text.casefold()
            ).strip()

            text_hash = sha256(
                normalized.encode("utf-8")
            ).hexdigest()

            if raw_hash in seen_files:
                problems.append(
                    f"Одинаковые файлы: "
                    f"{seen_files[raw_hash]} и {relative}"
                )
                continue

            if text_hash in seen_texts:
                problems.append(
                    f"Одинаковый текст: "
                    f"{seen_texts[text_hash]} и {relative}"
                )
                continue

            seen_files[raw_hash] = relative
            seen_texts[text_hash] = relative

            rows.append({
                "document_id": raw_hash[:20],
                "label": label,
                "group_id": group_id,
                "source_path": relative,
                "source_sha256": raw_hash,
                "text_sha256": text_hash,
                "text": text,
                "n_chars": len(text),
                "has_ocr": has_ocr,
                "ocr_confidence": (
                    ocr_confidence
                    if ocr_confidence is not None
                    else ""
                ),
                "split": "",
            })

    if problems:
        raise ValueError(
            "Ошибки в датасете:\n- " +
            "\n- ".join(problems)
        )

    return rows


def assign_splits(rows: list[dict], seed: int) -> None:
    """Разделяет данные по группам без пересечения шаблонов."""

    assignments = {}

    for label in CLASSES:
        groups = sorted({
            row["group_id"]
            for row in rows
            if row["label"] == label
        })

        if len(groups) < 3:
            raise ValueError(
                f"Класс '{label}': необходимо минимум "
                f"3 независимые группы, сейчас {len(groups)}"
            )

        random.Random(f"{seed}:{label}").shuffle(groups)

        val_count = max(1, round(len(groups) * 0.15))
        test_count = max(1, round(len(groups) * 0.15))

        if val_count + test_count >= len(groups):
            raise ValueError(
                f"Для класса '{label}' не осталось "
                "групп для обучения"
            )

        for index, group in enumerate(groups):
            if index < test_count:
                split = "test"
            elif index < test_count + val_count:
                split = "val"
            else:
                split = "train"

            assignments[group] = split

    for row in rows:
        row["split"] = assignments[row["group_id"]]


def write_csv(path: Path, records: list[dict]) -> None:
    with path.open(
        "w", encoding="utf-8", newline=""
    ) as file:
        writer = csv.DictWriter(
            file, fieldnames=CSV_FIELDS
        )
        writer.writeheader()
        writer.writerows(records)


def build_dataset(
    raw_dir: Path,
    out_dir: Path,
    *,
    seed: int = 42,
    min_chars: int = 50,
) -> dict:

    if not raw_dir.is_dir():
        raise ValueError(
            f"Нет каталога исходных документов: {raw_dir}"
        )

    if raw_dir.resolve() == out_dir.resolve():
        raise ValueError(
            "Входной и выходной каталоги должны различаться"
        )

    rows = collect_documents(raw_dir, min_chars)
    assign_splits(rows, seed)

    stats = {
        "seed": seed,
        "total": len(rows),
        "train": sum(r["split"] == "train" for r in rows),
        "val": sum(r["split"] == "val" for r in rows),
        "test": sum(r["split"] == "test" for r in rows),
        "by_class": {},
        "notes": [
            "Разделение выполняется по группам; "
            "фактические доли могут отличаться от 70/15/15.",
            "Тестовая выборка не используется "
            "для настройки гиперпараметров.",
        ],
    }

    for label in CLASSES:
        own = [
            row for row in rows
            if row["label"] == label
        ]

        counts = Counter(
            row["split"] for row in own
        )

        stats["by_class"][label] = {
            "total": len(own),
            "groups": len({
                row["group_id"] for row in own
            }),
            "train": counts["train"],
            "val": counts["val"],
            "test": counts["test"],
        }

        if len(own) < 30:
            stats["notes"].append(
                f"Класс {label}: всего {len(own)} документов. "
                "Желательно собрать 30–40."
            )

    out_dir.mkdir(parents=True, exist_ok=True)

    write_csv(out_dir / "dataset.csv", rows)

    for split in ("train", "val", "test"):
        write_csv(
            out_dir / f"{split}.csv",
            [
                row for row in rows
                if row["split"] == split
            ],
        )

    (out_dir / "stats.json").write_text(
        json.dumps(stats, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    return stats


def main():
    parser = argparse.ArgumentParser(
        description="Подготовка размеченного датасета"
    )

    parser.add_argument(
        "--raw",
        type=Path,
        default=Path("data/raw"),
    )

    parser.add_argument(
        "--out",
        type=Path,
        default=Path("data/processed"),
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    parser.add_argument(
        "--min-chars",
        type=int,
        default=50,
    )

    args = parser.parse_args()

    if args.min_chars < 1:
        parser.error(
            "--min-chars должен быть положительным"
        )

    try:
        stats = build_dataset(
            args.raw,
            args.out,
            seed=args.seed,
            min_chars=args.min_chars,
        )

    except ValueError as error:
        parser.exit(
            status=1,
            message=f"{error}\n",
        )

    print(
        json.dumps(stats, ensure_ascii=False, indent=2)
    )

    print(f"\nРезультаты сохранены в: {args.out}")


if __name__ == "__main__":
    main()
