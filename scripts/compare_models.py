
"""Сравнение метрик двух моделей классификации."""

import argparse
import csv
import json
from pathlib import Path


def compare(results_dir: Path) -> list[dict]:
    paths = {
        "TF-IDF + LogisticRegression":
            results_dir / "tfidf_metrics.json",

        "E5 + LogisticRegression":
            results_dir / "e5_metrics.json",
    }

    metrics = {
        name: json.loads(
            path.read_text(encoding="utf-8")
        )
        for name, path in paths.items()
    }

    first, second = metrics.values()

    for split in ("validation", "test"):
        if (
            first[split]["n_samples"]
            != second[split]["n_samples"]
            or first["labels"] != second["labels"]
        ):
            raise ValueError(
                "Число документов или порядок "
                "классов различаются"
            )

    rows = []

    for model_name, result in metrics.items():
        for split in ("validation", "test"):
            values = result[split]

            rows.append({
                "model": model_name,
                "split": split,
                "n_samples": values["n_samples"],
                "accuracy": values["accuracy"],
                "macro_f1": values["macro_f1"],
                "weighted_f1": values["weighted_f1"],
            })

    return rows


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--results",
        type=Path,
        default=Path("artifacts"),
    )

    args = parser.parse_args()

    rows = compare(args.results)
    output = args.results / "models_comparison.csv"

    with output.open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(rows[0]),
        )
        writer.writeheader()
        writer.writerows(rows)

    print(
        f"{'Модель':30} {'Выборка':12} "
        f"{'Accuracy':>9} {'Macro F1':>9} "
        f"{'Weighted F1':>11}"
    )

    for row in rows:
        print(
            f"{row['model']:30} "
            f"{row['split']:12} "
            f"{row['accuracy']:9.4f} "
            f"{row['macro_f1']:9.4f} "
            f"{row['weighted_f1']:11.4f}"
        )

    print(f"Результат записан: {output}")


if __name__ == "__main__":
    main()
