
"""Обучение TF-IDF + Logistic Regression."""

import argparse
import csv
import json
from pathlib import Path

import joblib
import sklearn

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.pipeline import Pipeline

from app.document_classes import CLASSES


LABELS = list(CLASSES)


def read_split(path: Path) -> list[dict]:
    with path.open(
        encoding="utf-8",
        newline="",
    ) as stream:
        rows = list(csv.DictReader(stream))

    if not rows:
        raise ValueError(f"Пустая выборка: {path}")

    for row in rows:
        if (
            row["label"] not in CLASSES
            or not row["text"].strip()
        ):
            raise ValueError(
                f"Некорректный пример в {path}: "
                f"{row.get('document_id')}"
            )

    return rows


def validate_splits(
    splits: dict[str, list[dict]]
) -> None:
    seen_groups = {}
    seen_texts = {}

    for split, rows in splits.items():
        labels = {r["label"] for r in rows}

        if labels != set(CLASSES):
            raise ValueError(
                f"В {split} должны быть все классы, "
                f"получено: {labels}"
            )

        for row in rows:
            group = row["group_id"]
            text_hash = row["text_sha256"]

            if (
                group in seen_groups
                and seen_groups[group] != split
            ):
                raise ValueError(
                    f"Утечка группы {group} в {split}"
                )

            if (
                text_hash in seen_texts
                and seen_texts[text_hash] != split
            ):
                raise ValueError(
                    "Дубликат текста между выборками: "
                    f"{text_hash}"
                )

            seen_groups[group] = split
            seen_texts[text_hash] = split


def evaluate(
    model,
    rows: list[dict],
) -> dict:

    actual = [
        row["label"]
        for row in rows
    ]

    predicted = model.predict([
        row["text"]
        for row in rows
    ]).tolist()

    return {
        "n_samples": len(rows),

        "accuracy": round(
            float(accuracy_score(actual, predicted)),
            4,
        ),

        "macro_f1": round(
            float(f1_score(
                actual,
                predicted,
                labels=LABELS,
                average="macro",
                zero_division=0,
            )),
            4,
        ),

        "weighted_f1": round(
            float(f1_score(
                actual,
                predicted,
                labels=LABELS,
                average="weighted",
                zero_division=0,
            )),
            4,
        ),

        "classification_report": classification_report(
            actual,
            predicted,
            labels=LABELS,
            output_dict=True,
            zero_division=0,
        ),

        "confusion_matrix": confusion_matrix(
            actual,
            predicted,
            labels=LABELS,
        ).tolist(),
    }


def train(
    data_dir: Path,
    model_path: Path,
    results_dir: Path,
) -> dict:

    splits = {
        name: read_split(
            data_dir / f"{name}.csv"
        )
        for name in ("train", "val", "test")
    }

    validate_splits(splits)

    model = Pipeline([
        (
            "tfidf",
            TfidfVectorizer(
                lowercase=True,
                ngram_range=(1, 2),
                min_df=1,
                max_features=30000,
                sublinear_tf=True,
            ),
        ),
        (
            "classifier",
            LogisticRegression(
                max_iter=2000,
                random_state=42,
                class_weight="balanced",
            ),
        ),
    ])

    model.fit(
        [
            row["text"]
            for row in splits["train"]
        ],
        [
            row["label"]
            for row in splits["train"]
        ],
    )

    metrics = {
        "model": "tfidf_logistic_regression",
        "dataset": (
            "synthetic (if created by generate_documents.py)"
        ),
        "scikit_learn_version": sklearn.__version__,
        "labels": LABELS,
        "train_samples": len(splits["train"]),
        "validation": evaluate(
            model, splits["val"]
        ),
        "test": evaluate(
            model, splits["test"]
        ),
    }

    model_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    results_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    joblib.dump(model, model_path)

    metrics_path = (
        results_dir / "tfidf_metrics.json"
    )

    metrics_path.write_text(
        json.dumps(
            metrics,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    matrix_path = (
        results_dir / "tfidf_confusion_matrix.csv"
    )

    with matrix_path.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as stream:
        writer = csv.writer(stream)

        writer.writerow([
            "true\\predicted",
            *LABELS,
        ])

        writer.writerows([
            [label, *row]
            for label, row in zip(
                LABELS,
                metrics["test"]["confusion_matrix"],
            )
        ])

    return metrics


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--data",
        type=Path,
        default=Path("data/processed"),
    )

    parser.add_argument(
        "--model",
        type=Path,
        default=Path("models/tfidf_logreg.joblib"),
    )

    parser.add_argument(
        "--results",
        type=Path,
        default=Path("artifacts"),
    )

    args = parser.parse_args()

    metrics = train(
        args.data,
        args.model,
        args.results,
    )

    for key in ("validation", "test"):
        result = metrics[key]

        print(
            f"{key}: n={result['n_samples']}, "
            f"accuracy={result['accuracy']:.4f}, "
            f"macro_f1={result['macro_f1']:.4f}"
        )

    print(f"Модель сохранена: {args.model}")

    print(
        "Метрики сохранены: "
        f"{args.results / 'tfidf_metrics.json'}"
    )


if __name__ == "__main__":
    main()
