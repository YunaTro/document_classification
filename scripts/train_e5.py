
"""Обучение Logistic Regression на E5-эмбеддингах."""

import argparse
import csv
import json
from pathlib import Path
from time import perf_counter

import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)

from app.e5_features import (
    MODEL_NAME,
    TEXT_PREFIX,
    encode_texts,
    load_encoder,
)
from scripts.train_tfidf import (
    LABELS,
    read_split,
    validate_splits,
)


def evaluate(clf, features, rows: list[dict]) -> dict:
    actual = [row["label"] for row in rows]
    predicted = clf.predict(features).tolist()

    return {
        "n_samples": len(rows),
        "accuracy": round(
            float(accuracy_score(actual, predicted)), 4
        ),
        "macro_f1": round(
            float(f1_score(
                actual, predicted,
                labels=LABELS,
                average="macro",
                zero_division=0,
            )), 4
        ),
        "weighted_f1": round(
            float(f1_score(
                actual, predicted,
                labels=LABELS,
                average="weighted",
                zero_division=0,
            )), 4
        ),
        "classification_report": classification_report(
            actual, predicted,
            labels=LABELS,
            output_dict=True,
            zero_division=0,
        ),
        "confusion_matrix": confusion_matrix(
            actual, predicted, labels=LABELS
        ).tolist(),
    }


def train(
    data_dir: Path,
    model_path: Path,
    results_dir: Path,
    batch_size: int = 8,
    device: str = "cpu",
) -> dict:

    splits = {
        name: read_split(data_dir / f"{name}.csv")
        for name in ("train", "val", "test")
    }
    validate_splits(splits)

    print(f"Загрузка {MODEL_NAME} (устройство: {device})")
    encoder = load_encoder(device=device)

    features = {}
    timings = {}

    for name, rows in splits.items():
        print(
            f"Эмбеддинги для {name}: "
            f"{len(rows)} документов"
        )

        start = perf_counter()

        features[name] = encode_texts(
            encoder,
            [r["text"] for r in rows],
            batch_size=batch_size,
        )

        timings[f"encode_{name}_seconds"] = round(
            perf_counter() - start, 3
        )

    classifier = LogisticRegression(
        max_iter=2000,
        random_state=42,
        class_weight="balanced",
    )

    start = perf_counter()

    classifier.fit(
        features["train"],
        [r["label"] for r in splits["train"]],
    )

    timings["fit_seconds"] = round(
        perf_counter() - start, 3
    )

    metrics = {
        "model": "multilingual_e5_small_logistic_regression",
        "encoder_name": MODEL_NAME,
        "prefix": TEXT_PREFIX,
        "embedding_dim": int(
            features["train"].shape[1]
        ),
        "dataset": (
            "synthetic "
            "(if created by generate_documents.py)"
        ),
        "labels": LABELS,
        "train_samples": len(splits["train"]),
        "validation": evaluate(
            classifier,
            features["val"],
            splits["val"],
        ),
        "test": evaluate(
            classifier,
            features["test"],
            splits["test"],
        ),
        "timings_seconds": timings,
    }

    model_path.parent.mkdir(
        parents=True, exist_ok=True
    )
    results_dir.mkdir(
        parents=True, exist_ok=True
    )

    joblib.dump(
        {
            "classifier": classifier,
            "encoder_name": MODEL_NAME,
            "text_prefix": TEXT_PREFIX,
            "normalize_embeddings": True,
        },
        model_path,
    )

    (results_dir / "e5_metrics.json").write_text(
        json.dumps(
            metrics, indent=2, ensure_ascii=False
        ),
        encoding="utf-8",
    )

    with (results_dir / "e5_confusion_matrix.csv").open(
        "w", encoding="utf-8", newline=""
    ) as output:
        writer = csv.writer(output)

        writer.writerow(["true\\predicted", *LABELS])

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
        default=Path("models/e5_logreg.joblib"),
    )
    parser.add_argument(
        "--results",
        type=Path,
        default=Path("artifacts"),
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
    )
    parser.add_argument(
        "--device",
        choices=["cpu", "cuda"],
        default="cpu",
    )

    args = parser.parse_args()

    if args.batch_size < 1:
        parser.error(
            "--batch-size должен быть положительным"
        )

    metrics = train(
        args.data,
        args.model,
        args.results,
        args.batch_size,
        args.device,
    )

    for name in ("validation", "test"):
        result = metrics[name]

        print(
            f"{name}: n={result['n_samples']}, "
            f"accuracy={result['accuracy']:.4f}, "
            f"macro_f1={result['macro_f1']:.4f}"
        )

    print(f"Сохранена модель: {args.model}")
    print(
        "Сохранены метрики: "
        f"{args.results / 'e5_metrics.json'}"
    )


if __name__ == "__main__":
    main()
