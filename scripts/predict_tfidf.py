
"""Проверка классификатора на одном документе."""

import argparse
from pathlib import Path

import joblib

from app.document_classes import CLASSES
from scripts.prepare_dataset import extract_text


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--file",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--model",
        type=Path,
        default=Path("models/tfidf_logreg.joblib"),
    )

    parser.add_argument(
        "--threshold",
        type=float,
        default=0.70,
    )

    args = parser.parse_args()

    if not 0 <= args.threshold <= 1:
        parser.error(
            "--threshold должен быть от 0 до 1"
        )

    # Загружаем только собственную доверенную модель.
    model = joblib.load(args.model)

    text, _, _ = extract_text(args.file)

    if not text.strip():
        raise ValueError(
            "Из документа не извлечён текст"
        )

    probs = model.predict_proba([text])[0]

    classes = (
        model.named_steps["classifier"].classes_
    )

    ranked = sorted(
        zip(classes, probs),
        key=lambda pair: pair[1],
        reverse=True,
    )

    for label, probability in ranked:
        print(
            f"{CLASSES[label]:20} "
            f"{probability:.2%}"
        )

    predicted_class = ranked[0][0]
    confidence = ranked[0][1]

    print(
        f"Класс: {CLASSES[predicted_class]}"
    )

    print(
        "Нужна ручная проверка: "
        f"{confidence < args.threshold}"
    )

    print(
        "Внимание: вероятности ещё не "
        "откалиброваны на реальных данных."
    )


if __name__ == "__main__":
    main()
