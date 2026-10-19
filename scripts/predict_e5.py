
"""Предсказание класса документа через E5."""

import argparse
from pathlib import Path

import joblib

from app.document_classes import CLASSES
from app.e5_features import (
    MODEL_NAME,
    TEXT_PREFIX,
    encode_texts,
    load_encoder,
)
from scripts.prepare_dataset import extract_text


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--file", type=Path, required=True
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=Path("models/e5_logreg.joblib"),
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.70,
    )
    parser.add_argument(
        "--device",
        choices=["cpu", "cuda"],
        default="cpu",
    )

    args = parser.parse_args()

    if not 0 <= args.threshold <= 1:
        parser.error(
            "Порог должен быть от 0 до 1"
        )

    # Загружаем только собственную доверенную модель.
    bundle = joblib.load(args.model)

    if (
        bundle["encoder_name"] != MODEL_NAME
        or bundle["text_prefix"] != TEXT_PREFIX
    ):
        raise ValueError(
            "Конфигурация эмбеддингов не совпадает "
            "с обучением"
        )

    text, _, _ = extract_text(args.file)

    if not text.strip():
        raise ValueError(
            "Не удалось извлечь текст из документа"
        )

    encoder = load_encoder(device=args.device)
    features = encode_texts(encoder, [text])

    classifier = bundle["classifier"]
    probabilities = classifier.predict_proba(features)[0]

    ranked = sorted(
        zip(classifier.classes_, probabilities),
        key=lambda pair: pair[1],
        reverse=True,
    )

    for label, probability in ranked:
        print(
            f"{CLASSES[label]:22} {probability:.2%}"
        )

    label, confidence = ranked[0]

    print(f"\nКласс: {CLASSES[label]}")
    print(
        "Нужна ручная проверка: "
        f"{confidence < args.threshold}"
    )
    print(
        "Вероятности ещё не откалиброваны "
        "на независимых реальных документах."
    )


if __name__ == "__main__":
    main()
