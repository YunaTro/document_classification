
"""Инференс двух обученных моделей классификации."""

from functools import lru_cache
from pathlib import Path

import joblib

from app.document_classes import CLASSES
from app.e5_features import (
    MODEL_NAME,
    TEXT_PREFIX,
    encode_texts,
    load_encoder,
)

MODEL_DIR = Path("models")
AVAILABLE_MODELS = ("tfidf", "e5")


@lru_cache(maxsize=1)
def _tfidf_model():
    path = MODEL_DIR / "tfidf_logreg.joblib"
    if not path.is_file():
        raise FileNotFoundError(f"Нет модели: {path}")

    # Загружаем только созданные нами доверенные файлы.
    return joblib.load(path)


@lru_cache(maxsize=1)
def _e5_bundle():
    path = MODEL_DIR / "e5_logreg.joblib"
    if not path.is_file():
        raise FileNotFoundError(f"Нет модели: {path}")

    bundle = joblib.load(path)

    if (
        bundle.get("encoder_name") != MODEL_NAME
        or bundle.get("text_prefix") != TEXT_PREFIX
        or bundle.get("normalize_embeddings") is not True
    ):
        raise ValueError(
            "Параметры E5 не совпадают с обучением"
        )

    return bundle


@lru_cache(maxsize=1)
def _e5_encoder():
    return load_encoder(device="cpu")


def predict(text: str, model_name: str) -> dict:
    """Возвращает класс и оценки всех шести классов."""

    if model_name not in AVAILABLE_MODELS:
        raise ValueError(
            f"Неизвестная модель: {model_name}"
        )

    if not text or not text.strip():
        raise ValueError("Пустой текст документа")

    if model_name == "tfidf":
        classifier = _tfidf_model()
        probabilities = classifier.predict_proba([text])[0]
        labels = classifier.classes_
        version = "tfidf_logreg_v1"

    else:
        classifier = _e5_bundle()["classifier"]
        features = encode_texts(
            _e5_encoder(), [text]
        )
        probabilities = classifier.predict_proba(
            features
        )[0]
        labels = classifier.classes_
        version = "e5_logreg_v1"

    scores = {
        str(label): float(probability)
        for label, probability in zip(
            labels, probabilities
        )
    }

    if set(scores) != set(CLASSES):
        raise ValueError(
            "Набор классов модели не совпадает с проектом"
        )

    predicted_class = max(scores, key=scores.get)

    return {
        "model_name": model_name,
        "model_version": version,
        "predicted_class": predicted_class,
        "confidence": scores[predicted_class],
        "scores": scores,
    }
