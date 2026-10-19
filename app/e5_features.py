
"""Преобразование текста в эмбеддинги multilingual-e5-small."""

import numpy as np

MODEL_NAME = "intfloat/multilingual-e5-small"
TEXT_PREFIX = "query: "


def load_encoder(device: str = "cpu"):
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(MODEL_NAME, device=device)


def encode_texts(
    encoder,
    texts: list[str],
    batch_size: int = 8,
) -> np.ndarray:
    """Создаёт L2-нормированные эмбеддинги текстов."""

    if not texts:
        raise ValueError("Список текстов пуст")

    if any(not text.strip() for text in texts):
        raise ValueError("Найден пустой текст")

    features = encoder.encode(
        [TEXT_PREFIX + text.strip() for text in texts],
        batch_size=batch_size,
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=True,
    )

    features = np.asarray(
        features, dtype=np.float32
    )

    if (
        features.ndim != 2
        or features.shape[0] != len(texts)
    ):
        raise ValueError(
            f"Неверная форма эмбеддингов: {features.shape}"
        )

    return features
