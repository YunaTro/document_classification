
import numpy as np
import pytest

from app.e5_features import encode_texts


class FakeEncoder:
    def encode(self, texts, **kwargs):
        self.received = texts
        self.settings = kwargs

        return np.array(
            [[1.0, 0.0], [0.0, 1.0]],
            dtype=np.float32,
        )


def test_e5_prefix_and_normalization():
    encoder = FakeEncoder()

    output = encode_texts(
        encoder,
        ["Договор", "Счёт"],
        batch_size=2,
    )

    assert encoder.received == [
        "query: Договор",
        "query: Счёт",
    ]

    assert encoder.settings[
        "normalize_embeddings"
    ] is True

    assert output.shape == (2, 2)


def test_e5_rejects_empty_text():
    with pytest.raises(
        ValueError,
        match="пустой текст",
    ):
        encode_texts(
            FakeEncoder(),
            ["Договор", "  "],
        )
