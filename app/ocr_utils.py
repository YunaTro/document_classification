"""Адаптировано из предыдущего проекта: OCR и ОТДЕЛЬНАЯ оценка качества OCR (0–100)."""
from collections import defaultdict

from PIL import Image, ImageFilter, ImageOps
import pytesseract

from app.config import settings


def preprocess_image(image: Image.Image) -> Image.Image:
    image = image.convert("L")
    image = ImageOps.autocontrast(image)
    return image.filter(ImageFilter.MedianFilter(size=3))


def ocr_image_with_confidence(image: Image.Image) -> tuple[str, float | None]:
    if not settings.enable_ocr:
        return "", None
    data = pytesseract.image_to_data(
        preprocess_image(image),
        lang=settings.ocr_langs,
        config=f"--oem {settings.ocr_oem} --psm {settings.ocr_psm}",
        output_type=pytesseract.Output.DICT,
    )
    lines: dict[tuple[int, int, int], list[str]] = defaultdict(list)
    scores: list[float] = []
    for i, word_raw in enumerate(data.get("text", [])):
        word = str(word_raw).strip()
        try:
            confidence = float(data["conf"][i])
        except (TypeError, ValueError):
            continue
        if not word or confidence < 0:
            continue
        key = (
            int(data["block_num"][i]),
            int(data["par_num"][i]),
            int(data["line_num"][i]),
        )
        lines[key].append(word)
        scores.append(confidence)
    text = "\n".join(" ".join(lines[k]) for k in sorted(lines)).strip()
    return text, round(sum(scores) / len(scores), 2) if scores else None
