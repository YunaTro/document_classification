
"""Конфигурация системы классификации документов."""

from dataclasses import dataclass
import os


@dataclass(frozen=True)
class Settings:
    postgres_dsn: str = os.getenv(
        "POSTGRES_DSN",
        "postgresql://docuser:docpass@postgres:5432/classifier_db",
    )

    # RustFS предоставляет S3-совместимый API
    s3_endpoint_url: str = os.getenv(
        "S3_ENDPOINT_URL",
        "http://rustfs:9000",
    )
    s3_public_endpoint_url: str = os.getenv(
        "S3_PUBLIC_ENDPOINT_URL",
        "http://localhost:9200",
    )
    s3_bucket: str = os.getenv(
        "S3_BUCKET",
        "classified-documents",
    )

    aws_access_key_id: str = os.getenv(
        "AWS_ACCESS_KEY_ID", ""
    )
    aws_secret_access_key: str = os.getenv(
        "AWS_SECRET_ACCESS_KEY", ""
    )
    aws_region: str = os.getenv(
        "AWS_REGION", "us-east-1"
    )

    enable_ocr: bool = (
        os.getenv("ENABLE_OCR", "true").lower() == "true"
    )
    ocr_langs: str = os.getenv("OCR_LANGS", "rus+eng")
    ocr_psm: int = int(os.getenv("OCR_PSM", "6"))
    ocr_oem: int = int(os.getenv("OCR_OEM", "3"))
    ocr_min_text_chars: int = int(
        os.getenv("OCR_MIN_TEXT_CHARS", "30")
    )
    max_upload_mb: int = int(
        os.getenv("MAX_UPLOAD_MB", "20")
    )

    # Будет использоваться после обучения классификатора
    classification_threshold: float = float(
        os.getenv("CLASSIFICATION_THRESHOLD", "0.70")
    )


settings = Settings()
