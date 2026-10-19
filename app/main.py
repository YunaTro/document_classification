"""FastAPI — первый этап: приём, сохранение и извлечение текста.

Классификаторы будут добавлены на следующем этапе; сейчас отсутствуют
предсказания классов и автоматическая маршрутизация.
"""
from contextlib import asynccontextmanager
from hashlib import sha256
import logging
from pathlib import Path
from tempfile import TemporaryDirectory
from uuid import UUID, uuid4

from fastapi import FastAPI, File, HTTPException, UploadFile

from app import db, storage
from app.config import settings
from app.loaders import SUPPORTED_EXTENSIONS, load_any

logger = logging.getLogger(__name__)

CLASSES = {
    "contract": "Договор", "invoice": "Счёт", "act": "Акт",
    "application": "Заявление", "memo": "Служебная записка", "report": "Отчёт",
}


@asynccontextmanager
async def lifespan(_: FastAPI):
    db.wait_for_postgres()
    storage.wait_for_s3()
    db.init_db()
    storage.ensure_bucket()
    yield


app = FastAPI(
    title="Document Classification & Routing — Stage 1",
    description="Загрузка, извлечение текста, PostgreSQL и MinIO. Классификация — следующий этап.",
    version="0.1.0",
    lifespan=lifespan,
)


@app.get("/health")
def health():
    return {"status": "ok", "stage": "ingestion", "classification_ready": False}


@app.get("/classes")
def classes():
    return [{"code": code, "title": title} for code, title in CLASSES.items()]


@app.get("/routing-rules")
def routing_rules():
    return db.get_routing_rules()


@app.post("/documents/upload", status_code=201)
async def upload_document(file: UploadFile = File(...)):
    original_name = (file.filename or "").replace("\\", "/").rsplit("/", 1)[-1]
    extension = Path(original_name).suffix.lower()
    if extension not in SUPPORTED_EXTENSIONS:
        raise HTTPException(status_code=415, detail="Допускаются PDF, DOCX и изображения")

    max_bytes = settings.max_upload_mb * 1024 * 1024
    doc_id = str(uuid4())
    key = f"raw/{doc_id}{extension}"
    file_hash = sha256()
    size = 0

    with TemporaryDirectory(prefix="doc_classify_") as tmp_dir:
        path = Path(tmp_dir) / f"input{extension}"
        with path.open("wb") as stream:
            while chunk := await file.read(1024 * 1024):
                size += len(chunk)
                if size > max_bytes:
                    raise HTTPException(
                        status_code=413,
                        detail=f"Размер файла превышает {settings.max_upload_mb} МБ",
                    )
                stream.write(chunk)
                file_hash.update(chunk)
        if size == 0:
            raise HTTPException(status_code=422, detail="Пустой файл")

        try:
            sections = load_any(str(path))
        except Exception as exc:
            logger.warning("Не удалось извлечь текст из %s: %s", original_name, exc)
            raise HTTPException(status_code=422, detail="Не удалось прочитать документ") from exc

        extracted_text = "\n\n".join(s["text"] for s in sections if s["text"])
        ocr_scores = [s["ocr_confidence"] for s in sections if s["ocr_confidence"] is not None]
        has_ocr = any(s["has_ocr"] for s in sections)
        extraction_status = "ready_for_classification" if extracted_text else "empty_text"
        record = {
            "id": doc_id,
            "original_name": original_name,
            "file_type": extension.lstrip("."),
            "mime_type": file.content_type,
            "file_size": size,
            "sha256": file_hash.hexdigest(),
            "s3_bucket": settings.s3_bucket,
            "s3_key": key,
            "extracted_text": extracted_text,
            "extraction_status": extraction_status,
            "has_ocr": has_ocr,
            "ocr_confidence": round(sum(ocr_scores) / len(ocr_scores), 2) if ocr_scores else None,
        }
        try:
            storage.upload_file(str(path), key)
        except Exception as exc:
            logger.exception("Ошибка S3 при загрузке документа")
            raise HTTPException(status_code=503, detail="Хранилище файлов недоступно") from exc
        try:
            db.add_document(record, sections)
        except Exception as exc:
            logger.exception("Ошибка при сохранении документа в БД")
            try:
                storage.delete_file(key)
            except Exception:
                logger.exception("Не удалось удалить S3-объект после ошибки БД")
            raise HTTPException(status_code=503, detail="Не удалось сохранить документ в БД") from exc

    return {
        "id": doc_id,
        "original_name": original_name,
        "extraction_status": extraction_status,
        "characters": len(extracted_text),
        "sections": len(sections),
        "has_ocr": has_ocr,
        "ocr_confidence": record["ocr_confidence"],
        "classification_status": "not_classified",
    }


@app.get("/documents")
def documents(limit: int = 100):
    if not 1 <= limit <= 500:
        raise HTTPException(status_code=422, detail="limit от 1 до 500")
    return db.list_documents(limit=limit)


@app.get("/documents/{document_id}")
def document(document_id: UUID):
    result = db.get_document(str(document_id))
    if not result:
        raise HTTPException(status_code=404, detail="Документ не найден")
    return result


@app.get("/documents/{document_id}/sections")
def document_sections(document_id: UUID):
    if not db.get_document(str(document_id)):
        raise HTTPException(status_code=404, detail="Документ не найден")
    return db.list_sections(str(document_id))


@app.get("/documents/{document_id}/download-link")
def document_download_link(document_id: UUID):
    result = db.get_document(str(document_id))
    if not result:
        raise HTTPException(status_code=404, detail="Документ не найден")
    return {"url": storage.presigned_url(result["s3_key"]), "expires_in": 3600}
