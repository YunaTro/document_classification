"""Операции PostgreSQL; никаких таблиц семантического поиска/ChromaDB."""
from pathlib import Path
import time
from uuid import uuid4

import psycopg
from psycopg.rows import dict_row

from app.config import settings

SCHEMA_SQL = Path(__file__).with_name("schema.sql").read_text(encoding="utf-8")


def connect():
    return psycopg.connect(settings.postgres_dsn, row_factory=dict_row)


def wait_for_postgres(attempts: int = 30, pause: int = 2):
    for attempt in range(attempts):
        try:
            with connect():
                return
        except psycopg.OperationalError:
            if attempt == attempts - 1:
                raise
            time.sleep(pause)


def init_db():
    with connect() as conn:
        conn.execute(SCHEMA_SQL)


def add_document(record: dict, sections: list[dict]):
    with connect() as conn:
        conn.execute(
            """INSERT INTO documents (
                id, original_name, file_type, mime_type, file_size, sha256,
                s3_bucket, s3_key, extracted_text, extraction_status,
                has_ocr, ocr_confidence
            ) VALUES (
                %(id)s, %(original_name)s, %(file_type)s, %(mime_type)s,
                %(file_size)s, %(sha256)s, %(s3_bucket)s, %(s3_key)s,
                %(extracted_text)s, %(extraction_status)s,
                %(has_ocr)s, %(ocr_confidence)s
            )""", record,
        )
        for idx, section in enumerate(sections):
            conn.execute(
                """INSERT INTO document_sections (
                    id, document_id, section_index, section_type,
                    page_number, extracted_text, has_ocr, ocr_confidence
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s)""",
                (
                    str(uuid4()), record["id"], idx, section["section_type"],
                    section["page_number"], section["text"], section["has_ocr"],
                    section["ocr_confidence"],
                ),
            )


def list_documents(limit: int = 100):
    with connect() as conn:
        return conn.execute(
            """SELECT id, original_name, file_type, file_size, extraction_status,
                has_ocr, ocr_confidence, classification_status, predicted_class,
                predicted_confidence, verified_class, department, requires_review,
                created_at FROM documents ORDER BY created_at DESC LIMIT %s""",
            (limit,),
        ).fetchall()


def get_document(document_id: str):
    with connect() as conn:
        return conn.execute(
            "SELECT * FROM documents WHERE id = %s", (document_id,)
        ).fetchone()


def list_sections(document_id: str):
    with connect() as conn:
        return conn.execute(
            """SELECT section_index, section_type, page_number, extracted_text,
                has_ocr, ocr_confidence FROM document_sections
                WHERE document_id = %s ORDER BY section_index""", (document_id,),
        ).fetchall()


def get_routing_rules():
    with connect() as conn:
        return conn.execute(
            "SELECT document_class, department, enabled FROM routing_rules ORDER BY document_class"
        ).fetchall()

def save_classification(
    document_id: str,
    prediction: dict,
    threshold: float,
    auto_route: bool,
):
    """Сохраняет результат модели и обновляет документ."""

    from psycopg.types.json import Jsonb

    with connect() as conn:
        document = conn.execute(
            """SELECT id, verified_class
               FROM documents
               WHERE id = %s FOR UPDATE""",
            (document_id,),
        ).fetchone()

        if document is None:
            return None

        if document["verified_class"] is not None:
            raise ValueError(
                "Класс уже подтверждён; "
                "повторная классификация запрещена"
            )

        label = prediction["predicted_class"]

        can_route = (
            auto_route
            and prediction["confidence"] >= threshold
        )

        rule = (
            conn.execute(
                """SELECT department
                   FROM routing_rules
                   WHERE document_class = %s
                   AND enabled = TRUE""",
                (label,),
            ).fetchone()
            if can_route
            else None
        )

        department = rule["department"] if rule else None

        status = (
            "routed" if department else "pending_review"
        )
        requires_review = department is None

        conn.execute(
            """INSERT INTO classification_results (
                id, document_id, model_name,
                model_version, predicted_class,
                confidence, scores
            ) VALUES (%s, %s, %s, %s, %s, %s, %s)""",
            (
                str(uuid4()),
                document_id,
                prediction["model_name"],
                prediction["model_version"],
                label,
                prediction["confidence"],
                Jsonb(prediction["scores"]),
            ),
        )

        conn.execute(
            """UPDATE documents
               SET predicted_class = %s,
                   predicted_confidence = %s,
                   model_name = %s,
                   department = %s,
                   requires_review = %s,
                   classification_status = %s,
                   updated_at = NOW()
               WHERE id = %s""",
            (
                label,
                prediction["confidence"],
                prediction["model_name"],
                department,
                requires_review,
                status,
                document_id,
            ),
        )

    return {
        **prediction,
        "classification_status": status,
        "requires_review": requires_review,
        "department": department,
    }


def get_classification_history(document_id: str):
    with connect() as conn:
        return conn.execute(
            """SELECT id, model_name, model_version,
                      predicted_class, confidence,
                      scores, created_at
               FROM classification_results
               WHERE document_id = %s
               ORDER BY created_at DESC, id DESC""",
            (document_id,),
        ).fetchall()


def list_review_queue(limit: int = 100):
    with connect() as conn:
        return conn.execute(
            """SELECT id, original_name, predicted_class,
                      predicted_confidence, model_name,
                      created_at
               FROM documents
               WHERE classification_status = 'pending_review'
               AND requires_review = TRUE
               ORDER BY created_at DESC
               LIMIT %s""",
            (limit,),
        ).fetchall()


def confirm_classification(
    document_id: str,
    verified_class: str,
    comment: str | None,
):
    """Подтверждает класс и назначает подразделение."""

    with connect() as conn:
        document = conn.execute(
            """SELECT id, predicted_class,
                      classification_status
               FROM documents
               WHERE id = %s FOR UPDATE""",
            (document_id,),
        ).fetchone()

        if document is None:
            return None

        if document["classification_status"] != "pending_review":
            raise ValueError(
                "Документ не ожидает ручной проверки"
            )

        rule = conn.execute(
            """SELECT department
               FROM routing_rules
               WHERE document_class = %s
               AND enabled = TRUE""",
            (verified_class,),
        ).fetchone()

        department = rule["department"] if rule else None

        status = (
            "confirmed"
            if department
            else "confirmed_unrouted"
        )

        conn.execute(
            """INSERT INTO review_events (
                id, document_id, predicted_class,
                verified_class, comment
            ) VALUES (%s, %s, %s, %s, %s)""",
            (
                str(uuid4()),
                document_id,
                document["predicted_class"],
                verified_class,
                comment,
            ),
        )

        conn.execute(
            """UPDATE documents
               SET verified_class = %s,
                   department = %s,
                   requires_review = FALSE,
                   classification_status = %s,
                   updated_at = NOW()
               WHERE id = %s""",
            (
                verified_class,
                department,
                status,
                document_id,
            ),
        )

    return {
        "document_id": document_id,
        "verified_class": verified_class,
        "department": department,
        "classification_status": status,
        "requires_review": False,
    }

