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
