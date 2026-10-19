"""Тесты API с подменой БД и S3; docker не обязателен."""
from io import BytesIO

import pytest
from docx import Document
from fastapi.testclient import TestClient

psycopg = pytest.importorskip("psycopg")
from app import main

client = TestClient(main.app)


def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["classification_ready"] is False


def test_classes():
    response = client.get("/classes")
    assert response.status_code == 200
    assert len(response.json()) == 6


def test_upload_rejects_unsupported():
    response = client.post(
        "/documents/upload", files={"file": ("info.txt", b"test", "text/plain")}
    )
    assert response.status_code == 415


def test_upload_saves_text_without_predicting(monkeypatch):
    stored = {}

    def fake_db_add(record, sections):
        stored["record"] = record
        stored["sections"] = sections

    monkeypatch.setattr(main.db, "add_document", fake_db_add)
    monkeypatch.setattr(main.storage, "upload_file", lambda path, key: None)

    doc = Document()
    doc.add_paragraph("Договор о поставке оборудования")
    bio = BytesIO()
    doc.save(bio)

    response = client.post(
        "/documents/upload",
        files={"file": ("sample.docx", bio.getvalue(), "application/vnd.openxmlformats-officedocument.wordprocessingml.document")},
    )
    assert response.status_code == 201, response.text
    result = response.json()
    assert result["classification_status"] == "not_classified"
    assert result["extraction_status"] == "ready_for_classification"
    assert "Договор" in stored["record"]["extracted_text"]
    assert "predicted_class" not in stored["record"]  # пока ничего не предсказываем
