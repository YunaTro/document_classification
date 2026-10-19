
from dataclasses import replace
from uuid import uuid4

from fastapi.testclient import TestClient

from app import main


client = TestClient(main.app)


def test_classification_and_review(monkeypatch):
    document_id = str(uuid4())

    monkeypatch.setattr(
        main,
        "settings",
        replace(
            main.settings,
            auto_route_enabled=False,
            classification_threshold=0.70,
        ),
    )

    document = {
        "id": document_id,
        "extracted_text": "Счёт на оплату услуг",
        "verified_class": None,
        "classification_status": "not_classified",
    }

    monkeypatch.setattr(
        main.db,
        "get_document",
        lambda _id: document,
    )

    prediction = {
        "model_name": "e5",
        "model_version": "e5_logreg_v1",
        "predicted_class": "invoice",
        "confidence": 0.3403,
        "scores": {
            "invoice": 0.3403,
            "contract": 0.12,
            "act": 0.13,
            "application": 0.13,
            "memo": 0.14,
            "report": 0.1397,
        },
    }

    monkeypatch.setattr(
        main.classification,
        "predict",
        lambda text, model: prediction,
    )

    def fake_save(doc_id, result, threshold, auto):
        assert doc_id == document_id
        assert auto is False
        assert threshold == 0.70

        return {
            **result,
            "classification_status": "pending_review",
            "requires_review": True,
            "department": None,
        }

    monkeypatch.setattr(
        main.db,
        "save_classification",
        fake_save,
    )

    response = client.post(
        f"/documents/{document_id}/classify",
        params={"model": "e5"},
    )

    assert response.status_code == 200

    result = response.json()

    assert result["predicted_class"] == "invoice"
    assert result["requires_review"] is True
    assert result["department"] is None

    monkeypatch.setattr(
        main.db,
        "confirm_classification",
        lambda doc_id, label, comment: {
            "document_id": doc_id,
            "verified_class": label,
            "department": "Бухгалтерия",
            "classification_status": "confirmed",
            "requires_review": False,
        },
    )

    response = client.post(
        f"/documents/{document_id}/review",
        json={
            "verified_class": "invoice",
            "comment": "Проверено",
        },
    )

    assert response.status_code == 200
    assert response.json()["department"] == "Бухгалтерия"


def test_invalid_review_class():
    response = client.post(
        f"/documents/{uuid4()}/review",
        json={"verified_class": "unknown"},
    )

    assert response.status_code == 422


def test_missing_document(monkeypatch):
    monkeypatch.setattr(
        main.db,
        "get_document",
        lambda _id: None,
    )

    response = client.post(
        f"/documents/{uuid4()}/classify"
    )

    assert response.status_code == 404
