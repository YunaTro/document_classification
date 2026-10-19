"""Простой рабочий интерфейс этапа 1; классификацию пока не имитируем."""
import os

import requests
import streamlit as st

API = os.getenv("API_BASE_URL", "http://api:8080").rstrip("/")
st.set_page_config(page_title="Классификация документов", page_icon="📄", layout="wide")
st.title("Автоматическая классификация и маршрутизация документов")
st.caption("Этап 1 — загрузка документов, извлечение текста и хранение. Обучение моделей — следующий этап.")


def request_json(method: str, path: str, **kwargs):
    response = requests.request(method, API + path, timeout=180, **kwargs)
    response.raise_for_status()
    return response.json()


tab_upload, tab_documents, tab_rules = st.tabs(["Загрузка", "Документы", "Классы и маршруты"])

with tab_upload:
    st.subheader("Загрузить документ")
    file = st.file_uploader("PDF, DOCX или изображение", type=["pdf", "docx", "png", "jpg", "jpeg", "tif", "tiff", "bmp"])
    if file and st.button("Загрузить и извлечь текст", type="primary"):
        try:
            result = request_json(
                "POST", "/documents/upload",
                files={"file": (file.name, file.getvalue(), file.type or "application/octet-stream")},
            )
            if result["extraction_status"] == "empty_text":
                st.warning("Документ сохранён, но извлечённый текст пуст.")
            else:
                st.success("Документ сохранён; текст извлечён.")
            st.json(result)
        except requests.RequestException as exc:
            st.error(f"Ошибка загрузки: {exc}")

with tab_documents:
    st.subheader("Загруженные документы")
    if st.button("Обновить список"):
        st.rerun()
    try:
        documents = request_json("GET", "/documents")
        if documents:
            st.dataframe(documents, use_container_width=True, hide_index=True)
            document_ids = {f"{x['original_name']} ({x['id']})": x["id"] for x in documents}
            selected = st.selectbox("Открыть документ", options=list(document_ids))
            if selected:
                identifier = document_ids[selected]
                detail = request_json("GET", f"/documents/{identifier}")
                st.text_area("Извлечённый текст", value=detail["extracted_text"], height=300)
                st.write("Статус классификации:", detail["classification_status"])
                if st.button("Получить ссылку на оригинал"):
                    link = request_json("GET", f"/documents/{identifier}/download-link")
                    st.link_button("Открыть оригинал (ссылка на 1 час)", link["url"])
        else:
            st.info("Документы ещё не загружены.")
    except requests.RequestException as exc:
        st.error(f"Backend недоступен: {exc}")

with tab_rules:
    st.subheader("Классы документов")
    st.info("Правила маршрутизации определены, но пока не применяются: сначала обучим классификатор.")
    try:
        names = {row["code"]: row["title"] for row in request_json("GET", "/classes")}
        rules = request_json("GET", "/routing-rules")
        for rule in rules:
            st.write(f"**{names.get(rule['document_class'], rule['document_class'])}** → {rule['department']}")
    except requests.RequestException as exc:
        st.error(f"Не удалось получить правила: {exc}")
