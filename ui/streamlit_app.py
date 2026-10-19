
"""Веб-интерфейс классификации и маршрутизации."""

import os

import requests
import streamlit as st


API = os.getenv(
    "API_BASE_URL", "http://api:8080"
).rstrip("/")

st.set_page_config(
    page_title="Классификация документов",
    page_icon="📄",
    layout="wide",
)

st.title("Классификация и маршрутизация документов")


def api_request(method: str, path: str, **kwargs):
    response = requests.request(
        method,
        API + path,
        timeout=300,
        **kwargs,
    )

    if not response.ok:
        try:
            detail = response.json().get(
                "detail", response.text
            )
        except ValueError:
            detail = response.text

        raise RuntimeError(
            f"{response.status_code}: {detail}"
        )

    return response.json()


def show_prediction(result: dict):
    classes = {
        c["code"]: c["title"]
        for c in api_request("GET", "/classes")
    }

    label = result["predicted_class"]

    st.success(
        f"Предсказание: {classes.get(label, label)}"
    )

    st.write(
        "Оценка модели:",
        f"{result['confidence']:.2%}",
    )

    st.write(
        "Статус:", result["classification_status"]
    )

    if result["requires_review"]:
        st.warning(
            "Нужна ручная проверка. "
            "Подразделение ещё не назначено."
        )
    else:
        st.info(
            f"Направление: {result['department']}"
        )

    for label, probability in sorted(
        result["scores"].items(),
        key=lambda pair: -pair[1],
    ):
        st.write(
            f"{classes.get(label, label)} "
            f"— {probability:.2%}"
        )
        st.progress(
            min(1.0, max(0.0, probability))
        )


try:
    health = api_request("GET", "/health")

    if not health["auto_route_enabled"]:
        st.info(
            "Автоматическая маршрутизация отключена. "
            "Требуется подтверждение пользователя."
        )

except (RuntimeError, requests.RequestException) as exc:
    st.error(f"Backend недоступен: {exc}")
    st.stop()


MODELS = {
    "tfidf": "TF-IDF + LogisticRegression",
    "e5": "E5 + LogisticRegression",
}


tab_upload, tab_documents, tab_review, tab_rules = st.tabs([
    "Загрузка",
    "Документы",
    "Ручная проверка",
    "Маршруты",
])


with tab_upload:
    selected_model = st.selectbox(
        "Классификатор",
        list(MODELS),
        format_func=lambda code: MODELS[code],
        key="upload_model",
    )

    incoming = st.file_uploader(
        "PDF, DOCX или изображение",
        type=[
            "pdf", "docx", "png", "jpg",
            "jpeg", "tif", "tiff", "bmp",
        ],
    )

    if incoming and st.button(
        "Загрузить и классифицировать",
        type="primary",
    ):
        try:
            uploaded = api_request(
                "POST",
                "/documents/upload",
                files={
                    "file": (
                        incoming.name,
                        incoming.getvalue(),
                        incoming.type
                        or "application/octet-stream",
                    )
                },
            )

            st.success(
                f"Файл загружен: {uploaded['id']}"
            )

            if uploaded["extraction_status"] == "empty_text":
                st.warning(
                    "Текст не извлечён. "
                    "Классификация невозможна."
                )
            else:
                prediction = api_request(
                    "POST",
                    f"/documents/{uploaded['id']}/classify",
                    params={"model": selected_model},
                )
                show_prediction(prediction)

        except (RuntimeError, requests.RequestException) as exc:
            st.error(str(exc))


with tab_documents:
    try:
        documents = api_request("GET", "/documents")

        if not documents:
            st.info("Нет загруженных документов.")

        else:
            st.dataframe(
                documents,
                use_container_width=True,
                hide_index=True,
            )

            options = {
                f"{d['original_name']} — {str(d['id'])[:8]}":
                    str(d["id"])
                for d in documents
            }

            selected = st.selectbox(
                "Выбери документ", list(options)
            )

            doc_id = options[selected]
            details = api_request(
                "GET", f"/documents/{doc_id}"
            )

            st.write(
                "Текущий статус:",
                details["classification_status"],
            )

            st.text_area(
                "Извлечённый текст",
                details["extracted_text"],
                height=250,
            )

            if details["classification_status"] not in (
                "confirmed", "confirmed_unrouted"
            ):
                model = st.selectbox(
                    "Модель",
                    list(MODELS),
                    format_func=lambda code: MODELS[code],
                    key="document_model",
                )

                if st.button(
                    "Классифицировать выбранный документ"
                ):
                    result = api_request(
                        "POST",
                        f"/documents/{doc_id}/classify",
                        params={"model": model},
                    )
                    show_prediction(result)

            if st.button("Показать историю предсказаний"):
                history = api_request(
                    "GET",
                    f"/documents/{doc_id}/classifications",
                )
                st.dataframe(
                    history,
                    use_container_width=True,
                )

            if st.button("Получить ссылку на оригинал"):
                link = api_request(
                    "GET",
                    f"/documents/{doc_id}/download-link",
                )
                st.link_button(
                    "Открыть оригинал", link["url"]
                )

    except (RuntimeError, requests.RequestException) as exc:
        st.error(str(exc))


with tab_review:
    st.subheader("Документы, требующие проверки")

    try:
        queue = api_request(
            "GET", "/documents/review-queue"
        )

        if not queue:
            st.success("Очередь пуста.")

        else:
            st.dataframe(
                queue,
                use_container_width=True,
                hide_index=True,
            )

            names = {
                f"{d['original_name']} — {str(d['id'])[:8]}":
                    str(d["id"])
                for d in queue
            }

            chosen = st.selectbox(
                "Документ для проверки",
                list(names),
            )

            doc_id = names[chosen]
            details = api_request(
                "GET", f"/documents/{doc_id}"
            )

            labels = {
                c["code"]: c["title"]
                for c in api_request("GET", "/classes")
            }

            st.caption(
                f"Модель: {details['model_name']}; "
                f"оценка: "
                f"{details['predicted_confidence']:.2%}"
            )

            st.text_area(
                "Текст документа",
                details["extracted_text"],
                height=260,
                key=f"review_{doc_id}",
            )

            codes = list(labels)
            predicted = details["predicted_class"]

            approved = st.selectbox(
                "Подтверждённый тип документа",
                codes,
                index=(
                    codes.index(predicted)
                    if predicted in codes
                    else 0
                ),
                format_func=lambda code: labels[code],
            )

            comment = st.text_input(
                "Комментарий (необязательно)"
            )

            if st.button(
                "Подтвердить и назначить маршрут",
                type="primary",
            ):
                result = api_request(
                    "POST",
                    f"/documents/{doc_id}/review",
                    json={
                        "verified_class": approved,
                        "comment": comment or None,
                    },
                )

                st.success(
                    "Подтверждено. Подразделение: "
                    f"{result['department'] or 'нет активного правила'}"
                )

                st.rerun()

    except (RuntimeError, requests.RequestException) as exc:
        st.error(str(exc))


with tab_rules:
    st.subheader("Правила маршрутизации")

    st.caption(
        "Правила хранятся в PostgreSQL."
    )

    try:
        labels = {
            c["code"]: c["title"]
            for c in api_request("GET", "/classes")
        }

        rules = api_request(
            "GET", "/routing-rules"
        )

        for rule in rules:
            code = rule["document_class"]
            department = rule["department"]
            enabled = rule["enabled"]

            st.write(
                f"**{labels.get(code, code)}** → "
                f"{department} "
                f"({'включено' if enabled else 'выключено'})"
            )

    except (RuntimeError, requests.RequestException) as exc:
        st.error(str(exc))
