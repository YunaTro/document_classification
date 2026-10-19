-- Самостоятельная схема проекта маршрутизации. БД прежнего проекта не используется.
CREATE TABLE IF NOT EXISTS documents (
    id UUID PRIMARY KEY,
    original_name TEXT NOT NULL,
    file_type TEXT NOT NULL,
    mime_type TEXT,
    file_size BIGINT NOT NULL CHECK (file_size > 0),
    sha256 CHAR(64) NOT NULL,
    s3_bucket TEXT NOT NULL,
    s3_key TEXT NOT NULL UNIQUE,
    extracted_text TEXT NOT NULL DEFAULT '',
    extraction_status TEXT NOT NULL DEFAULT 'ready_for_classification',
    has_ocr BOOLEAN NOT NULL DEFAULT FALSE,
    ocr_confidence REAL CHECK (ocr_confidence BETWEEN 0 AND 100),
    predicted_class TEXT,
    predicted_confidence REAL CHECK (predicted_confidence BETWEEN 0 AND 1),
    verified_class TEXT,
    department TEXT,
    requires_review BOOLEAN NOT NULL DEFAULT FALSE,
    classification_status TEXT NOT NULL DEFAULT 'not_classified',
    model_name TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS document_sections (
    id UUID PRIMARY KEY,
    document_id UUID NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    section_index INT NOT NULL,
    section_type TEXT NOT NULL,
    page_number INT,
    extracted_text TEXT NOT NULL DEFAULT '',
    has_ocr BOOLEAN NOT NULL DEFAULT FALSE,
    ocr_confidence REAL CHECK (ocr_confidence BETWEEN 0 AND 100),
    UNIQUE(document_id, section_index)
);

-- История предсказаний для будущего сравнения нескольких классификаторов.
CREATE TABLE IF NOT EXISTS classification_results (
    id UUID PRIMARY KEY,
    document_id UUID NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    model_name TEXT NOT NULL,
    model_version TEXT NOT NULL,
    predicted_class TEXT NOT NULL,
    confidence REAL NOT NULL CHECK (confidence BETWEEN 0 AND 1),
    scores JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Действия пользователя; заполняется на следующем этапе.
CREATE TABLE IF NOT EXISTS review_events (
    id UUID PRIMARY KEY,
    document_id UUID NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    predicted_class TEXT,
    verified_class TEXT NOT NULL,
    comment TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Учебные правила маршрутизации, в дальнейшем управляются приложением.
CREATE TABLE IF NOT EXISTS routing_rules (
    document_class TEXT PRIMARY KEY,
    department TEXT NOT NULL,
    enabled BOOLEAN NOT NULL DEFAULT TRUE
);
INSERT INTO routing_rules (document_class, department) VALUES
    ('contract', 'Юридический отдел'),
    ('invoice', 'Бухгалтерия'),
    ('act', 'Бухгалтерия'),
    ('application', 'Отдел кадров'),
    ('memo', 'Руководство'),
    ('report', 'Руководство')
ON CONFLICT (document_class) DO NOTHING;

CREATE INDEX IF NOT EXISTS idx_documents_created_at ON documents(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_documents_requires_review ON documents(requires_review);
CREATE INDEX IF NOT EXISTS idx_sections_document ON document_sections(document_id);
CREATE INDEX IF NOT EXISTS idx_results_document ON classification_results(document_id);
