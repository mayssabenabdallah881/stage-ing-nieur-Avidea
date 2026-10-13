"""
Schéma relationnel SQLite de l'agent de souscription (rapport, chapitre 12).

Entités principales du rapport : dossiers, documents, extractions, scores, matching, business_rules,
decisions, agent_trace. Tables complémentaires : anomalies (signaux FRA-xx) et evaluation_results (métriques
calculées contre une vérité terrain).
Clés primaires, clés étrangères (ON DELETE CASCADE) et index ; initialisation automatique par le dépôt.
"""

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS dossiers (
    dossier_id TEXT PRIMARY KEY,
    created_at TEXT NOT NULL,
    status TEXT NOT NULL,
    decision TEXT,            -- copie de decisions.decision, pour les listes et le tableau de bord
    reason TEXT,
    fallback_triggered INTEGER DEFAULT 0,
    review_required INTEGER DEFAULT 0,
    reconstruction_json TEXT, -- identité / véhicule / autres + valeurs consolidées
    rejection_type TEXT       -- PUISSANCE_SUP_125 / PUISSANCE_SUP_49_APRES_2025 (si REJECTED)
);

CREATE TABLE IF NOT EXISTS documents (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    dossier_id TEXT NOT NULL,
    filename TEXT NOT NULL,
    document_type TEXT NOT NULL,
    category TEXT,
    quality_score REAL,
    sharpness REAL,
    is_readable INTEGER,
    fallback_used INTEGER DEFAULT 0,
    extraction_status TEXT,
    classification_confidence REAL,
    confidence REAL,
    FOREIGN KEY (dossier_id) REFERENCES dossiers(dossier_id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS extractions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    dossier_id TEXT NOT NULL,
    filename TEXT NOT NULL,
    field_name TEXT NOT NULL,
    value TEXT,
    raw_value TEXT,
    confidence REAL,
    status TEXT,
    FOREIGN KEY (dossier_id) REFERENCES dossiers(dossier_id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS scores (
    dossier_id TEXT PRIMARY KEY,
    missing_field_count INTEGER,
    fields_total INTEGER,
    extraction_score REAL,
    completeness_score REAL,
    validity_score REAL,
    identity_match_score REAL,
    chassis_match_score REAL,
    displacement_consistency REAL,
    date_consistency REAL,
    global_data_confidence REAL,
    identity_match INTEGER,
    chassis_match INTEGER,
    displacement_consistent INTEGER,
    date_consistent INTEGER,
    components_used TEXT,
    FOREIGN KEY (dossier_id) REFERENCES dossiers(dossier_id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS matching (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    dossier_id TEXT NOT NULL,
    field TEXT NOT NULL,
    document_a TEXT NOT NULL,
    document_b TEXT NOT NULL,
    value_a TEXT,
    value_b TEXT,
    status TEXT NOT NULL,
    similarity REAL,
    message TEXT,
    FOREIGN KEY (dossier_id) REFERENCES dossiers(dossier_id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS business_rules (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    dossier_id TEXT NOT NULL,
    rule_name TEXT NOT NULL,
    status TEXT NOT NULL,
    code TEXT,
    rejection_type TEXT,
    message TEXT,
    FOREIGN KEY (dossier_id) REFERENCES dossiers(dossier_id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS anomalies (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    dossier_id TEXT NOT NULL,
    code TEXT NOT NULL,
    name TEXT NOT NULL,
    severity TEXT,
    field TEXT,
    documents TEXT,
    message TEXT,
    FOREIGN KEY (dossier_id) REFERENCES dossiers(dossier_id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS decisions (
    dossier_id TEXT PRIMARY KEY,
    decision TEXT NOT NULL,
    reason TEXT,
    rejection_codes TEXT,
    explanation_json TEXT,
    evidence_json TEXT,
    decided_at TEXT,
    rejection_type TEXT,
    FOREIGN KEY (dossier_id) REFERENCES dossiers(dossier_id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS agent_trace (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    dossier_id TEXT NOT NULL,
    step TEXT NOT NULL,
    status TEXT NOT NULL,
    timestamp TEXT NOT NULL,
    duration_ms INTEGER,
    message TEXT,
    FOREIGN KEY (dossier_id) REFERENCES dossiers(dossier_id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS evaluation_results (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    dossier_id TEXT NOT NULL,
    created_at TEXT NOT NULL,
    metrics_json TEXT NOT NULL,
    FOREIGN KEY (dossier_id) REFERENCES dossiers(dossier_id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_evaluation_dossier ON evaluation_results(dossier_id);
CREATE INDEX IF NOT EXISTS idx_dossiers_status ON dossiers(status);
CREATE INDEX IF NOT EXISTS idx_dossiers_decision ON dossiers(decision);
CREATE INDEX IF NOT EXISTS idx_documents_dossier ON documents(dossier_id);
CREATE INDEX IF NOT EXISTS idx_extractions_dossier ON extractions(dossier_id);
CREATE INDEX IF NOT EXISTS idx_matching_dossier ON matching(dossier_id);
CREATE INDEX IF NOT EXISTS idx_rules_dossier ON business_rules(dossier_id);
CREATE INDEX IF NOT EXISTS idx_anomalies_dossier ON anomalies(dossier_id);
CREATE INDEX IF NOT EXISTS idx_trace_dossier ON agent_trace(dossier_id);
"""

# Anciennes tables (versions précédentes du prototype) -> noms du rapport
LEGACY_TABLE_RENAMES = {
    "extracted_fields": "extractions",
    "matching_results": "matching",
    "agent_traces": "agent_trace",
}

# Colonnes ajoutées depuis : (table, colonne, définition) — appliquées aux bases existantes
ADDED_COLUMNS = [
    ("dossiers", "reconstruction_json", "TEXT"),
    ("documents", "category", "TEXT"),
    ("documents", "extraction_status", "TEXT"),
    ("documents", "classification_confidence", "REAL"),
    ("documents", "confidence", "REAL"),
    ("business_rules", "code", "TEXT"),
    ("business_rules", "rejection_type", "TEXT"),
    ("dossiers", "rejection_type", "TEXT"),
    ("decisions", "rejection_type", "TEXT"),
]
