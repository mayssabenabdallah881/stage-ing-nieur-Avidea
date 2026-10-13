"""Persistance SQLite (chapitre 12) et backend API (chapitre 4)."""

import sqlite3

from fastapi.testclient import TestClient

from conftest import FakeVLMClient, field, make_readable_image, pair
from src.agent.orchestrator import SouscriptionAgentOrchestrator
from src.database.repository import SouscriptionRepository
from src.evaluation.evaluator import DossierEvaluator
from src.evaluation.ground_truth import GroundTruthManager
from src.vision.extraction import VLMExtractionService

GOOD = {"cin_passeport": field("05826391"), "nom_prenom_acheteur": field("A B"), "numero_chassis": field("9MWX05047"), "cylindree": field("49")}


def _orch(tmp_path, repo):
    client = FakeVLMClient(classification={"document_type": "CONTRAT"}, extraction={"document_type": "CONTRAT", "fields": GOOD})
    return SouscriptionAgentOrchestrator(repository=repo, vlm_service=VLMExtractionService(client),
                                         evaluator=DossierEvaluator(GroundTruthManager(tmp_path / "absent.json")))


def test_schema_has_the_tables_of_the_report(tmp_path):
    repo = SouscriptionRepository(db_path=tmp_path / "t.db")
    with sqlite3.connect(repo.db_path) as c:
        tables = {r[0] for r in c.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        fks = c.execute("PRAGMA foreign_key_list(scores)").fetchall()
    assert {"dossiers", "documents", "extractions", "scores", "matching", "business_rules", "decisions", "agent_trace"} <= tables
    assert fks and fks[0][2] == "dossiers"


def test_every_table_is_filled_and_reloaded(tmp_path):
    repo = SouscriptionRepository(db_path=tmp_path / "t.db")
    state = _orch(tmp_path, repo).process_dossier(pair(tmp_path), "DOS-1")
    assert state.persistence_error is None
    d = repo.get_dossier_details("DOS-1")
    assert d["decision"] == "ACCEPTED" and d["decision_details"]["decision"] == "ACCEPTED"
    assert d["scores"]["completeness_score"] == 0.5 and {x["category"] for x in d["documents"]} == {"IDENTITE", "VEHICULE"}
    assert d["reconstruction"]["vehicule"] and d["traces"][-1]["step"] == "PERSISTENCE"
    stats = repo.get_dashboard_stats()
    assert stats["accepted_count"] == 1 and stats["anomalies_total"] == 0 and stats["avg_global_data_confidence"] is not None


def test_resaving_a_dossier_replaces_children(tmp_path):
    repo = SouscriptionRepository(db_path=tmp_path / "t.db")
    orch = _orch(tmp_path, repo)
    files = pair(tmp_path)
    orch.process_dossier(files, "DOS-1"); orch.process_dossier(files, "DOS-1")
    assert len(repo.get_dossier_details("DOS-1")["documents"]) == 2 and len(repo.list_dossiers()) == 1


def test_legacy_database_is_upgraded_without_losing_data(tmp_path):
    path = tmp_path / "old.db"
    with sqlite3.connect(path) as c:
        c.executescript("""
            CREATE TABLE dossiers (dossier_id TEXT PRIMARY KEY, created_at TEXT NOT NULL, status TEXT NOT NULL,
                decision TEXT, reason TEXT, fallback_triggered INTEGER DEFAULT 0, review_required INTEGER DEFAULT 0);
            CREATE TABLE extracted_fields (id INTEGER PRIMARY KEY AUTOINCREMENT, dossier_id TEXT NOT NULL, filename TEXT NOT NULL,
                field_name TEXT NOT NULL, value TEXT, raw_value TEXT, confidence REAL, status TEXT);
            CREATE TABLE business_rules (id INTEGER PRIMARY KEY AUTOINCREMENT, dossier_id TEXT NOT NULL, rule_name TEXT NOT NULL,
                status TEXT NOT NULL, message TEXT);
            INSERT INTO dossiers VALUES ('OLD', '2026-01-01', 'APPROVED', 'APPROVED', 'ok', 0, 0);
            INSERT INTO extracted_fields (dossier_id, filename, field_name, value) VALUES ('OLD', 'f.jpg', 'cin', '05826391');
        """)
    repo = SouscriptionRepository(db_path=path)
    d = repo.get_dossier_details("OLD")
    assert d["decision"] == "ACCEPTED" and d["status"] == "ACCEPTED"
    assert d["fields"][0]["value"] == "05826391"
    with sqlite3.connect(path) as c:
        assert "code" in {r[1] for r in c.execute("PRAGMA table_info(business_rules)")}


# --- API ---

def _client(tmp_path, monkeypatch):
    from src.api import main
    repo = SouscriptionRepository(db_path=tmp_path / "api.db")
    monkeypatch.setattr(main, "INPUT_DIR", tmp_path / "uploads")
    monkeypatch.setattr(main, "_repo", repo)
    monkeypatch.setattr(main, "_orchestrator", _orch(tmp_path, repo))
    return TestClient(main.app)


def test_api_upload_analyze_and_consult(tmp_path, monkeypatch):
    c = _client(tmp_path, monkeypatch)
    cin, img = pair(tmp_path, "scan.jpg")
    with open(img, "rb") as f, open(cin, "rb") as g:
        r = c.post("/dossiers", files=[("files", ("cin_doc.jpg", g.read(), "image/jpeg")),
                                       ("files", ("../../etc/scan.jpg", f.read(), "image/jpeg")), ("files", ("note.txt", b"x", "text/plain"))])
    assert r.status_code == 200
    body = r.json()
    assert body["decision"] == "ACCEPTED" and len(body["documents"]) == 2       # le .txt est ignoré
    assert any(d["extraction"]["numero_chassis"] == "9MWX05047" for d in body["documents"])
    assert (tmp_path / "uploads" / body["dossier_id"] / "scan.jpg").exists()     # nom nettoyé : aucun chemin

    assert c.get("/dossiers").json()[0]["dossier_id"] == body["dossier_id"]
    assert c.get(f"/dossiers/{body['dossier_id']}").json()["scores"]["completeness_score"] == 0.5
    assert c.get(f"/dossiers/{body['dossier_id']}/traces").json()
    assert c.get("/stats").json()["accepted_count"] == 1


def test_api_errors_and_health(tmp_path, monkeypatch):
    c = _client(tmp_path, monkeypatch)
    assert c.get("/dossiers/nope").status_code == 404
    assert c.post("/dossiers/samples/nope").status_code == 404
    assert c.post("/dossiers").status_code == 422
    assert c.get("/health").json()["model"] == "gemini-3.1-flash-lite"
    r = c.post("/dossiers", files=[("files", ("a.txt", b"x", "text/plain"))])       # aucun document recevable
    assert r.json()["decision"] == "REVIEW_REQUIRED"
    assert c.get(f"/dossiers/{r.json()['dossier_id']}").status_code == 200           # conservé dans l'historique


def test_rejection_type_is_persisted_and_listed(tmp_path):
    repo = SouscriptionRepository(db_path=tmp_path / "t.db")
    client = FakeVLMClient(classification={"document_type": "CONTRAT"}, extraction={"document_type": "CONTRAT", "fields": {
        **GOOD, "cylindree": field("150")}})
    orch = SouscriptionAgentOrchestrator(repository=repo, vlm_service=VLMExtractionService(client),
                                         evaluator=DossierEvaluator(GroundTruthManager(tmp_path / "absent.json")))
    state = orch.process_dossier(pair(tmp_path), "DOS-REJ")
    assert state.status == "REJECTED" and state.decision.rejection_type == "PUISSANCE_SUP_125"
    d = repo.get_dossier_details("DOS-REJ")
    assert d["rejection_type"] == "PUISSANCE_SUP_125" and d["decision_details"]["rejection_type"] == "PUISSANCE_SUP_125"
    assert any(r["rejection_type"] == "PUISSANCE_SUP_125" for r in d["rules"])
    assert repo.list_dossiers()[0]["rejection_type"] == "PUISSANCE_SUP_125"
