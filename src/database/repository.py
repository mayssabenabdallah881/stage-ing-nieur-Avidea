"""
Couche d'accès aux données SQLite (Repository Pattern).
Utilise des requêtes paramétrées, des transactions et l'initialisation automatique.
"""

import json
import sqlite3
from pathlib import Path
from typing import Dict, List, Any, Optional
from src.config import DB_PATH
from src.agent.reconstruction import category_of
from src.database.models import SCHEMA_SQL, LEGACY_TABLE_RENAMES, ADDED_COLUMNS
from src.agent.state import DossierState


class SouscriptionRepository:
    """Gestionnaire de persistance relationnelle locale SQLite."""

    def __init__(self, db_path: Optional[Path] = None):
        self.db_path = str(db_path or DB_PATH)
        self.init_db()

    def _get_connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON;")
        return conn

    def init_db(self):
        """Initialise automatiquement les tables (et met à niveau une base créée par une version précédente)."""
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        with self._get_connection() as conn:
            self._migrate_legacy_tables(conn)
            conn.executescript(SCHEMA_SQL)
            self._migrate_legacy_columns_and_values(conn)

    @staticmethod
    def _migrate_legacy_tables(conn: sqlite3.Connection) -> None:
        """Renomme les anciennes tables avant la création du schéma, pour ne perdre aucune donnée."""
        existing = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type = 'table'")}
        for old, new in LEGACY_TABLE_RENAMES.items():
            if old in existing and new not in existing:
                conn.execute(f"ALTER TABLE {old} RENAME TO {new}")

    @staticmethod
    def _migrate_legacy_columns_and_values(conn: sqlite3.Connection) -> None:
        for table, column, ddl in ADDED_COLUMNS:
            cols = {r[1] for r in conn.execute(f"PRAGMA table_info({table})")}
            if column not in cols:
                conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {ddl}")
        # Bases créées avec la décision « APPROVED » (version intermédiaire) -> « ACCEPTED »
        conn.execute("UPDATE dossiers SET decision = 'ACCEPTED' WHERE decision = 'APPROVED'")
        conn.execute("UPDATE dossiers SET status = 'ACCEPTED' WHERE status = 'APPROVED'")
        conn.execute("UPDATE evaluation_results SET metrics_json = REPLACE(metrics_json, '\"APPROVED\"', '\"ACCEPTED\"')")

    def save_dossier(self, state: DossierState):
        """Enregistre l'ensemble des données d'un dossier dans une transaction atomique."""
        with self._get_connection() as conn:
            cur = conn.cursor()
            did = state.dossier_id

            # 1. dossiers
            dec_str = state.decision.decision.value if state.decision else state.status
            reason_str = state.decision.reason if state.decision else ""
            reconstruction = state.reconstruction.model_dump_json() if state.reconstruction else None

            cur.execute("""
                INSERT OR REPLACE INTO dossiers (
                    dossier_id, created_at, status, decision, reason, fallback_triggered, review_required,
                    reconstruction_json, rejection_type
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (did, state.created_at, state.status, dec_str, reason_str,
                  1 if state.fallback_triggered else 0, 1 if state.review_required else 0, reconstruction,
                  state.decision.rejection_type if state.decision else None))

            # Les tables filles sont réécrites à chaque enregistrement (INSERT OR REPLACE ne déclenche pas le CASCADE)
            for table in ("documents", "extractions", "scores", "matching", "business_rules",
                          "anomalies", "decisions", "agent_trace", "evaluation_results"):
                cur.execute(f"DELETE FROM {table} WHERE dossier_id = ?", (did,))

            # 2. documents & extractions
            for p, ext in state.extractions.items():
                q = state.quality_checks.get(p)
                cur.execute("""
                    INSERT INTO documents (
                        dossier_id, filename, document_type, category, quality_score, sharpness, is_readable,
                        fallback_used, extraction_status, classification_confidence, confidence
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (did, ext.filename, ext.document_type.value, category_of(ext.document_type),
                      q.quality_score if q else 0.0, q.sharpness if q else 0.0,
                      1 if (q and q.is_readable) else 0, 1 if ext.fallback_used else 0,
                      ext.extraction_status.value, ext.classification_confidence, ext.confidence))

                for f_name, f_obj in ext.fields.items():
                    cur.execute("""
                        INSERT INTO extractions (dossier_id, filename, field_name, value, raw_value, confidence, status)
                        VALUES (?, ?, ?, ?, ?, ?, ?)
                    """, (did, ext.filename, f_name, str(f_obj.value) if f_obj.value is not None else None,
                          f_obj.raw_value, f_obj.confidence, f_obj.status))

            # 3. scores
            sc = state.scores
            if sc:
                cur.execute("""
                    INSERT INTO scores (
                        dossier_id, missing_field_count, fields_total, extraction_score, completeness_score,
                        validity_score, identity_match_score, chassis_match_score, displacement_consistency,
                        date_consistency, global_data_confidence, identity_match, chassis_match,
                        displacement_consistent, date_consistent, components_used
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (did, sc.missing_field_count, sc.fields_total, sc.extraction_score, sc.completeness_score,
                      sc.validity_score, sc.identity_match_score, sc.chassis_match_score, sc.displacement_consistency,
                      sc.date_consistency, sc.global_data_confidence, sc.identity_match, sc.chassis_match,
                      sc.displacement_consistent, sc.date_consistent, json.dumps(sc.components_used)))

            # 4. matching
            for m in state.matching_results:
                cur.execute("""
                    INSERT INTO matching (
                        dossier_id, field, document_a, document_b, value_a, value_b, status, similarity, message
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (did, m.field, m.document_a, m.document_b, m.value_a, m.value_b,
                      m.status.value, m.similarity, m.message))

            # 5. business_rules
            for b in state.business_rule_results:
                cur.execute("""
                    INSERT INTO business_rules (dossier_id, rule_name, status, code, rejection_type, message)
                    VALUES (?, ?, ?, ?, ?, ?)
                """, (did, b.rule_name, b.status.value, b.code, b.rejection_type, b.message))

            # 6. anomalies
            for a in state.anomalies:
                cur.execute("""
                    INSERT INTO anomalies (dossier_id, code, name, severity, field, documents, message)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                """, (did, a.code, a.name, a.severity, a.field, json.dumps(a.documents, ensure_ascii=False), a.message))

            # 7. decisions
            if state.decision:
                d = state.decision
                cur.execute("""
                    INSERT INTO decisions (
                        dossier_id, decision, reason, rejection_codes, explanation_json, evidence_json, decided_at,
                        rejection_type
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """, (did, d.decision.value, d.reason, json.dumps(d.rejection_codes),
                      json.dumps(d.explanation, ensure_ascii=False),
                      json.dumps(d.evidence, ensure_ascii=False, default=str), state.created_at,
                      d.rejection_type))

            # 8. agent_trace
            for t in state.traces:
                cur.execute("""
                    INSERT INTO agent_trace (dossier_id, step, status, timestamp, duration_ms, message)
                    VALUES (?, ?, ?, ?, ?, ?)
                """, (did, t.step.value, t.status.value, t.timestamp, t.duration_ms, t.message))

            # 9. evaluation_results (uniquement si une vérité terrain a permis une vraie évaluation)
            if state.evaluation:
                cur.execute("""
                    INSERT INTO evaluation_results (dossier_id, created_at, metrics_json)
                    VALUES (?, ?, ?)
                """, (did, state.evaluation.get("evaluated_at", state.created_at),
                      json.dumps(state.evaluation, ensure_ascii=False)))

    def list_dossiers(self) -> List[Dict[str, Any]]:
        """Liste synthétique des dossiers pour les vues Dossiers / Historique (du plus récent au plus ancien)."""
        with self._get_connection() as conn:
            cur = conn.cursor()
            cur.execute("""
                SELECT
                    d.dossier_id,
                    d.created_at,
                    d.status,
                    d.decision,
                    d.reason,
                    d.fallback_triggered,
                    d.review_required,
                    d.rejection_type,
                    (SELECT COUNT(*) FROM documents doc WHERE doc.dossier_id = d.dossier_id) AS docs_count,
                    (SELECT AVG(doc.quality_score) FROM documents doc WHERE doc.dossier_id = d.dossier_id) AS avg_quality,
                    (SELECT COUNT(*) FROM anomalies a WHERE a.dossier_id = d.dossier_id) AS anomalies_count,
                    (SELECT s.global_data_confidence FROM scores s WHERE s.dossier_id = d.dossier_id) AS global_data_confidence
                FROM dossiers d
                ORDER BY d.created_at DESC
            """)
            return [dict(row) for row in cur.fetchall()]

    def get_dossier_details(self, dossier_id: str) -> Optional[Dict[str, Any]]:
        """Charge toutes les tables liées à un dossier spécifique."""
        with self._get_connection() as conn:
            cur = conn.cursor()
            cur.execute("SELECT * FROM dossiers WHERE dossier_id = ?", (dossier_id,))
            d_row = cur.fetchone()
            if not d_row:
                return None

            dossier = dict(d_row)
            recon = dossier.pop("reconstruction_json", None)
            dossier["reconstruction"] = json.loads(recon) if recon else None

            def rows(query: str) -> List[Dict[str, Any]]:
                cur.execute(query, (dossier_id,))
                return [dict(r) for r in cur.fetchall()]

            dossier["documents"] = rows("SELECT * FROM documents WHERE dossier_id = ? ORDER BY id")
            dossier["fields"] = rows("SELECT * FROM extractions WHERE dossier_id = ? ORDER BY id")
            dossier["matching"] = rows("SELECT * FROM matching WHERE dossier_id = ? ORDER BY id")
            dossier["rules"] = rows("SELECT * FROM business_rules WHERE dossier_id = ? ORDER BY id")
            dossier["traces"] = rows("SELECT * FROM agent_trace WHERE dossier_id = ? ORDER BY id ASC")

            anomalies = rows("SELECT * FROM anomalies WHERE dossier_id = ? ORDER BY id")
            for a in anomalies:
                a["documents"] = json.loads(a["documents"]) if a.get("documents") else []
            dossier["anomalies"] = anomalies

            cur.execute("SELECT * FROM scores WHERE dossier_id = ?", (dossier_id,))
            sc = cur.fetchone()
            if sc:
                scores = dict(sc)
                scores["components_used"] = json.loads(scores["components_used"]) if scores.get("components_used") else []
                dossier["scores"] = scores
            else:
                dossier["scores"] = None

            cur.execute("SELECT * FROM decisions WHERE dossier_id = ?", (dossier_id,))
            dec = cur.fetchone()
            if dec:
                details = dict(dec)
                details["rejection_codes"] = json.loads(details["rejection_codes"]) if details.get("rejection_codes") else []
                details["explanation"] = json.loads(details["explanation_json"]) if details.get("explanation_json") else []
                details["evidence"] = json.loads(details["evidence_json"]) if details.get("evidence_json") else {}
                for k in ("explanation_json", "evidence_json"):
                    details.pop(k, None)
                dossier["decision_details"] = details
            else:
                dossier["decision_details"] = None

            cur.execute("SELECT metrics_json FROM evaluation_results WHERE dossier_id = ? ORDER BY id DESC LIMIT 1", (dossier_id,))
            ev = cur.fetchone()
            dossier["evaluation"] = json.loads(ev["metrics_json"]) if ev else None

            return dossier

    def get_dashboard_stats(self) -> Dict[str, Any]:
        """Calcule les indicateurs clés directement depuis les tables SQLite."""
        with self._get_connection() as conn:
            cur = conn.cursor()

            def scalar(query: str) -> Any:
                cur.execute(query)
                return cur.fetchone()[0]

            total = scalar("SELECT COUNT(*) FROM dossiers") or 0
            fallback_count = scalar("SELECT COUNT(*) FROM dossiers WHERE fallback_triggered = 1") or 0
            avg_quality = scalar("SELECT AVG(quality_score) FROM documents")
            avg_confidence = scalar("SELECT AVG(global_data_confidence) FROM scores")
            avg_completeness = scalar("SELECT AVG(completeness_score) FROM scores")

            # Métriques d'évaluation : uniquement celles réellement calculées contre une vérité terrain
            cur.execute("SELECT metrics_json FROM evaluation_results")
            evals = [json.loads(r[0]) for r in cur.fetchall()]
            em = [e["extraction"]["exact_match_ratio"] for e in evals if "exact_match_ratio" in e.get("extraction", {})]
            dec = [e["decision"]["correct"] for e in evals if "decision" in e]

            def rnd(v: Optional[float], n: int = 4) -> Optional[float]:
                return round(v, n) if v is not None else None

            return {
                "total_dossiers": total,
                "accepted_count": scalar("SELECT COUNT(*) FROM dossiers WHERE decision = 'ACCEPTED'") or 0,
                "missing_document_count": scalar("SELECT COUNT(*) FROM dossiers WHERE decision = 'DOCUMENT_MANQUANT'") or 0,
                "rejected_count": scalar("SELECT COUNT(*) FROM dossiers WHERE decision = 'REJECTED'") or 0,
                "review_count": scalar("SELECT COUNT(*) FROM dossiers WHERE decision = 'REVIEW_REQUIRED'") or 0,
                "total_documents": scalar("SELECT COUNT(*) FROM documents") or 0,
                "anomalies_total": scalar("SELECT COUNT(*) FROM anomalies") or 0,
                "dossiers_with_anomalies": scalar("SELECT COUNT(DISTINCT dossier_id) FROM anomalies") or 0,
                "fallback_count": fallback_count,
                "fallback_rate": round(fallback_count / total * 100.0, 1) if total > 0 else 0.0,
                "avg_quality": rnd(avg_quality),
                "avg_global_data_confidence": rnd(avg_confidence),
                "avg_completeness": rnd(avg_completeness),
                "evaluated_dossiers": len(evals),
                "mean_exact_match": round(sum(em) / len(em), 4) if em else None,
                "decision_accuracy": round(sum(dec) / len(dec), 4) if dec else None,
            }
