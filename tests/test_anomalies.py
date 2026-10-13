"""Anomalies FRA-01 à FRA-09 : signaux de risque, jamais des rejets."""

import json

from conftest import FakeVLMClient, field, make_readable_image, pair
from src.agent.orchestrator import SouscriptionAgentOrchestrator
from src.business.anomalies import detect_anomalies
from src.business.codes import ANOMALY_CODES, REJECTION_CODES
from src.database.repository import SouscriptionRepository
from src.evaluation.evaluator import DossierEvaluator
from src.evaluation.ground_truth import GroundTruthManager
from src.validation.schemas import (
    DocumentExtraction, ExtractedField, DocumentTypeEnum as T, ExtractionStatusEnum,
    MatchingResult, MatchingStatusEnum as M,
)
from src.vision.extraction import VLMExtractionService


def mr(field, status):
    return MatchingResult(field=field, document_a="a", document_b="b", value_a="1", value_b="2", status=status, message="m")


def codes(anoms):
    return sorted(a.code for a in anoms)


def test_code_tables_match_the_report():
    assert REJECTION_CODES == {"REJ-01": "DISPLACEMENT_ABOVE_LIMIT", "REJ-02": "DATE_CONDITION_NOT_MET",
                               "REJ-03": "BUSINESS_INELIGIBILITY"}
    assert [ANOMALY_CODES[f"FRA-0{i}"] for i in range(1, 10)] == [
        "CHASSIS_MISMATCH", "IDENTITY_MISMATCH", "CIN_MISMATCH", "DISPLACEMENT_CONFLICT", "DATE_INCONSISTENCY",
        "POSSIBLE_DOCUMENT_TAMPERING", "DUPLICATE_DOCUMENT", "UNKNOWN_DOCUMENT", "SUSPICIOUS_FIELD"]


def test_matching_conflicts_map_to_fra_01_to_05():
    results = [mr("numero_chassis", M.MISMATCH), mr("nom", M.MISMATCH), mr("cin_passeport", M.MISMATCH),
               mr("cylindree", M.MISMATCH), mr("date_vente", M.REVIEW), mr("prenom", M.MATCH)]
    anoms = detect_anomalies([], results)
    assert codes(anoms) == ["FRA-01", "FRA-02", "FRA-03", "FRA-04", "FRA-05"]
    assert next(a for a in anoms if a.code == "FRA-05").severity == "MEDIUM"       # REVIEW = signal moyen
    assert next(a for a in anoms if a.code == "FRA-01").severity == "HIGH"


def test_validation_issue_on_date_is_fra_05_other_is_fra_09():
    issues = [{"document": "d", "field": "date_vente", "issue": "Date invalide"},
              {"document": "d", "field": "cylindree", "issue": "hors plage"},
              {"document": "d", "field": None, "issue": "schéma"}]
    assert codes(detect_anomalies([], [], issues)) == ["FRA-05", "FRA-09"]


def test_unknown_document_only_when_it_was_actually_read():
    read = DocumentExtraction(document_id="a", filename="a", document_type=T.UNKNOWN)
    unreadable = DocumentExtraction(document_id="b", filename="b", document_type=T.UNKNOWN,
                                    extraction_status=ExtractionStatusEnum.UNREADABLE)
    assert codes(detect_anomalies([read, unreadable], [])) == ["FRA-08"]


def test_tampering_flag_comes_only_from_the_model_and_duplicates_from_files():
    ext = DocumentExtraction(document_id="a", filename="a.jpg", document_type=T.CONTRAT,
                             visual_flags=["numero_chassis : dernier chiffre surchargé"])
    anoms = detect_anomalies([ext], [], duplicate_groups=[["x.jpg", "y.jpg"]])
    assert codes(anoms) == ["FRA-06", "FRA-07"]


def test_contradictory_field_is_fra_09():
    ext = DocumentExtraction(document_id="a", filename="a", document_type=T.CONTRAT,
                             fields={"numero_chassis": ExtractedField(value="A", status="CONTRADICTORY", source_document="a")})
    assert codes(detect_anomalies([ext], [])) == ["FRA-09"]


# --- bout en bout ---

def _orch(tmp_path, client):
    repo = SouscriptionRepository(db_path=tmp_path / "t.db")
    return SouscriptionAgentOrchestrator(
        repository=repo, vlm_service=VLMExtractionService(client),
        evaluator=DossierEvaluator(GroundTruthManager(tmp_path / "absent.json"))), repo


GOOD = {"document_type": "CONTRAT", "fields": {"cin_passeport": field("05826391"), "nom_prenom_acheteur": field("A B"),
                                               "numero_chassis": field("9MWX05047"), "cylindree": field("49")}}


def test_tampering_signal_from_model_forces_review_and_is_stored(tmp_path):
    client = FakeVLMClient(classification={"document_type": "CONTRAT"}, extraction={**GOOD, "signes_alteration": ["rature sur le châssis"]})
    orch, repo = _orch(tmp_path, client)
    state = orch.process_dossier(pair(tmp_path), "DOS-T")
    assert state.status == "REVIEW_REQUIRED" and "FRA-06" in [a.code for a in state.anomalies]
    assert [a["code"] for a in repo.get_dossier_details("DOS-T")["anomalies"]] == ["FRA-06"]


def test_duplicate_file_is_flagged(tmp_path):
    cin = make_readable_image(tmp_path / "cin_doc.jpg", seed=9)
    a = make_readable_image(tmp_path / "a.jpg", seed=1)
    (tmp_path / "b.jpg").write_bytes((tmp_path / "a.jpg").read_bytes())
    orch, _ = _orch(tmp_path, FakeVLMClient(classification={"document_type": "CONTRAT"}, extraction=GOOD))
    state = orch.process_dossier([cin, a, str(tmp_path / "b.jpg")], "DOS-D")
    assert "FRA-07" in [x.code for x in state.anomalies] and state.status == "REVIEW_REQUIRED"


def test_clean_dossier_has_no_anomaly_and_is_approved(tmp_path):
    orch, _ = _orch(tmp_path, FakeVLMClient(classification={"document_type": "CONTRAT"}, extraction=GOOD))
    state = orch.process_dossier(pair(tmp_path), "DOS-OK")
    assert state.anomalies == [] and state.status == "ACCEPTED"
