"""
Test d'intégration bout-en-bout de l'AgentOrchestrator.
"""

from conftest import make_readable_image
from src.agent.orchestrator import SouscriptionAgentOrchestrator
from src.database.repository import SouscriptionRepository


def test_orchestrator_execution(tmp_path):
    # Base temporaire
    db_file = tmp_path / "test_underwriting.db"
    repo = SouscriptionRepository(db_path=db_file)
    orchestrator = SouscriptionAgentOrchestrator(repository=repo)

    # Création de 3 vraies images d'épreuve pour le loader
    files = []
    for fname in ["20260331_190558.jpeg", "20260331_190449.jpeg", "20260331_190508.jpeg"]:
        img_p = tmp_path / fname
        files.append(make_readable_image(img_p, seed=len(files)))

    state = orchestrator.process_dossier(files, dossier_id="TEST-DOS-001")
    assert state.dossier_id == "TEST-DOS-001"
    assert state.status in ["ACCEPTED", "REJECTED", "REVIEW_REQUIRED"]
    assert len(state.traces) >= 5

    # Vérification persistance SQLite
    loaded = repo.get_dossier_details("TEST-DOS-001")
    assert loaded is not None
    assert loaded["dossier_id"] == "TEST-DOS-001"
    assert len(loaded["documents"]) == 3
    assert len(loaded["traces"]) >= 5


def test_orchestrator_empty_dossier(tmp_path):
    db_file = tmp_path / "test_underwriting_empty.db"
    repo = SouscriptionRepository(db_path=db_file)
    orchestrator = SouscriptionAgentOrchestrator(repository=repo)

    state = orchestrator.process_dossier([], dossier_id="EMPTY-001")
    assert state.status == "REVIEW_REQUIRED"
    assert "Dossier incomplet" in state.decision.reason


# ---------------------------------------------------------------------------
# Workflow agentique : étapes, décisions intermédiaires, jamais de donnée inventée
# ---------------------------------------------------------------------------

import json

from conftest import FakeVLMClient, field, make_black_image, pair
from src.evaluation.evaluator import DossierEvaluator
from src.evaluation.ground_truth import GroundTruthManager
from src.vision.extraction import VLMExtractionService
from src.validation.schemas import StepEnum


def _orchestrator(tmp_path, client, evaluator=None):
    repo = SouscriptionRepository(db_path=tmp_path / "t.db")
    return SouscriptionAgentOrchestrator(
        repository=repo, vlm_service=VLMExtractionService(client),
        evaluator=evaluator or DossierEvaluator(GroundTruthManager(tmp_path / "absent.json")),
    ), repo


def _steps(state):
    return [t.step for t in state.traces]


def test_full_workflow_contains_all_steps(tmp_path):
    client = FakeVLMClient(
        classification={"document_type": "CONTRAT", "confidence": 0.9},
        extraction={"document_type": "CONTRAT", "fields": {
            "cin_passeport": field("05826391"), "nom_prenom_acheteur": field("Abdelkarim Habria"),
            "numero_chassis": field("9MWX05047"), "cylindree": field("49"),
        }},
    )
    orch, _ = _orchestrator(tmp_path, client)
    state = orch.process_dossier(pair(tmp_path), "DOS-X")

    for step in (StepEnum.RECEIVED, StepEnum.QUALITY_CHECK, StepEnum.CLASSIFICATION, StepEnum.EXTRACTION_VLM,
                 StepEnum.VALIDATION, StepEnum.NORMALIZATION, StepEnum.MATCHING, StepEnum.DATA_QUALITY,
                 StepEnum.BUSINESS_RULES, StepEnum.DECISION):
        assert step in _steps(state), step
    assert client.calls.count("classification") == 2 and client.calls.count("extraction") == 2   # 2 pièces, pas de fallback
    assert state.status == "ACCEPTED"
    assert state.data_quality["fields_present"] >= 4


def test_unusable_document_never_reaches_vlm(tmp_path):
    client = FakeVLMClient()
    orch, repo = _orchestrator(tmp_path, client)
    state = orch.process_dossier([make_black_image(tmp_path / "noir.jpg")], "DOS-BLACK")

    assert client.calls == []                                      # aucun appel au VLM
    assert state.status == "REVIEW_REQUIRED"
    assert any("inexploitable" in r for r in state.review_reasons)
    assert StepEnum.PREPROCESSING in _steps(state)
    assert repo.get_dossier_details("DOS-BLACK") is not None


def test_vlm_unavailable_gives_review_and_no_invented_values(tmp_path):
    client = FakeVLMClient(available=False)
    orch, _ = _orchestrator(tmp_path, client)
    state = orch.process_dossier([make_readable_image(tmp_path / "d.jpg")], "DOS-NOKEY")

    assert client.calls == []
    assert state.status == "REVIEW_REQUIRED"
    ext = next(iter(state.extractions.values()))
    assert all(f.value is None for f in ext.fields.values())


def test_fallback_recovers_missing_value(tmp_path):
    client = FakeVLMClient(
        classification={"document_type": "CIN"},
        extraction={"document_type": "CIN", "fields": {"nom_prenom_acheteur": field("Abdelkarim Habria")}},   # cin absent
        fallback={"fields": {"cin_passeport": field("05826391")}},
    )
    orch, _ = _orchestrator(tmp_path, client)
    state = orch.process_dossier([make_readable_image(tmp_path / "cin.jpg")], "DOS-FB")

    ext = next(iter(state.extractions.values()))
    assert state.fallback_triggered and ext.fallback_used and ext.attempts_count == 2   # 2 tentatives max
    assert ext.fields["cin_passeport"].value == "05826391"
    assert ext.first_attempt_fields["cin_passeport"].value is None            # la tentative 1 est conservée
    assert client.calls.count("fallback") == 1


def test_fallback_contradiction_requires_review(tmp_path):
    # Contrat sans cylindrée -> fallback ; la seconde passe donne un châssis différent de la première
    client = FakeVLMClient(
        classification={"document_type": "CONTRAT"},
        extraction={"document_type": "CONTRAT", "fields": {
            "cin_passeport": field("05826391"), "nom_prenom_acheteur": field("Abdelkarim Habria"), "numero_chassis": field("9MWX05047")}},
        fallback={"fields": {"cylindree": field("49"), "numero_chassis": field("ZZZ99999")}},
    )
    orch, _ = _orchestrator(tmp_path, client)
    state = orch.process_dossier(pair(tmp_path), "DOS-CONTRA")

    assert state.status == "REVIEW_REQUIRED"
    assert any("Contradiction" in r for r in state.review_reasons)


def test_classification_conflict_requires_review(tmp_path):
    client = FakeVLMClient(
        classification={"document_type": "CONTRAT"},
        extraction={"document_type": "QUITTANCE", "fields": {
            "cin_passeport": field("05826391"), "nom_prenom_acheteur": field("A B"), "numero_chassis": field("9MWX05047"),
            "cylindree": field("49")}},
    )
    orch, _ = _orchestrator(tmp_path, client)
    state = orch.process_dossier(pair(tmp_path), "DOS-CONFLICT")
    assert state.status == "REVIEW_REQUIRED"
    assert any("ambigu" in r for r in state.review_reasons)


def test_invalid_value_triggers_validation_review_and_is_kept(tmp_path):
    client = FakeVLMClient(
        classification={"document_type": "CONTRAT"},
        extraction={"document_type": "CONTRAT", "fields": {
            "cin_passeport": field("123"), "nom_prenom_acheteur": field("A B"), "numero_chassis": field("9MWX05047"),
            "cylindree": field("49")}},
    )
    orch, _ = _orchestrator(tmp_path, client)
    state = orch.process_dossier(pair(tmp_path), "DOS-INVALID")
    ext = next(e for e in state.extractions.values() if e.document_type.value == "CONTRAT")
    assert ext.fields["cin_passeport"].value == "123" and ext.fields["cin_passeport"].status == "AMBIGUOUS"   # signalé, jamais corrigé
    assert state.status == "REVIEW_REQUIRED"


def test_evaluation_with_ground_truth_is_computed_and_stored(tmp_path):
    gt_file = tmp_path / "gt.json"
    gt_file.write_text(json.dumps({"dossiers": [{
        "dossier_id": "DOS-GT",
        "expected_fields": {"cin_passeport": "05826391", "numero_chassis": "9MWX05047", "cylindree": 49, "nom_prenom_acheteur": "Abdelkarim Habria"},
        "expected_cross_matching": {"cin_passeport": "MATCH"},
        "expected_decision": "ACCEPTED",
    }]}))
    client = FakeVLMClient(
        classification={"document_type": "CONTRAT"},
        extraction={"document_type": "CONTRAT", "fields": {
            "cin_passeport": field("05826391"), "nom_prenom_acheteur": field("Abdelkarim Habria"),
            "numero_chassis": field("9MWX05047"), "cylindree": field("49")}},
    )
    orch, repo = _orchestrator(tmp_path, client, DossierEvaluator(GroundTruthManager(gt_file)))
    state = orch.process_dossier(pair(tmp_path), "DOS-GT")

    assert state.evaluation["extraction"]["exact_match_ratio"] == 1.0
    assert state.evaluation["decision"]["correct"] is True
    stats = repo.get_dashboard_stats()
    assert stats["evaluated_dossiers"] == 1 and stats["mean_exact_match"] == 1.0
    assert repo.get_dossier_details("DOS-GT")["evaluation"]["decision"]["expected"] == "ACCEPTED"


def test_offline_mock_results_are_never_evaluated(tmp_path):
    """Le mode test renvoie des valeurs écrites à la main : aucune métrique ne doit en être tirée."""
    repo = SouscriptionRepository(db_path=tmp_path / "m.db")
    orch = SouscriptionAgentOrchestrator(repository=repo)       # client réel -> mode hors-ligne (conftest)
    files = [make_readable_image(tmp_path / n, seed=i) for i, n in
             enumerate(["20260331_190558.jpeg", "20260331_190449.jpeg", "20260331_190508.jpeg"])]
    state = orch.process_dossier(files, "DOS-001-HABRIA")       # dossier qui possède une vérité terrain
    assert state.offline_mock_used is True
    assert state.evaluation is None
    assert repo.get_dashboard_stats()["evaluated_dossiers"] == 0
