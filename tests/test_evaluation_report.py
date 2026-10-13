"""Évaluation du rapport (chapitre 9) : classification, normalisation, analyse d'erreurs, benchmark."""

import json

from conftest import FakeVLMClient, field, make_readable_image, pair
from src.agent.orchestrator import SouscriptionAgentOrchestrator
from src.database.repository import SouscriptionRepository
from src.evaluation.benchmark import aggregate_evaluations, render_latex_table, run_benchmark
from src.evaluation.error_analysis import ErrorAnalyzer, ErrorCategory
from src.evaluation.evaluator import DossierEvaluator
from src.evaluation.ground_truth import GroundTruthManager
from src.evaluation.metrics import DocumentAIMetricsCalculator as M
from src.vision.extraction import VLMExtractionService


def test_classification_metrics_by_hand():
    exp = ["CIN", "CIN", "CONTRAT", "CONTRAT", "QUITTANCE"]
    pred = ["CIN", "CONTRAT", "CONTRAT", "CONTRAT", "QUITTANCE"]
    r = M.evaluate_classification(pred, exp)
    assert r["accuracy"] == 0.8 and r["correct"] == 4
    assert r["per_class"]["CIN"]["precision"] == 1.0 and r["per_class"]["CIN"]["recall"] == 0.5
    assert r["per_class"]["CONTRAT"]["precision"] == round(2 / 3, 4) and r["per_class"]["CONTRAT"]["recall"] == 1.0
    assert r["per_class"]["CIN"]["f1_score"] == round(2 * 1.0 * 0.5 / 1.5, 4)
    assert r["confusion_matrix"]["CIN"] == {"CIN": 1, "CONTRAT": 1}                 # CIN confondue avec CONTRAT
    assert r["macro_recall"] == round((0.5 + 1.0 + 1.0) / 3, 4)


def test_classification_without_annotation_gives_error_not_numbers():
    assert "error" in M.evaluate_classification([], [])


def _gt(tmp_path, decision="ACCEPTED", chassis="9MWX05047"):
    f = tmp_path / "gt.json"
    f.write_text(json.dumps({"dossiers": [{
        "dossier_id": "DOS-GT", "expected_decision": decision,
        "expected_fields": {"cin_passeport": "05826391", "numero_chassis": chassis, "cylindree": 49},
        "expected_cross_matching": {},
        "documents": [{"file": "c.jpg", "document_type": "CONTRAT"}]}]}))
    return GroundTruthManager(f)


def _run(tmp_path, gt, extraction, classification="CONTRAT"):
    client = FakeVLMClient(classification={"document_type": classification},
                           extraction={"document_type": classification, "fields": extraction})
    orch = SouscriptionAgentOrchestrator(repository=SouscriptionRepository(db_path=tmp_path / "t.db"),
                                         vlm_service=VLMExtractionService(client), evaluator=DossierEvaluator(gt))
    return orch.process_dossier(pair(tmp_path), "DOS-GT")


GOOD = {"cin_passeport": field("05826391"), "nom_prenom_acheteur": field("A B"), "numero_chassis": field("9MWX05047"), "cylindree": field("49")}


def test_dossier_evaluation_contains_all_report_metrics(tmp_path):
    ev = _run(tmp_path, _gt(tmp_path), GOOD).evaluation
    assert ev["classification"]["accuracy"] == 1.0 and ev["extraction"]["exact_match_ratio"] == 1.0
    assert ev["normalization"]["rate"] == 1.0 and ev["decision"]["correct"] is True
    assert ev["completeness"] == 0.5 and ev["error_analysis"] == []


def test_classification_error_is_attributed(tmp_path):
    ev = _run(tmp_path, _gt(tmp_path), GOOD, classification="FACTURE").evaluation
    cats = [e["category"] for e in ev["error_analysis"]]
    assert ErrorCategory.CLASSIFICATION.value in cats


def test_extraction_error_and_decision_impact(tmp_path):
    bad = {**GOOD, "numero_chassis": field("ZZZ00000")}
    ev = _run(tmp_path, _gt(tmp_path), bad).evaluation
    cats = ErrorAnalyzer.summarize(ev["error_analysis"])
    assert cats.get(ErrorCategory.EXTRACTION.value) == 1


def test_normalization_error_is_distinguished_from_extraction_error():
    ev = {"extraction": {"details": {"cylindree": {"expected": 49, "predicted": 150, "status": "MISMATCH"}}},
          "normalization": {"errors": [{"field": "cylindree", "raw_value": "49 cc", "value": 150, "expected": 49}]}}
    assert [e["category"] for e in ErrorAnalyzer.analyze(ev)] == [ErrorCategory.NORMALIZATION.value]


def test_missing_field_on_unreadable_image_is_a_quality_error():
    ev = {"extraction": {"details": {"cin_passeport": {"expected": "1", "predicted": None, "status": "MISSING"}}},
          "quality": [{"file": "a.jpg", "quality_score": 0.1, "is_readable": False}]}
    assert ErrorAnalyzer.analyze(ev)[0]["category"] == ErrorCategory.IMAGE_QUALITY.value


def test_wrong_decision_with_clean_inputs_is_a_business_rule_error():
    ev = {"extraction": {"details": {}}, "matching": {"details": {}},
          "decision": {"expected": "REJECTED", "predicted": "ACCEPTED", "correct": False}}
    assert ErrorAnalyzer.analyze(ev)[0]["category"] == ErrorCategory.BUSINESS_RULE.value


def test_benchmark_aggregates_only_real_evaluations_and_writes_files(tmp_path, monkeypatch):
    gt = _gt(tmp_path)
    client = FakeVLMClient(classification={"document_type": "CONTRAT"}, extraction={"document_type": "CONTRAT", "fields": GOOD})
    files = pair(tmp_path)
    monkeypatch.setattr("src.evaluation.benchmark.get_dossier", lambda d: {"dossier_id": d})
    monkeypatch.setattr("src.evaluation.benchmark.dossier_paths", lambda d: files)
    orch = SouscriptionAgentOrchestrator(repository=SouscriptionRepository(db_path=tmp_path / "t.db"),
                                         vlm_service=VLMExtractionService(client), evaluator=DossierEvaluator(gt))
    out = tmp_path / "out"
    report = run_benchmark(orch, gt, out)
    assert report["dossiers_evaluated"] == 1 and report["classification"]["accuracy"] == 1.0
    assert report["extraction"]["exact_match"] == 1.0 and report["decision_accuracy"] == 1.0
    tex = (out / "evaluation_table.tex").read_text(encoding="utf-8")
    assert "Accuracy classification & 100,0" in tex and "Exact Match extraction" in tex
    assert json.loads((out / "evaluation_report.json").read_text(encoding="utf-8"))["limitation"]


def test_benchmark_refuses_offline_mock_results(tmp_path, monkeypatch):
    gt = _gt(tmp_path)
    orch = SouscriptionAgentOrchestrator(repository=SouscriptionRepository(db_path=tmp_path / "t.db"), evaluator=DossierEvaluator(gt))
    img = make_readable_image(tmp_path / "20260331_190558.jpeg")
    monkeypatch.setattr("src.evaluation.benchmark.get_dossier", lambda d: {})
    monkeypatch.setattr("src.evaluation.benchmark.dossier_paths", lambda d: [img])
    report = run_benchmark(orch, gt, tmp_path / "out")      # client réel en mode hors-ligne (conftest)
    assert report["dossiers_evaluated"] == 0 and "DOS-GT" in report["skipped_dossiers"]
    assert "non mesuré" in render_latex_table(report)


def test_empty_aggregate_has_no_numbers():
    assert "classification" not in aggregate_evaluations({}) and "extraction" not in aggregate_evaluations({})


def test_shipped_ground_truth_uses_report_vocabulary():
    gt = GroundTruthManager()
    for d in gt.list_annotated_dossiers():
        assert d["expected_decision"] in ("ACCEPTED", "REJECTED", "REVIEW_REQUIRED", "DOCUMENT_MANQUANT")
        assert "cin_passeport" in d["expected_fields"] and "nom_prenom_acheteur" in d["expected_fields"]
        assert d["documents"] and all(x["document_type"] for x in d["documents"])
