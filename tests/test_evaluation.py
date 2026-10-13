"""Tests des métriques Data Science : calculs vérifiés à la main sur de petits exemples."""

from src.evaluation.metrics import DocumentAIMetricsCalculator as M
from src.evaluation.evaluator import aggregate_matching
from src.validation.schemas import MatchingResult, MatchingStatusEnum


def test_extraction_metrics():
    gt = {"cin_passeport": "05826391", "cylindree": 49, "numero_chassis": "9MWX05047", "date_vente": "11/02/2021"}
    pred = {"cin_passeport": "05826391", "cylindree": "49", "numero_chassis": "9MWX0504", "date_vente": None}
    r = M.evaluate_extraction(pred, gt)
    assert r["exact_matches"] == 2 and r["exact_match_ratio"] == 0.5
    assert r["missing_rate"] == 0.25 and r["error_rate"] == 0.25
    assert r["precision"] == round(2 / 3, 4) and r["recall"] == round(2 / 3, 4)


def test_matching_metrics():
    expected = {"cin_passeport": "MATCH", "nom_prenom_acheteur": "MATCH", "numero_chassis": "MISMATCH", "cylindree": "MATCH"}
    predicted = {"cin_passeport": "MATCH", "nom_prenom_acheteur": "REVIEW", "numero_chassis": "MISMATCH"}   # cylindrée jamais comparée
    r = M.evaluate_matching(predicted, expected)
    assert r["correct"] == 2 and r["accuracy"] == 0.5
    assert r["details"]["cylindree"]["predicted"] == "NOT_COMPARED"
    assert r["per_label"]["MISMATCH"]["f1_score"] == 1.0
    assert r["per_label"]["MATCH"]["precision"] == 1.0 and r["per_label"]["MATCH"]["recall"] == round(1 / 3, 4)


def test_fallback_improvement_is_measured_on_ground_truth():
    gt = {"cin_passeport": "05826391", "numero_chassis": "9MWX05047"}
    before = {"cin_passeport": "05826391"}
    after = {"cin_passeport": "05826391", "numero_chassis": "9MWX05047"}
    r = M.evaluate_fallback_improvement(before, after, gt)
    assert r["exact_match_before"] == 0.5 and r["exact_match_after"] == 1.0
    assert r["improvement"] == 0.5 and r["recovery_rate"] == 1.0


def test_fallback_recovery_rate_none_when_nothing_was_missing():
    gt = {"cin_passeport": "05826391"}
    r = M.evaluate_fallback_improvement({"cin_passeport": "05826391"}, {"cin_passeport": "05826391"}, gt)
    assert r["recovery_rate"] is None


def test_no_ground_truth_gives_error_not_numbers():
    assert "error" in M.evaluate_extraction({"cin_passeport": "1"}, {})
    assert "error" in M.evaluate_matching({"cin_passeport": "MATCH"}, {})


def test_aggregate_matching_takes_worst_status_and_groups_names():
    def r(field, status):
        return MatchingResult(field=field, document_a="a", document_b="b", status=status)
    agg = aggregate_matching([
        r("nom", MatchingStatusEnum.MATCH), r("prenom", MatchingStatusEnum.REVIEW),
        r("cin_passeport", MatchingStatusEnum.MATCH), r("cin_passeport", MatchingStatusEnum.MISMATCH),
    ])
    assert agg == {"nom_prenom_acheteur": "REVIEW", "cin_passeport": "MISMATCH"}
