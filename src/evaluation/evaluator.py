"""
Évaluation d'un dossier traité : Ground Truth -> Prédiction -> Métriques (rapport, chapitre 9).
Aucune métrique n'est produite sans vérité terrain, ni pour des extractions issues du mode hors-ligne de test.
"""

from datetime import datetime
from typing import Any, Dict, List, Optional

from src.agent.reconstruction import consolidate_fields
from src.agent.state import DossierState
from src.evaluation.ground_truth import GroundTruthManager
from src.evaluation.metrics import DocumentAIMetricsCalculator as Metrics
from src.validation.schemas import DocumentExtraction, MatchingResult
from src.vision.extraction import _normalize

NAME_FIELDS = {"nom", "prenom", "nom_prenom_acheteur"}


def aggregate_matching(results: List[MatchingResult]) -> Dict[str, str]:
    """Statut global par champ : MISMATCH > REVIEW > MATCH. nom/prenom sont regroupés sous 'nom_prenom_acheteur'."""
    rank = {"MATCH": 0, "REVIEW": 1, "MISMATCH": 2}
    agg: Dict[str, str] = {}
    for r in results:
        key = "nom_prenom_acheteur" if r.field in NAME_FIELDS else r.field
        status = r.status.value
        if status not in rank:
            continue
        if key not in agg or rank[status] > rank[agg[key]]:
            agg[key] = status
    return agg


def _same(a: Any, b: Any) -> bool:
    return a is not None and b is not None and str(a).strip().lower() == str(b).strip().lower()


def normalization_errors(extractions: List[DocumentExtraction], expected_fields: Dict[str, Any],
                         consolidated: Dict[str, Any]) -> Dict[str, Any]:
    """
    Erreur de normalisation = la lecture brute (raw_value) était correcte, mais la valeur finale retenue ne l'est pas.
    Taux de valeurs correctement normalisées = valeurs finales correctes / valeurs évaluées, sur les champs
    dont une valeur brute a été lue.
    """
    evaluated, correct, errors = 0, 0, []
    for name, expected in expected_fields.items():
        if expected is None:
            continue
        for ext in extractions:
            f = ext.fields.get(name)
            if f is None or f.value is None or f.raw_value in (None, ""):
                continue
            evaluated += 1
            if _same(f.value, expected):
                correct += 1
            else:
                renormalized, _ = _normalize(name, f.raw_value, f.raw_value)
                if _same(renormalized, expected):
                    errors.append({"field": name, "document": ext.filename, "raw_value": f.raw_value,
                                   "value": f.value, "expected": expected})
    return {"evaluated": evaluated, "correct": correct,
            "rate": round(correct / evaluated, 4) if evaluated else None, "errors": errors}


class DossierEvaluator:
    def __init__(self, ground_truth: Optional[GroundTruthManager] = None):
        self.ground_truth = ground_truth or GroundTruthManager()

    def evaluate(self, state: DossierState) -> Optional[Dict[str, Any]]:
        """Retourne les métriques du dossier, ou None si l'évaluation n'est pas possible/honnête."""
        if state.offline_mock_used:
            return None
        gt = self.ground_truth.get_ground_truth_for_dossier(state.dossier_id)
        if not gt:
            return None

        extractions = list(state.extractions.values())
        expected_fields = gt.get("expected_fields", {})

        after = consolidate_fields(extractions)
        result: Dict[str, Any] = {
            "evaluated_at": datetime.now().isoformat(timespec="seconds"),
            "extraction": Metrics.evaluate_extraction(after, expected_fields),
            "normalization": normalization_errors(extractions, expected_fields, after),
            "matching": Metrics.evaluate_matching(
                aggregate_matching(state.matching_results), gt.get("expected_cross_matching", {})
            ),
        }

        # Taux de contradictions : comparaisons en désaccord franc / comparaisons réalisées
        compared = len(state.matching_results)
        mismatches = sum(1 for m in state.matching_results if m.status.value == "MISMATCH")
        result["contradiction_rate"] = round(mismatches / compared, 4) if compared else None

        # Complétude et cohérences issues des scores interprétables du dossier
        if state.scores:
            result["completeness"] = state.scores.completeness_score
            result["identity_match"] = state.scores.identity_match
            result["chassis_match"] = state.scores.chassis_match

        # Classification documentaire (type réel annoté par pièce) et lisibilité annotée
        annotations = self.ground_truth.get_document_annotations(state.dossier_id)
        pairs = [{"file": e.filename, "expected": annotations[e.filename]["document_type"],
                  "predicted": e.document_type.value}
                 for e in extractions if e.filename in annotations and annotations[e.filename].get("document_type")]
        if pairs:
            result["classification"] = Metrics.evaluate_classification(
                [p["predicted"] for p in pairs], [p["expected"] for p in pairs])
            result["classification"]["pairs"] = pairs

        result["quality"] = [
            {"file": e.filename, "quality_score": q.quality_score, "is_readable": q.is_readable,
             "expected_readable": annotations.get(e.filename, {}).get("readable")}
            for path, e in state.extractions.items() if (q := state.quality_checks.get(path))
        ]

        if state.fallback_triggered:
            result["fallback"] = Metrics.evaluate_fallback_improvement(
                consolidate_fields(extractions, first_attempt=True), after, expected_fields
            )

        if state.decision:
            expected_decision = gt.get("expected_decision")
            result["decision"] = {
                "predicted": state.decision.decision.value,
                "expected": expected_decision,
                "correct": state.decision.decision.value == expected_decision,
            }

        from src.evaluation.error_analysis import ErrorAnalyzer
        result["error_analysis"] = ErrorAnalyzer.analyze(result)
        return result
