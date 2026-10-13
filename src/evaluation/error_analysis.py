"""
Analyse des erreurs (rapport, chapitre 9) : isole la cause première d'un écart avec la vérité terrain.

Catégories du rapport :
    erreur de classification | d'extraction | liée à la qualité de l'image | de normalisation | de matching |
    liée à une règle métier

Principe d'attribution (de l'amont vers l'aval, pour ne pas imputer à l'aval une erreur née en amont) :
  1. classification : type prédit différent du type réel annoté ;
  2. champ manquant ou faux : qualité de l'image si une pièce du dossier était illisible, sinon normalisation si la
     lecture brute était correcte, sinon extraction ;
  3. matching : statut différent de l'attendu ;
  4. règle métier : décision erronée alors que l'extraction et le matching sont corrects.
"""

from enum import Enum
from typing import Any, Dict, List

LOW_QUALITY_SCORE = 0.5


class ErrorCategory(str, Enum):
    CLASSIFICATION = "ERREUR_CLASSIFICATION"
    EXTRACTION = "ERREUR_EXTRACTION"
    IMAGE_QUALITY = "ERREUR_QUALITE_IMAGE"
    NORMALIZATION = "ERREUR_NORMALISATION"
    MATCHING = "ERREUR_MATCHING"
    BUSINESS_RULE = "ERREUR_REGLE_METIER"


class ErrorAnalyzer:
    """Diagnostic des erreurs à partir du dictionnaire d'évaluation d'un dossier (DossierEvaluator.evaluate)."""

    @classmethod
    def analyze(cls, evaluation: Dict[str, Any]) -> List[Dict[str, Any]]:
        errors: List[Dict[str, Any]] = []

        # 1. Classification
        for pair in evaluation.get("classification", {}).get("pairs", []):
            if pair["expected"] != pair["predicted"]:
                errors.append({
                    "category": ErrorCategory.CLASSIFICATION.value, "target": pair["file"],
                    "detail": f"type réel {pair['expected']}, type prédit {pair['predicted']}",
                })

        # 2. Extraction / qualité de l'image / normalisation
        low_quality = [q["file"] for q in evaluation.get("quality", [])
                       if not q.get("is_readable") or (q.get("quality_score") or 1.0) < LOW_QUALITY_SCORE]
        norm_errors = {e["field"]: e for e in evaluation.get("normalization", {}).get("errors", [])}
        extraction_failed_fields = set()
        for field, d in evaluation.get("extraction", {}).get("details", {}).items():
            if d["status"] == "EXACT_MATCH":
                continue
            extraction_failed_fields.add(field)
            detail = f"attendu {d['expected']!r}, obtenu {d['predicted']!r} ({d['status']})"
            if field in norm_errors and d["status"] == "MISMATCH":
                e = norm_errors[field]
                errors.append({"category": ErrorCategory.NORMALIZATION.value, "target": field,
                               "detail": f"lecture brute {e['raw_value']!r} correcte, valeur finale {e['value']!r} : {detail}"})
            elif low_quality:
                errors.append({"category": ErrorCategory.IMAGE_QUALITY.value, "target": field,
                               "detail": f"{detail} ; pièce(s) de faible qualité : {', '.join(low_quality)}"})
            else:
                errors.append({"category": ErrorCategory.EXTRACTION.value, "target": field, "detail": detail})

        # 3. Matching
        matching_wrong = False
        for field, d in evaluation.get("matching", {}).get("details", {}).items():
            if not d["correct"]:
                matching_wrong = True
                upstream = " (peut découler d'une erreur d'extraction sur ce champ)" if field in extraction_failed_fields else ""
                errors.append({"category": ErrorCategory.MATCHING.value, "target": field,
                               "detail": f"attendu {d['expected']}, obtenu {d['predicted']}{upstream}"})

        # 4. Règle métier : décision erronée alors que les entrées étaient correctes
        decision = evaluation.get("decision")
        if decision and decision.get("expected") is not None and not decision["correct"]:
            if not extraction_failed_fields and not matching_wrong:
                errors.append({"category": ErrorCategory.BUSINESS_RULE.value, "target": "decision",
                               "detail": f"attendu {decision['expected']}, obtenu {decision['predicted']} "
                                         f"avec extraction et matching corrects"})
            else:
                errors.append({"category": "DECISION_IMPACTEE", "target": "decision",
                               "detail": f"attendu {decision['expected']}, obtenu {decision['predicted']} : "
                                         f"conséquence d'une erreur en amont (extraction ou matching)"})
        return errors

    @staticmethod
    def summarize(errors: List[Dict[str, Any]]) -> Dict[str, int]:
        counts: Dict[str, int] = {}
        for e in errors:
            counts[e["category"]] = counts.get(e["category"], 0) + 1
        return counts
