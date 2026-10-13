"""
Calcul des métriques Data Science réelles pour l'évaluation du système Document AI.
Aucune métrique inventée : tous les scores découlent strictement de la comparaison
Prédictions du VLM vs Vérité Terrain annotée.
"""

from typing import Dict, List, Any, Optional


class DocumentAIMetricsCalculator:
    """Calculateur de métriques de précision pour l'extraction et le matching."""

    @classmethod
    def evaluate_extraction(
        cls,
        predicted_fields: Dict[str, Any],
        ground_truth_fields: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Compare les champs prédits aux champs de référence :
        - Exact Match (EM) : Valeur exacte après normalisation
        - Missing Rate : Champs présents dans GT mais prédits null par le VLM
        - Error Rate : Champs prédits avec une valeur différente de GT
        """
        total_fields = len(ground_truth_fields)
        if total_fields == 0:
            return {"error": "Aucun champ de référence disponible."}

        exact_matches = 0
        missing_count = 0
        mismatch_count = 0

        field_comparisons = {}

        for f_name, expected_val in ground_truth_fields.items():
            pred_val = predicted_fields.get(f_name)

            # Nettoyage pour comparaison équitable
            exp_str = str(expected_val).strip().lower() if expected_val is not None else None
            pred_str = str(pred_val).strip().lower() if pred_val is not None else None

            if exp_str is not None and pred_str is None:
                missing_count += 1
                status = "MISSING"
            elif exp_str == pred_str:
                exact_matches += 1
                status = "EXACT_MATCH"
            else:
                mismatch_count += 1
                status = "MISMATCH"

            field_comparisons[f_name] = {
                "expected": expected_val,
                "predicted": pred_val,
                "status": status
            }

        accuracy = exact_matches / total_fields
        missing_rate = missing_count / total_fields
        error_rate = mismatch_count / total_fields

        # Precision & Recall orientés détection d'entités
        tp = exact_matches
        fp = mismatch_count
        fn = missing_count

        precision = (tp / (tp + fp)) if (tp + fp) > 0 else 0.0
        recall = (tp / (tp + fn)) if (tp + fn) > 0 else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

        return {
            "total_fields": total_fields,
            "exact_matches": exact_matches,
            "exact_match_ratio": round(accuracy, 4),
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1_score": round(f1, 4),
            "missing_rate": round(missing_rate, 4),
            "error_rate": round(error_rate, 4),
            "details": field_comparisons
        }

    @classmethod
    def evaluate_classification(cls, predicted: List[str], expected: List[str]) -> Dict[str, Any]:
        """
        Métriques de classification documentaire (rapport, chapitre 9) :
        Accuracy = (TP+TN)/(TP+TN+FP+FN) ; ici, accuracy globale = pièces correctement classées / pièces évaluées ;
        Precision = TP/(TP+FP), Recall = TP/(TP+FN), F1 = 2*P*R/(P+R) par classe, puis macro-moyenne ;
        matrice de confusion {attendu: {prédit: effectif}} pour repérer les catégories confondues.
        """
        if len(predicted) != len(expected):
            raise ValueError("Les listes prédite et attendue doivent avoir la même longueur.")
        if not expected:
            return {"error": "Aucune pièce annotée à évaluer."}

        labels = sorted(set(expected) | set(predicted))
        confusion: Dict[str, Dict[str, int]] = {e: {} for e in sorted(set(expected))}
        for e, p_ in zip(expected, predicted):
            confusion[e][p_] = confusion[e].get(p_, 0) + 1

        per_class: Dict[str, Dict[str, Any]] = {}
        for label in labels:
            tp = sum(1 for e, p_ in zip(expected, predicted) if e == label and p_ == label)
            fp = sum(1 for e, p_ in zip(expected, predicted) if e != label and p_ == label)
            fn = sum(1 for e, p_ in zip(expected, predicted) if e == label and p_ != label)
            prec = tp / (tp + fp) if (tp + fp) else 0.0
            rec = tp / (tp + fn) if (tp + fn) else 0.0
            f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
            per_class[label] = {"precision": round(prec, 4), "recall": round(rec, 4), "f1_score": round(f1, 4),
                                "support": sum(1 for e in expected if e == label), "tp": tp, "fp": fp, "fn": fn}

        scored = [v for v in per_class.values() if v["support"] > 0]   # classes réellement présentes dans la référence
        correct = sum(1 for e, p_ in zip(expected, predicted) if e == p_)
        macro = lambda key: round(sum(v[key] for v in scored) / len(scored), 4) if scored else 0.0
        return {
            "total": len(expected),
            "correct": correct,
            "accuracy": round(correct / len(expected), 4),
            "macro_precision": macro("precision"),
            "macro_recall": macro("recall"),
            "macro_f1": macro("f1_score"),
            "per_class": per_class,
            "confusion_matrix": confusion,
        }

    @classmethod
    def evaluate_fallback_recovery(cls, total_fallback_attempts: int, recovered_fields_count: int) -> Dict[str, Any]:
        """
        Calcule l'efficacité du mécanisme de fallback ciblé :
        Fallback Recovery Rate = (Champs récupérés grâce au fallback / Tentatives de fallback)
        """
        if total_fallback_attempts == 0:
            return {
                "fallback_attempts": 0,
                "recovered_fields": 0,
                "recovery_rate": 0.0,
                "note": "Aucun fallback déclenché sur ce dossier."
            }

        rate = recovered_fields_count / total_fallback_attempts
        return {
            "fallback_attempts": total_fallback_attempts,
            "recovered_fields": recovered_fields_count,
            "recovery_rate": round(rate, 4)
        }

    @classmethod
    def evaluate_matching(cls, predicted: Dict[str, str], expected: Dict[str, str]) -> Dict[str, Any]:
        """
        Compare les statuts de matching prédits (par champ) aux statuts attendus.
        Un champ attendu mais non comparé par le système compte comme une erreur (statut 'NOT_COMPARED').
        """
        if not expected:
            return {"error": "Aucun statut de matching de référence disponible."}

        comparisons = {}
        correct = 0
        for field, exp in expected.items():
            pred = predicted.get(field, "NOT_COMPARED")
            ok = pred == exp
            correct += int(ok)
            comparisons[field] = {"expected": exp, "predicted": pred, "correct": ok}

        labels = sorted(set(expected.values()) | {v for v in predicted.values()})
        per_label = {}
        f1_values = []
        for label in labels:
            tp = sum(1 for f, e in expected.items() if e == label and predicted.get(f) == label)
            fp = sum(1 for f, p_ in predicted.items() if p_ == label and expected.get(f) != label and f in expected)
            fn = sum(1 for f, e in expected.items() if e == label and predicted.get(f) != label)
            support = sum(1 for e in expected.values() if e == label)
            prec = tp / (tp + fp) if (tp + fp) else 0.0
            rec = tp / (tp + fn) if (tp + fn) else 0.0
            f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
            per_label[label] = {"precision": round(prec, 4), "recall": round(rec, 4),
                                "f1_score": round(f1, 4), "support": support}
            if support > 0:
                f1_values.append(f1)

        return {
            "total_fields": len(expected),
            "correct": correct,
            "accuracy": round(correct / len(expected), 4),
            "macro_f1": round(sum(f1_values) / len(f1_values), 4) if f1_values else 0.0,
            "per_label": per_label,
            "details": comparisons,
        }

    @classmethod
    def evaluate_fallback_improvement(
        cls,
        before_fields: Dict[str, Any],
        after_fields: Dict[str, Any],
        ground_truth_fields: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Mesure l'apport réel du fallback : mêmes champs de référence évalués avant (tentative 1) et après fallback.
        Recovery rate = champs absents après la tentative 1 puis retrouvés avec la BONNE valeur / champs absents après la tentative 1.
        """
        before = cls.evaluate_extraction(before_fields, ground_truth_fields)
        after = cls.evaluate_extraction(after_fields, ground_truth_fields)
        if "error" in before:
            return before

        missing_before = [f for f, d in before["details"].items() if d["status"] == "MISSING"]
        recovered = [f for f in missing_before if after["details"][f]["status"] == "EXACT_MATCH"]

        return {
            "exact_match_before": before["exact_match_ratio"],
            "exact_match_after": after["exact_match_ratio"],
            "improvement": round(after["exact_match_ratio"] - before["exact_match_ratio"], 4),
            "missing_before": len(missing_before),
            "recovered_correctly": len(recovered),
            "recovery_rate": round(len(recovered) / len(missing_before), 4) if missing_before else None,
        }
