"""
Benchmark sur le corpus annoté (rapport, chapitre 14 : « Résultats Data Science »).

Exécute le pipeline sur chaque dossier possédant une vérité terrain, puis agrège les métriques réellement mesurées.
Principes :
  - aucune valeur n'est inventée : un dossier traité en mode hors-ligne de test, ou sans appel réel au VLM,
    n'est jamais évalué (il est listé comme « ignoré ») ;
  - les résultats portent sur un corpus très limité : ils décrivent ce corpus, pas l'ensemble des dossiers d'assurance ;
  - le tableau LaTeX produit reprend les lignes du tableau « Indicateurs d'évaluation » du rapport.
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.config import OUTPUT_DIR
from src.documents.manifest import dossier_paths, get_dossier
from src.evaluation.error_analysis import ErrorAnalyzer
from src.evaluation.ground_truth import GroundTruthManager
from src.evaluation.metrics import DocumentAIMetricsCalculator as Metrics


def _mean(values: List[Optional[float]]) -> Optional[float]:
    vals = [v for v in values if v is not None]
    return round(sum(vals) / len(vals), 4) if vals else None


def aggregate_evaluations(evaluations: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    """Agrège les évaluations par dossier (micro-moyennes pour l'extraction, effectifs cumulés pour la classification)."""
    report: Dict[str, Any] = {"dossiers_evaluated": len(evaluations)}

    pairs = [p for ev in evaluations.values() for p in ev.get("classification", {}).get("pairs", [])]
    if pairs:
        report["classification"] = Metrics.evaluate_classification(
            [p["predicted"] for p in pairs], [p["expected"] for p in pairs])
        report["classification"].pop("pairs", None)

    ext = [ev["extraction"] for ev in evaluations.values() if "total_fields" in ev.get("extraction", {})]
    if ext:
        total = sum(e["total_fields"] for e in ext)
        tp = sum(e["exact_matches"] for e in ext)
        missing = round(sum(e["missing_rate"] * e["total_fields"] for e in ext))
        wrong = round(sum(e["error_rate"] * e["total_fields"] for e in ext))
        prec = tp / (tp + wrong) if (tp + wrong) else 0.0
        rec = tp / (tp + missing) if (tp + missing) else 0.0
        report["extraction"] = {
            "fields_evaluated": total, "exact_matches": tp,
            "exact_match": round(tp / total, 4), "missing_rate": round(missing / total, 4),
            "error_rate": round(wrong / total, 4), "precision": round(prec, 4), "recall": round(rec, 4),
            "f1_score": round(2 * prec * rec / (prec + rec), 4) if (prec + rec) else 0.0,
        }

    norm = [ev["normalization"] for ev in evaluations.values() if ev.get("normalization", {}).get("evaluated")]
    if norm:
        evaluated = sum(n["evaluated"] for n in norm)
        report["normalization_rate"] = round(sum(n["correct"] for n in norm) / evaluated, 4)

    report["completeness"] = _mean([ev.get("completeness") for ev in evaluations.values()])
    report["chassis_coherence"] = _mean([ev.get("chassis_match") for ev in evaluations.values()])
    report["identity_coherence"] = _mean([ev.get("identity_match") for ev in evaluations.values()])
    report["contradiction_rate"] = _mean([ev.get("contradiction_rate") for ev in evaluations.values()])

    decisions = [ev["decision"] for ev in evaluations.values() if ev.get("decision", {}).get("expected")]
    if decisions:
        report["decision_accuracy"] = round(sum(d["correct"] for d in decisions) / len(decisions), 4)

    all_errors = [e for ev in evaluations.values() for e in ev.get("error_analysis", [])]
    report["error_summary"] = ErrorAnalyzer.summarize(all_errors)
    return report


def _pct(v: Optional[float]) -> str:
    return "non mesuré" if v is None else f"{v * 100:.1f}\\,\\%".replace(".", ",")


def render_latex_table(report: Dict[str, Any]) -> str:
    """Lignes du tableau « Indicateurs d'évaluation » du rapport, remplies avec les valeurs réellement mesurées."""
    cls = report.get("classification", {})
    ext = report.get("extraction", {})
    n_docs, n_fields = cls.get("total"), ext.get("fields_evaluated")
    n_classes = len([c for c in cls.get("per_class", {}).values() if c["support"] > 0])
    rows = [
        ("Accuracy classification", _pct(cls.get("accuracy")), f"{n_docs} pièces annotées" if n_docs else "Sur corpus annoté"),
        ("Precision", _pct(cls.get("macro_precision")), f"Macro-moyenne ({n_classes} classes)"),
        ("Recall", _pct(cls.get("macro_recall")), f"Macro-moyenne ({n_classes} classes)"),
        ("F1-score", _pct(cls.get("macro_f1")), f"Macro-moyenne ({n_classes} classes)"),
        ("Exact Match extraction", _pct(ext.get("exact_match")), f"{n_fields} champs évalués" if n_fields else "Champs évalués"),
        ("Complétude", _pct(report.get("completeness")), "Champs renseignés (moyenne par dossier)"),
        ("Cohérence châssis", _pct(report.get("chassis_coherence")), "Matching inter-documents"),
        ("Cohérence identité", _pct(report.get("identity_coherence")), "Matching inter-documents"),
    ]
    body = "\n".join(f"{name} & {value} & {comment}\\\\" for name, value, comment in rows)
    return (
        "% Généré par `python run.py evaluate` — valeurs mesurées sur le corpus annoté, jamais estimées.\n"
        f"% {report.get('dossiers_evaluated', 0)} dossier(s) évalué(s).\n"
        "\\begin{tabularx}{0.9\\textwidth}{>{\\bfseries}l c X}\n\\toprule\n"
        "Indicateur & Valeur & Commentaire\\\\\n\\midrule\n" + body + "\n\\bottomrule\n\\end{tabularx}\n"
    )


def run_benchmark(orchestrator, ground_truth: Optional[GroundTruthManager] = None,
                  output_dir: Optional[Path] = None, progress=None) -> Dict[str, Any]:
    """Traite tous les dossiers annotés qui figurent au manifeste et écrit evaluation_report.json / evaluation_table.tex."""
    gt = ground_truth or GroundTruthManager()
    evaluations: Dict[str, Dict[str, Any]] = {}
    skipped: Dict[str, str] = {}

    for annotated in gt.list_annotated_dossiers():
        d_id = annotated["dossier_id"]
        try:
            paths = dossier_paths(get_dossier(d_id))
        except KeyError:
            skipped[d_id] = "dossier absent du manifeste data/inputs_manifest.json"
            continue
        state = orchestrator.process_dossier(paths, dossier_id=d_id)
        if progress:
            progress(d_id, state)
        if state.evaluation is None:
            skipped[d_id] = ("mode hors-ligne de test : valeurs non issues du modèle, évaluation refusée"
                             if state.offline_mock_used else "évaluation impossible (extraction non réalisée)")
            continue
        evaluations[d_id] = state.evaluation

    report = aggregate_evaluations(evaluations)
    report.update({
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "skipped_dossiers": skipped,
        "per_dossier": evaluations,
        "limitation": "Corpus restreint : ces résultats ne se généralisent pas à l'ensemble des dossiers d'assurance.",
    })

    out = Path(output_dir) if output_dir else OUTPUT_DIR
    out.mkdir(parents=True, exist_ok=True)
    (out / "evaluation_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    (out / "evaluation_table.tex").write_text(render_latex_table(report), encoding="utf-8")
    return report
