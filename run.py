"""
Point d'entrée : traitement d'un dossier en ligne de commande ou lancement de l'interface Streamlit.

    python run.py                  # traite le premier dossier du manifeste
    python run.py DOS-003-LAAJMI   # traite un dossier précis
    python run.py batch [dossier]  # traite TOUTES les images d'un dossier en parallèle (défaut : data/input/Deux_Roues)
                                   # et les range par dossier client / type dans output/classified/
    python run.py api              # lance le backend API (FastAPI, http://127.0.0.1:8000/docs)
    python run.py app              # lance l'interface Streamlit (le backend doit tourner)
    python run.py evaluate         # traite les dossiers annotés (vrai appel Gemini) et calcule les métriques réelles
"""

import sys
import subprocess

from src.agent.orchestrator import SouscriptionAgentOrchestrator
from src.database.repository import SouscriptionRepository
from src.documents.manifest import list_dossiers, get_dossier, dossier_paths


def run_cli(dossier_id=None):
    dossier = get_dossier(dossier_id) if dossier_id else list_dossiers()[0]

    print(f"Agentic AI de souscription d'assurance motos | dossier {dossier['dossier_id']}")
    orchestrator = SouscriptionAgentOrchestrator(repository=SouscriptionRepository())
    state = orchestrator.process_dossier(dossier_paths(dossier), dossier_id=dossier["dossier_id"])

    if state.offline_mock_used:
        print("\n/!\\ MODE TEST HORS-LIGNE : extractions écrites à la main, NON produites par le modèle. "
              "Résultats non évaluables.")
    if state.persistence_error:
        print(f"\n/!\\ Résultat NON enregistré dans SQLite : {state.persistence_error}")

    print(f"\nStatut : {state.status}")
    if state.decision:
        print(f"Décision : {state.decision.decision.value}")
        print(f"Motif : {state.decision.reason}")

    print(f"\nTraces ({len(state.traces)} étapes) :")
    for t in state.traces:
        print(f"  [{t.timestamp}] {t.step.value:18} | {t.status.value:8} | {t.message} ({t.duration_ms}ms)")


def run_batch(folder=None):
    """Traite d'un coup toutes les images d'un dossier, puis les range par dossier client et par type de pièce."""
    from pathlib import Path
    from src.config import DEFAULT_BATCH_INPUT
    from src.documents.batch import BatchIntake

    input_dir = Path(folder) if folder else DEFAULT_BATCH_INPUT
    print(f"Agentic AI de souscription d'assurance motos - traitement en lot | dossier d'entrée : {input_dir}")

    def progress(item, done, total):
        label = item.document_type.value if item.status == "CLASSIFIED" else item.status
        print(f"  [{done:>2}/{total}] {item.filename:48} -> {label}")

    report = BatchIntake().process_folder(input_dir, progress=progress)

    for w in report.warnings:
        print(f"\n/!\\ {w}")
    print(f"\n{report.total_files} fichier(s), {report.total_items} pièce(s) analysée(s) en parallèle.")
    for dossier_id, files in report.dossiers.items():
        print(f"  {dossier_id}: {len(files)} pièce(s)")
    print(f"  {len(report.unassigned)} non affectée(s), {len(report.to_check)} à vérifier")
    print(f"\nRésultat rangé dans : {report.output_dir}")


def run_evaluate():
    """Benchmark sur la vérité terrain : métriques mesurées, jamais estimées (rapport, chapitre 14)."""
    from src.config import OUTPUT_DIR
    from src.evaluation.benchmark import run_benchmark

    orchestrator = SouscriptionAgentOrchestrator(repository=SouscriptionRepository())
    if not orchestrator.vlm_service.is_available():
        print("Aucune clé GEMINI_API_KEY : évaluation impossible (rien n'est inventé).")
        return
    report = run_benchmark(orchestrator, progress=lambda d_id, st: print(f"  {d_id:24} -> {st.status}"))

    print(f"\n{report['dossiers_evaluated']} dossier(s) évalué(s).")
    for d_id, why in report["skipped_dossiers"].items():
        print(f"  ignoré : {d_id} ({why})")
    cls, ext = report.get("classification"), report.get("extraction")
    if cls:
        print(f"Classification : accuracy {cls['accuracy']:.1%}, F1 macro {cls['macro_f1']:.1%} ({cls['total']} pièces)")
    if ext:
        print(f"Extraction     : exact match {ext['exact_match']:.1%} ({ext['fields_evaluated']} champs)")
    print(f"\nRapport : {OUTPUT_DIR / 'evaluation_report.json'}\nTableau LaTeX : {OUTPUT_DIR / 'evaluation_table.tex'}")


if __name__ == "__main__":
    arg = sys.argv[1] if len(sys.argv) > 1 else None
    if arg == "batch":
        run_batch(sys.argv[2] if len(sys.argv) > 2 else None)
    elif arg == "evaluate":
        run_evaluate()
    elif arg == "api":
        from src.config import API_HOST, API_PORT
        subprocess.run([sys.executable, "-m", "uvicorn", "src.api.main:app", "--host", API_HOST, "--port", str(API_PORT)])
    elif arg in ("app", "streamlit"):
        subprocess.run([sys.executable, "-m", "streamlit", "run", "frontend/streamlit_app.py", "--server.port=3000"])
    else:
        run_cli(arg)
