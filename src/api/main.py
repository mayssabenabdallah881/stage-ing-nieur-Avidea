"""
Backend API (FastAPI) — couche entre l'interface Streamlit et l'orchestrateur agentique (rapport, chapitre 4).

    Interface Streamlit  ->  API Backend  ->  Agentic Orchestrator  ->  ...  ->  Pydantic + SQLite

Lancement :  python run.py api      (ou : uvicorn src.api.main:app --port 8000)
Documentation interactive : http://127.0.0.1:8000/docs
"""

import re
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, File, HTTPException, UploadFile

from src.agent.orchestrator import SouscriptionAgentOrchestrator
from src.agent.state import DossierState
from src.config import GEMINI_MODEL, INPUT_DIR
from src.database.repository import SouscriptionRepository
from src.documents.loader import DocumentLoader
from src.documents.manifest import dossier_paths, get_dossier, list_dossiers
from src.evaluation.ground_truth import GroundTruthManager
from src.vision.gemini_client import GeminiVLMClient

app = FastAPI(title="Agentic AI de souscription d'assurance motos — API", version="1.0")

_repo: Optional[SouscriptionRepository] = None
_orchestrator: Optional[SouscriptionAgentOrchestrator] = None


def get_repository() -> SouscriptionRepository:
    global _repo
    if _repo is None:
        _repo = SouscriptionRepository()
    return _repo


def get_orchestrator() -> SouscriptionAgentOrchestrator:
    global _orchestrator
    if _orchestrator is None:
        _orchestrator = SouscriptionAgentOrchestrator(repository=get_repository())
    return _orchestrator


def _safe_name(name: str) -> str:
    """Nom de fichier téléversé nettoyé : jamais de chemin, jamais de caractères dangereux."""
    base = Path(name or "document").name
    return re.sub(r"[^\w.\-() ]", "_", base) or "document"


def summarize(state: DossierState) -> Dict[str, Any]:
    """Résumé JSON d'un dossier qui vient d'être traité."""
    d = state.decision
    return {
        "dossier_id": state.dossier_id,
        "status": state.status,
        "decision": d.decision.value if d else None,
        "reason": d.reason if d else None,
        "rejection_codes": d.rejection_codes if d else [],
        "rejection_type": d.rejection_type if d else None,
        "explanation": d.explanation if d else [],
        "anomalies": [a.model_dump() for a in state.anomalies],
        "scores": state.scores.model_dump() if state.scores else None,
        "reconstruction": state.reconstruction.model_dump() if state.reconstruction else None,
        "review_reasons": state.review_reasons,
        "unreadable_documents": state.unreadable_documents,
        "fallback_triggered": state.fallback_triggered,
        "offline_mock_used": state.offline_mock_used,
        "persistence_error": state.persistence_error,
        "documents": [e.to_report_dict(d.decision.value if d else None) | {"filename": e.filename}
                      for e in state.extractions.values()],
    }


@app.get("/health")
def health() -> Dict[str, Any]:
    vlm = GeminiVLMClient()
    return {"status": "ok", "model": GEMINI_MODEL, "vlm_available": bool(vlm.client or vlm.api_key),
            "offline_mock": GeminiVLMClient.offline_mock_enabled()}


@app.get("/stats")
def stats() -> Dict[str, Any]:
    out = get_repository().get_dashboard_stats()
    out["annotated_dossiers"] = len(GroundTruthManager().list_annotated_dossiers())
    return out


@app.get("/samples")
def samples() -> List[Dict[str, Any]]:
    """Dossiers d'exemple du manifeste (data/inputs_manifest.json)."""
    return [{"dossier_id": d["dossier_id"], "client": d["client"], "documents": len(d["documents"])}
            for d in list_dossiers()]


@app.post("/dossiers")
async def create_dossier(files: List[UploadFile] = File(...)) -> Dict[str, Any]:
    """Reçoit plusieurs documents, lance l'analyse agentique complète et retourne la décision."""
    if not files:
        raise HTTPException(status_code=400, detail="Aucun fichier reçu.")
    d_id = f"DOS-IMPORT-{datetime.now().strftime('%Y%m%d-%H%M%S-%f')[:-3]}"
    upload_dir = INPUT_DIR / d_id
    upload_dir.mkdir(parents=True, exist_ok=True)

    paths: List[str] = []
    for i, uf in enumerate(files):
        name = _safe_name(uf.filename)
        if Path(name).suffix.lower() not in DocumentLoader.SUPPORTED_EXTENSIONS:
            continue                                   # formats non supportés ignorés (le dossier peut rester vide)
        target = upload_dir / name
        if target.exists():
            target = upload_dir / f"{target.stem}_{i}{target.suffix}"
        with open(target, "wb") as out:
            shutil.copyfileobj(uf.file, out)
        paths.append(str(target))

    state = get_orchestrator().process_dossier(paths, dossier_id=d_id)
    return summarize(state)


@app.post("/dossiers/samples/{dossier_id}")
def run_sample(dossier_id: str) -> Dict[str, Any]:
    try:
        dossier = get_dossier(dossier_id)
    except KeyError:
        raise HTTPException(status_code=404, detail=f"Dossier d'exemple inconnu : {dossier_id}")
    state = get_orchestrator().process_dossier(dossier_paths(dossier), dossier_id=dossier_id)
    return summarize(state)


@app.get("/dossiers")
def list_all() -> List[Dict[str, Any]]:
    return get_repository().list_dossiers()


@app.get("/dossiers/{dossier_id}")
def get_one(dossier_id: str) -> Dict[str, Any]:
    details = get_repository().get_dossier_details(dossier_id)
    if details is None:
        raise HTTPException(status_code=404, detail=f"Dossier introuvable : {dossier_id}")
    return details


@app.get("/dossiers/{dossier_id}/traces")
def get_traces(dossier_id: str) -> List[Dict[str, Any]]:
    details = get_repository().get_dossier_details(dossier_id)
    if details is None:
        raise HTTPException(status_code=404, detail=f"Dossier introuvable : {dossier_id}")
    return details["traces"]
