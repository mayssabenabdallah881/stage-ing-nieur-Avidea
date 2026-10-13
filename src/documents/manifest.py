"""
Lecture de la liste des documents d'entrée (data/inputs_manifest.json).
Associe chaque dossier de souscription à ses fichiers (images ou PDF).
"""

import json
from typing import Any, Dict, List

from src.config import MANIFEST_PATH, SAMPLES_DIR


def load_manifest() -> Dict[str, Any]:
    with open(MANIFEST_PATH, encoding="utf-8") as f:
        return json.load(f)


def list_dossiers() -> List[Dict[str, Any]]:
    return load_manifest()["dossiers"]


def get_dossier(dossier_id: str) -> Dict[str, Any]:
    for d in list_dossiers():
        if d["dossier_id"] == dossier_id:
            return d
    raise KeyError(f"Dossier inconnu : {dossier_id}")


def dossier_paths(dossier: Dict[str, Any]) -> List[str]:
    return [str(SAMPLES_DIR / doc["file"]) for doc in dossier["documents"]]
