"""
Chargeur des annotations de référence (Ground Truth) — rapport, chapitre 9.

Structure d'un dossier annoté (data/ground_truth/dossiers_ground_truth.json) :
    dossier_id, client_name
    expected_decision        décision attendue (ACCEPTED / REJECTED / REVIEW_REQUIRED / DOCUMENT_MANQUANT), lorsqu'elle est définie
    expected_fields          valeurs correctes des champs du dossier (cin_passeport, numero_chassis, cylindree...)
    expected_cross_matching  statut attendu du matching par champ (MATCH / REVIEW / MISMATCH)
    documents                [{ "file": ..., "document_type": ..., "readable": true|false (optionnel) }]
                             type réel de chaque pièce (et lisibilité réelle si annotée)
La vérité terrain doit être vérifiée à la main : elle n'est jamais produite par le modèle.
"""

import json
from pathlib import Path
from typing import Dict, Any, List, Optional
from src.config import GROUND_TRUTH_DIR


class GroundTruthManager:
    """Gestionnaire des données de référence (Vérité Terrain)."""

    def __init__(self, gt_path: Optional[Path] = None):
        self.gt_path = gt_path or (GROUND_TRUTH_DIR / "dossiers_ground_truth.json")
        self._data: Dict[str, Any] = {}
        self.load()

    def load(self):
        """Charge le fichier JSON de vérité terrain."""
        if self.gt_path.exists():
            with open(self.gt_path, "r", encoding="utf-8") as f:
                self._data = json.load(f)
        else:
            self._data = {"dossiers": []}

    def get_ground_truth_for_dossier(self, dossier_id: str) -> Optional[Dict[str, Any]]:
        """Recherche la vérité terrain par identifiant de dossier."""
        for d in self._data.get("dossiers", []):
            if d.get("dossier_id") == dossier_id:
                return d
        return None

    def get_document_annotations(self, dossier_id: str) -> Dict[str, Dict[str, Any]]:
        """Annotations par pièce : {nom_de_fichier: {document_type, readable?}} (vide si non annoté)."""
        gt = self.get_ground_truth_for_dossier(dossier_id) or {}
        return {d["file"]: d for d in gt.get("documents", []) if d.get("file")}

    def list_annotated_dossiers(self) -> List[Dict[str, Any]]:
        """Retourne l'ensemble des dossiers annotés pour le benchmark."""
        return self._data.get("dossiers", [])
