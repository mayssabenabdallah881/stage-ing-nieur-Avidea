"""
Stratégie agentique de Fallback contrôlé.
Règles strictes (rapport, chapitre 6) :
- Tentative 1 : Extraction standard.
- Si un champ critique (chassis, cylindree, cin_passeport) est null ou incertain -> Quality Check, prétraitement, Tentative 2 ciblée.
- Si Tentative 1 est null et Tentative 2 est ABC123456 -> Conserver la valeur extraite en Tentative 2.
- Si Tentative 1 = ABC123456 et Tentative 2 = XYZ987654 (contradiction) -> Déclencher REVIEW_REQUIRED.
- Ne jamais inventer de données ni forcer un choix arbitraire.
"""

from typing import Dict, Any, Optional, Tuple
from src.validation.schemas import ExtractedField, ExtractionStatusEnum
from src.preprocessing.image_processor import preprocess_document_image


class FallbackAgent:
    """Gestionnaire déterministe des re-tentatives après prétraitement ciblé."""

    CRITICAL_FIELDS = ["numero_chassis", "cylindree", "cin_passeport", "nom_prenom_acheteur"]

    @classmethod
    def should_trigger_fallback(cls, fields: Dict[str, ExtractedField], doc_type: str) -> Tuple[bool, Optional[str]]:
        """
        Détermine si un fallback est justifié pour ce type de document.
        Ne déclenche pas de fallback si le champ n'est pas censé figurer sur ce document.
        """
        def missing(name: str) -> bool:
            f = fields.get(name)
            return f is None or f.value is None

        if doc_type in ("CIN", "PASSPORT"):
            if missing("cin_passeport"):
                return True, "Numéro de CIN / passeport absent ou illisible sur la pièce d'identité."
            if missing("nom_prenom_acheteur"):
                return True, "Nom/prénom illisible sur la pièce d'identité."

        elif doc_type in ("CONTRAT", "FACTURE"):
            if missing("numero_chassis"):
                return True, "Numéro de châssis / cadre non extrait sur le contrat ou la facture."
            if missing("cylindree"):
                return True, "Cylindrée motocycle non lisible sur le contrat/facture."

        elif doc_type == "CARTE_GRISE":
            if missing("numero_chassis"):
                return True, "Numéro de châssis non extrait sur la carte grise."

        elif doc_type == "QUITTANCE":
            if missing("numero_chassis"):
                return True, "Numéro d'immatriculation / châssis manquant sur la quittance fiscale."

        return False, None

    @classmethod
    def reconcile_attempts(
        cls,
        attempt_1_fields: Dict[str, ExtractedField],
        attempt_2_fields: Dict[str, ExtractedField]
    ) -> Tuple[Dict[str, ExtractedField], bool, Optional[str]]:
        """
        Compare Tentative 1 et Tentative 2 pour chaque champ.
        Retourne (champs_réconciliés, contradiction_détectée, message_explicatif).
        """
        reconciled = dict(attempt_1_fields)
        contradiction_detected = False
        reasons = []

        all_keys = set(attempt_1_fields.keys()).union(attempt_2_fields.keys())

        for key in all_keys:
            f1 = attempt_1_fields.get(key)
            f2 = attempt_2_fields.get(key)

            v1 = f1.value if f1 else None
            v2 = f2.value if f2 else None

            # Cas 1: Tentative 1 nulle, Tentative 2 a trouvé une valeur
            if (v1 is None or v1 == "") and (v2 is not None and v2 != ""):
                reconciled[key] = f2
                reconciled[key].status = "RECOVERED_BY_FALLBACK"

            # Cas 2: Les deux ont trouvé la même valeur
            elif v1 == v2:
                if f1:
                    reconciled[key] = f1

            # Cas 3: Les deux ont trouvé des valeurs différentes non nulles -> Contradiction!
            elif v1 is not None and v2 is not None and str(v1).strip() != str(v2).strip():
                contradiction_detected = True
                msg = f"Contradiction champ '{key}': Tentative 1='{v1}' vs Tentative 2='{v2}'"
                reasons.append(msg)
                if key in reconciled:
                    reconciled[key].status = "CONTRADICTORY"

        reason_str = " ; ".join(reasons) if contradiction_detected else None
        return reconciled, contradiction_detected, reason_str
