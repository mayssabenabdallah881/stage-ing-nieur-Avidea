"""
Détection des anomalies FRA-01 à FRA-09 (rapport, chapitre 8).

IMPORTANT : ces anomalies sont des SIGNAUX DE RISQUE à faire contrôler par un humain, jamais des preuves de fraude,
et elles ne produisent jamais un rejet à elles seules (au plus un REVIEW_REQUIRED).
Elles sont déduites uniquement de faits observés : résultats du matching, validation des valeurs, contradiction
entre tentatives d'extraction, type de pièce non identifié, doublons de fichiers, signes d'altération visibles
explicitement signalés par le VLM. Aucun signal n'est supposé ni inventé.
"""

from typing import Dict, List, Optional

from src.business.codes import ANOMALY_CODES
from src.validation.schemas import (
    Anomaly, DocumentExtraction, DocumentTypeEnum, ExtractionStatusEnum, MatchingResult, MatchingStatusEnum
)

# champ comparé par le matching -> code d'anomalie
MATCHING_FIELD_TO_CODE: Dict[str, str] = {
    "numero_chassis": "FRA-01",
    "nom": "FRA-02", "prenom": "FRA-02", "nom_prenom_acheteur": "FRA-02",
    "cin_passeport": "FRA-03",
    "cylindree": "FRA-04",
    "date_vente": "FRA-05", "date_mec": "FRA-05",
}
DATE_FIELDS = ("date_vente", "date_mec")


def _anomaly(code: str, severity: str, message: str, field: Optional[str] = None,
             documents: Optional[List[str]] = None) -> Anomaly:
    return Anomaly(code=code, name=ANOMALY_CODES[code], severity=severity, message=message,
                   field=field, documents=documents or [])


def detect_anomalies(
    extractions: List[DocumentExtraction],
    matching_results: List[MatchingResult],
    validation_issues: Optional[List[Dict[str, Optional[str]]]] = None,
    duplicate_groups: Optional[List[List[str]]] = None,
) -> List[Anomaly]:
    out: List[Anomaly] = []

    # FRA-01..05 : contradictions (MISMATCH) ou rapprochements ambigus (REVIEW) entre pièces
    for m in matching_results:
        code = MATCHING_FIELD_TO_CODE.get(m.field)
        if code is None or m.status not in (MatchingStatusEnum.MISMATCH, MatchingStatusEnum.REVIEW):
            continue
        severity = "HIGH" if m.status == MatchingStatusEnum.MISMATCH else "MEDIUM"
        out.append(_anomaly(code, severity, m.message or f"{m.field} : {m.value_a} / {m.value_b}",
                            m.field, [m.document_a, m.document_b]))

    # FRA-05 / FRA-09 : valeurs invalides ou implausibles relevées par l'étape de validation
    for issue in validation_issues or []:
        field = issue.get("field")
        if field is None:
            continue  # erreur de schéma Pydantic : gérée comme contrôle requis, pas comme anomalie de champ
        code = "FRA-05" if field in DATE_FIELDS else "FRA-09"
        out.append(_anomaly(code, "MEDIUM", str(issue.get("issue")), field, [str(issue.get("document"))]))

    for ext in extractions:
        # FRA-09 : champ contradictoire entre deux tentatives d'extraction
        for name, f in ext.fields.items():
            if f.status == "CONTRADICTORY":
                out.append(_anomaly("FRA-09", "HIGH",
                                    f"Valeur contradictoire entre deux lectures du même document ({name})",
                                    name, [ext.filename]))

        # FRA-06 : signes d'altération visibles signalés par le VLM (jamais déduits par le système)
        if ext.visual_flags:
            out.append(_anomaly("FRA-06", "HIGH",
                                "Signes d'altération visibles signalés par le modèle : " + " ; ".join(ext.visual_flags),
                                None, [ext.filename]))

        # FRA-08 : pièce lue mais non identifiée, ou type non confirmé par la seconde lecture
        read_ok = ext.extraction_status not in (ExtractionStatusEnum.FAILED, ExtractionStatusEnum.UNREADABLE)
        if read_ok and ext.document_type == DocumentTypeEnum.UNKNOWN:
            out.append(_anomaly("FRA-08", "MEDIUM", "Type de document non identifié", None, [ext.filename]))
        elif read_ok and ext.classification_conflict:
            out.append(_anomaly("FRA-08", "MEDIUM",
                                "Type de document non confirmé : la classification et l'extraction ne concordent pas",
                                None, [ext.filename]))

    # FRA-07 : fichiers strictement identiques (même contenu) dans le dossier
    for group in duplicate_groups or []:
        out.append(_anomaly("FRA-07", "MEDIUM", "Fichiers identiques fournis plusieurs fois : " + ", ".join(group),
                            None, list(group)))

    return out
