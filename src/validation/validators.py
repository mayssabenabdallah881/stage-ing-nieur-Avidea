"""
Validation des extractions (étape VALIDATION) et contrôle de qualité des données du dossier (étape DATA_QUALITY).

Principe : on SIGNALE une valeur douteuse (statut AMBIGUOUS), on ne la corrige jamais et on ne la remplace jamais.
Les compteurs produits sont des comptages réels, pas des scores.
"""

from datetime import datetime
from typing import Any, Dict, List, Optional

from pydantic import ValidationError

from src.config import CIN_LENGTH, DISPLACEMENT_PLAUSIBLE_RANGE, PASSPORT_LENGTH_RANGE
from src.validation.schemas import (
    DocumentExtraction, DocumentQuality, DocumentTypeEnum, ExtractionStatusEnum, MatchingResult, MatchingStatusEnum
)

ALL_FIELDS = ["nom", "prenom", "cin_passeport", "nom_prenom_acheteur",
              "cylindree", "date_vente", "date_mec", "numero_chassis"]
CRITICAL_FIELDS = ["cin_passeport", "nom_prenom_acheteur", "numero_chassis", "cylindree"]
NAME_FIELDS = ("nom", "prenom", "nom_prenom_acheteur")
LOW_MODEL_CONFIDENCE = 0.5  # information du modèle, non calibrée


def _parse_date(text: str) -> Optional[datetime]:
    try:
        d, m, y = (int(x) for x in str(text).split("/"))
        return datetime(y, m, d)
    except (ValueError, TypeError):
        return None


def validate_extraction(extraction: DocumentExtraction) -> List[Dict[str, Optional[str]]]:
    """
    Contrôle le schéma Pydantic puis la plausibilité des valeurs.
    Les champs douteux passent au statut AMBIGUOUS ; la valeur est conservée telle quelle.
    Retourne la liste des anomalies : [{"document", "field", "issue"}].
    """
    issues: List[Dict[str, Optional[str]]] = []

    def flag(field: Optional[str], issue: str) -> None:
        issues.append({"document": extraction.filename, "field": field, "issue": issue})
        if field and field in extraction.fields:
            extraction.fields[field].status = "AMBIGUOUS"

    try:
        DocumentExtraction.model_validate(extraction.model_dump())
    except ValidationError as e:
        flag(None, f"Schéma Pydantic invalide : {e.errors()[0].get('msg', 'erreur')}")
        return issues

    if extraction.extraction_status in (ExtractionStatusEnum.FAILED, ExtractionStatusEnum.UNREADABLE):
        return issues  # rien à valider : tous les champs sont à null

    f = extraction.fields

    ident = f.get("cin_passeport")
    if ident and ident.value is not None:
        v = str(ident.value)
        is_cin = v.isdigit() and len(v) == CIN_LENGTH
        is_passport = v.isalnum() and any(c.isalpha() for c in v) \
            and PASSPORT_LENGTH_RANGE[0] <= len(v) <= PASSPORT_LENGTH_RANGE[1]
        if extraction.document_type == DocumentTypeEnum.CIN and not is_cin:
            flag("cin_passeport", f"CIN invalide ({v}) : {CIN_LENGTH} chiffres attendus")
        elif extraction.document_type == DocumentTypeEnum.PASSPORT and not is_passport:
            flag("cin_passeport", f"Numéro de passeport invalide ({v}) : "
                                  f"{PASSPORT_LENGTH_RANGE[0]} à {PASSPORT_LENGTH_RANGE[1]} caractères alphanumériques attendus")
        elif not (is_cin or is_passport):
            flag("cin_passeport", f"Numéro d'identité invalide ({v}) : CIN à {CIN_LENGTH} chiffres ou passeport attendu")

    cyl = f.get("cylindree")
    if cyl and cyl.value is not None:
        low, high = DISPLACEMENT_PLAUSIBLE_RANGE
        try:
            if not (low <= int(cyl.value) <= high):
                flag("cylindree", f"Cylindrée {cyl.value} cc hors plage plausible ({low}-{high})")
        except (ValueError, TypeError):
            flag("cylindree", f"Cylindrée non numérique ({cyl.value})")

    chassis = f.get("numero_chassis")
    if chassis and chassis.value is not None and len(str(chassis.value)) < 5:
        flag("numero_chassis", f"Numéro de châssis trop court ({chassis.value})")

    for name in ("date_vente", "date_mec"):
        fld = f.get(name)
        if fld and fld.value is not None:
            dt = _parse_date(fld.value)
            if dt is None:
                flag(name, f"Date invalide ({fld.value}) : JJ/MM/AAAA attendu")
            elif dt > datetime.now():
                flag(name, f"Date dans le futur ({fld.value})")

    for name in NAME_FIELDS:
        fld = f.get(name)
        if fld and fld.value is not None and not any(ch.isalpha() for ch in str(fld.value)):
            flag(name, f"Nom sans lettres ({fld.value})")

    return issues


def assess_data_quality(
    extractions: List[DocumentExtraction],
    matching_results: List[MatchingResult],
    quality_checks: Optional[Dict[str, DocumentQuality]] = None,
) -> Dict[str, Any]:
    """
    Bilan de la qualité des données du dossier, uniquement à partir de comptages :
    champs présents / manquants, valeurs ambiguës, résultats de matching, documents en échec.
    """
    present: Dict[str, str] = {}
    for name in ALL_FIELDS:
        for ext in extractions:
            fld = ext.fields.get(name)
            if fld and fld.value is not None and str(fld.value) != "":
                present[name] = ext.filename
                break

    missing = [n for n in ALL_FIELDS if n not in present]
    critical_missing = [n for n in CRITICAL_FIELDS if n not in present]

    ambiguous = sorted({
        f"{ext.filename}:{n}"
        for ext in extractions for n, fld in ext.fields.items()
        if fld.status in ("AMBIGUOUS", "CONTRADICTORY")
    })
    low_conf = sorted({
        f"{ext.filename}:{n}"
        for ext in extractions for n, fld in ext.fields.items()
        if fld.value is not None and fld.confidence is not None and fld.confidence < LOW_MODEL_CONFIDENCE
    })

    counts = {s.value: 0 for s in (MatchingStatusEnum.MATCH, MatchingStatusEnum.MISMATCH, MatchingStatusEnum.REVIEW)}
    for m in matching_results:
        if m.status.value in counts:
            counts[m.status.value] += 1

    failed_docs = [e.filename for e in extractions
                   if e.extraction_status in (ExtractionStatusEnum.FAILED, ExtractionStatusEnum.UNREADABLE)]
    unreadable_quality = [
        k for k, q in (quality_checks or {}).items() if not q.is_readable
    ]

    ok = not (critical_missing or ambiguous or counts["MISMATCH"] or counts["REVIEW"] or failed_docs)

    return {
        "documents_total": len(extractions),
        "documents_failed": failed_docs,
        "documents_low_technical_quality": len(unreadable_quality),
        "fields_present": len(present),
        "fields_total": len(ALL_FIELDS),
        "fields_missing": missing,
        "critical_fields_missing": critical_missing,
        "ambiguous_fields": ambiguous,
        "low_model_confidence_fields": low_conf,
        "matching": counts,
        "classification_conflicts": [e.filename for e in extractions if e.classification_conflict],
        "fallback_documents": [e.filename for e in extractions if e.fallback_used],
        "status": "OK" if ok else "ATTENTION",
    }
