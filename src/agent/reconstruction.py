"""
Reconstruction du dossier (rapport, chapitres 4 et 7).

Les pièces classifiées sont regroupées en trois catégories :

    DOSSIER
    +-- Identité : CIN, PASSPORT                (pièces exigées, configurables)
    +-- Véhicule : CONTRAT, FACTURE             (pièces exigées, configurables)
    +-- Autres   : CARTE_GRISE, QUITTANCE, PERMIS

et chaque champ du dossier est consolidé : la valeur retenue est la première valeur NON NULLE, en interrogeant
d'abord la pièce la plus pertinente pour ce champ (ex. le châssis d'abord sur la carte grise, l'identité d'abord
sur la CIN). Rien n'est déduit ni remplacé : un champ absent partout reste absent.
Les désaccords entre pièces ne sont PAS tranchés ici : ils sont détectés par le matching.
"""

from typing import Any, Dict, List, Optional, Tuple

from src.config import REQUIRED_IDENTITY_TYPES, REQUIRED_VEHICLE_TYPES, REQUIRED_PAYMENT_TYPES

from src.validation.schemas import (
    DocumentExtraction, DocumentTypeEnum as T, DossierReconstruction, ExtractedField, ExtractionStatusEnum,
)

# Catégories pilotées par config/business_rules.yaml (required_documents) :
# identité = pièces d'identité exigées ; véhicule = pièces de vente exigées ; autres = toute autre pièce identifiée.
IDENTITY_TYPES = tuple(T(t) for t in REQUIRED_IDENTITY_TYPES)
VEHICLE_TYPES = tuple(T(t) for t in REQUIRED_VEHICLE_TYPES)
PAYMENT_TYPES = tuple(T(t) for t in REQUIRED_PAYMENT_TYPES)
OTHER_TYPES = tuple(t for t in T if t not in IDENTITY_TYPES + VEHICLE_TYPES + PAYMENT_TYPES and t != T.UNKNOWN)

_IDENTITY_FIRST = (T.CIN, T.PASSPORT, T.PERMIS, T.CONTRAT, T.FACTURE, T.CARTE_GRISE, T.QUITTANCE)
FIELD_SOURCE_PRIORITY: Dict[str, Tuple[T, ...]] = {
    "nom": _IDENTITY_FIRST,
    "prenom": _IDENTITY_FIRST,
    "cin_passeport": (T.CIN, T.PASSPORT, T.CONTRAT, T.FACTURE, T.CARTE_GRISE, T.PERMIS, T.QUITTANCE),
    "nom_prenom_acheteur": (T.CIN, T.PASSPORT, T.CONTRAT, T.FACTURE, T.CARTE_GRISE, T.PERMIS, T.QUITTANCE),
    "numero_chassis": (T.CARTE_GRISE, T.FACTURE, T.CONTRAT, T.QUITTANCE),
    "cylindree": (T.CARTE_GRISE, T.FACTURE, T.CONTRAT, T.QUITTANCE),
    "date_vente": (T.FACTURE, T.CONTRAT, T.QUITTANCE),
    "date_mec": (T.CARTE_GRISE, T.CONTRAT, T.FACTURE),
}
FIELD_ORDER = list(FIELD_SOURCE_PRIORITY)


def category_of(doc_type: T) -> str:
    if doc_type in IDENTITY_TYPES:
        return "IDENTITE"
    if doc_type in VEHICLE_TYPES:
        return "VEHICULE"
    if doc_type in PAYMENT_TYPES:
        return "PAIEMENT"
    if doc_type in OTHER_TYPES:
        return "AUTRES"
    return "NON_IDENTIFIE"


def _is_identified(ext: DocumentExtraction) -> bool:
    return ext.document_type != T.UNKNOWN and ext.extraction_status not in (
        ExtractionStatusEnum.FAILED, ExtractionStatusEnum.UNREADABLE)


def missing_required_documents(rec: DossierReconstruction) -> List[str]:
    """Catégories exigées sans aucune pièce identifiée (liste vide : exigence désactivée dans la configuration)."""
    out = []
    if IDENTITY_TYPES and not rec.identite:
        out.append(f"IDENTITE ({' ou '.join(REQUIRED_IDENTITY_TYPES)})")
    if VEHICLE_TYPES and not rec.vehicule:
        out.append(f"VEHICULE ({' ou '.join(REQUIRED_VEHICLE_TYPES)})")
    if PAYMENT_TYPES and not rec.paiement:
        out.append(f"PAIEMENT ({' ou '.join(REQUIRED_PAYMENT_TYPES)})")
    return out


def consolidate(extractions: List[DocumentExtraction], first_attempt: bool = False
                ) -> Dict[str, Tuple[Any, ExtractedField, str]]:
    """Pour chaque champ renseigné : (valeur retenue, champ source, nom du fichier source)."""
    out: Dict[str, Tuple[Any, ExtractedField, str]] = {}
    for name in FIELD_ORDER:
        priority = FIELD_SOURCE_PRIORITY[name]
        ranked = sorted(
            enumerate(extractions),
            key=lambda it: (priority.index(it[1].document_type) if it[1].document_type in priority else len(priority), it[0]),
        )
        for _, ext in ranked:
            fields = ext.first_attempt_fields if first_attempt and ext.first_attempt_fields else ext.fields
            f = fields.get(name)
            if f is not None and f.value is not None and str(f.value) != "":
                out[name] = (f.value, f, ext.filename)
                break
    return out


def consolidate_fields(extractions: List[DocumentExtraction], first_attempt: bool = False) -> Dict[str, Any]:
    """Valeur retenue par champ (champs absents omis)."""
    return {k: v[0] for k, v in consolidate(extractions, first_attempt).items()}


def reconstruct_dossier(extractions: List[DocumentExtraction]) -> DossierReconstruction:
    rec = DossierReconstruction()
    for ext in extractions:
        if not _is_identified(ext):
            rec.non_identifies.append(ext.filename)
            continue
        cat = category_of(ext.document_type)
        {"IDENTITE": rec.identite, "VEHICULE": rec.vehicule, "PAIEMENT": rec.paiement, "AUTRES": rec.autres}[cat].append(ext.filename)
        if ext.document_type.value not in rec.types_presents:
            rec.types_presents.append(ext.document_type.value)

    rec.pieces_manquantes = missing_required_documents(rec)

    detail = consolidate(extractions)
    rec.consolidated = {k: v[0] for k, v in detail.items()}
    rec.consolidated_source = {k: v[2] for k, v in detail.items()}
    return rec
