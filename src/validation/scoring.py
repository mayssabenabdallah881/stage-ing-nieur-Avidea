"""
Feature engineering et scores interprétables (rapport, chapitre 9).

Ce sont des INDICATEURS INTERPRÉTABLES conçus pour le prototype, calculés uniquement à partir de comptages et de
résultats de comparaison réels. Ce ne sont PAS des probabilités calibrées (de fraude ou de justesse) : une
calibration exigerait un protocole statistique sur un corpus annoté bien plus large.
Quand un score n'est pas calculable (aucune comparaison possible, aucun champ renseigné...) il vaut None :
on ne comble jamais par une valeur par défaut.

Variables :
    missing_field_count = somme_i I(champ_i = null)           (champs consolidés du dossier)
    completeness        = champs renseignés / champs totaux
    identity_match, chassis_match, date_consistent, displacement_consistent  dans {0, 1} (None si non comparable)
Scores :
    extraction_score        moyenne des confiances déclarées par le VLM sur les champs renseignés (non calibrée)
    completeness_score      = completeness
    validity_score          champs renseignés non douteux / champs renseignés
    identity_match_score    part des comparaisons d'identité (CIN/passeport, nom, prénom, nom complet) en MATCH
    chassis_match_score     part des comparaisons de châssis en MATCH
    displacement_consistency  part des comparaisons de cylindrée en MATCH
    date_consistency        part des comparaisons de dates (vente, mise en circulation) en MATCH
    global_data_confidence  moyenne pondérée des scores disponibles (poids ci-dessous, renormalisés)
"""

from typing import Dict, List, Optional, Tuple

from src.agent.reconstruction import consolidate_fields
from src.validation.schemas import DocumentExtraction, DossierScores, MatchingResult, MatchingStatusEnum
from src.validation.validators import ALL_FIELDS

IDENTITY_FIELDS = ("cin_passeport", "nom", "prenom", "nom_prenom_acheteur")
CHASSIS_FIELDS = ("numero_chassis",)
DISPLACEMENT_FIELDS = ("cylindree",)
DATE_FIELDS = ("date_vente", "date_mec")

GLOBAL_WEIGHTS: Dict[str, float] = {
    "extraction_score": 0.15,
    "completeness_score": 0.20,
    "validity_score": 0.20,
    "identity_match_score": 0.15,
    "chassis_match_score": 0.15,
    "displacement_consistency": 0.075,
    "date_consistency": 0.075,
}


def _ratio(num: int, den: int) -> Optional[float]:
    return round(num / den, 4) if den else None


def _match_share(results: List[MatchingResult], fields: Tuple[str, ...]) -> Tuple[Optional[float], Optional[int]]:
    """(part de MATCH, indicateur binaire 0/1) pour un groupe de champs ; (None, None) si rien n'a été comparé."""
    relevant = [m for m in results if m.field in fields]
    if not relevant:
        return None, None
    matches = sum(1 for m in relevant if m.status == MatchingStatusEnum.MATCH)
    return _ratio(matches, len(relevant)), int(matches == len(relevant))


def compute_scores(extractions: List[DocumentExtraction], matching_results: List[MatchingResult]) -> DossierScores:
    consolidated = consolidate_fields(extractions)
    filled = [n for n in ALL_FIELDS if n in consolidated]

    # Fiabilité de lecture : tous les champs renseignés, toutes pièces confondues
    values = [f for e in extractions for f in e.fields.values() if f.value is not None and str(f.value) != ""]
    confidences = [f.confidence for f in values if isinstance(f.confidence, (int, float))]
    sound = [f for f in values if f.status not in ("AMBIGUOUS", "CONTRADICTORY")]

    scores = DossierScores(
        missing_field_count=len(ALL_FIELDS) - len(filled),
        fields_total=len(ALL_FIELDS),
        extraction_score=round(sum(confidences) / len(confidences), 4) if confidences else None,
        completeness_score=_ratio(len(filled), len(ALL_FIELDS)),
        validity_score=_ratio(len(sound), len(values)),
    )
    scores.identity_match_score, scores.identity_match = _match_share(matching_results, IDENTITY_FIELDS)
    scores.chassis_match_score, scores.chassis_match = _match_share(matching_results, CHASSIS_FIELDS)
    scores.displacement_consistency, scores.displacement_consistent = _match_share(matching_results, DISPLACEMENT_FIELDS)
    scores.date_consistency, scores.date_consistent = _match_share(matching_results, DATE_FIELDS)

    available = {k: getattr(scores, k) for k in GLOBAL_WEIGHTS if getattr(scores, k) is not None}
    if available:
        total_weight = sum(GLOBAL_WEIGHTS[k] for k in available)
        scores.global_data_confidence = round(sum(GLOBAL_WEIGHTS[k] * v for k, v in available.items()) / total_weight, 4)
        scores.components_used = list(available)
    return scores
