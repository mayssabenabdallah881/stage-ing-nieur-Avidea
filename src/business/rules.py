"""
Moteur de Règles Métier déterministe pour la souscription d'assurance deux-roues (rapport, chapitre 8).
Aucune IA générative n'intervient dans l'application des règles : logique 100% déterministe,
séparée du modèle d'intelligence artificielle.

Pièces obligatoires : au moins une pièce d'identité (CIN ou PASSPORT) ET une pièce de vente (CONTRAT ou FACTURE),
listes configurables ; sinon DOCUMENT_MANQUANT.

Règles de cylindrée (seuils lus dans config/business_rules.yaml) :
  - cylindrée > 125 cc                  -> FAILED, REJ-01, rejection_type PUISSANCE_SUP_125
  - cylindrée <= 49 cc                  -> condition satisfaite
  - 49 cc < cylindrée <= 125 cc         -> condition de date : la date de mise en circulation (date_mec) doit être
                                           explicitement identifiée. Une date_mec absente n'est JAMAIS remplacée par
                                           la date_vente : le dossier part en REVIEW_REQUIRED.
                                           Date postérieure au 01/01/2025     -> FAILED, REJ-02, rejection_type PUISSANCE_SUP_49_APRES_2025
"""

from datetime import datetime
from typing import Any, Dict, List, Optional

from src.config import (
    MAX_ALLOWED_DISPLACEMENT, SMALL_ENGINE_THRESHOLD, MEC_RESTRICTION_YEAR, RESTRICTION_DATE_FIELD, REJECTION_TYPES
)
from src.agent.reconstruction import consolidate, reconstruct_dossier
from src.validation.schemas import (
    BusinessRuleResult, BusinessRuleStatusEnum,
    DocumentExtraction, MatchingResult, MatchingStatusEnum
)

R = BusinessRuleStatusEnum
UNCERTAIN_STATUSES = ("AMBIGUOUS", "CONTRADICTORY")


class BusinessRulesEngine:
    """Moteur de règles d'éligibilité pour les deux-roues."""

    @classmethod
    def evaluate_dossier(
        cls,
        extractions: List[DocumentExtraction],
        matching_results: List[MatchingResult]
    ) -> List[BusinessRuleResult]:
        """Évalue toutes les règles d'éligibilité sur l'ensemble du dossier."""
        results: List[BusinessRuleResult] = []
        detail = consolidate(extractions)

        def value(name: str) -> Optional[Any]:
            return detail[name][0] if name in detail else None

        def uncertain(name: str) -> bool:
            return name in detail and detail[name][1].status in UNCERTAIN_STATUSES

        identity_val = value("cin_passeport")
        chassis_val = value("numero_chassis")
        cylindree_val = value("cylindree")

        # 0. RÈGLE 0 : pièces obligatoires (identité : CIN ou passeport ET véhicule : contrat ou facture)
        rec = reconstruct_dossier(extractions)
        if rec.pieces_manquantes and rec.non_identifies:
            # Une pièce illisible ou non identifiée pourrait être celle qui manque : on ne conclut pas à son absence.
            results.append(BusinessRuleResult(
                rule_name="REQUIRED_DOCUMENTS", status=R.REVIEW_REQUIRED, missing_data=True,
                message=f"Pièce(s) obligatoire(s) non retrouvée(s) ({', '.join(rec.pieces_manquantes)}) alors que "
                        f"{len(rec.non_identifies)} pièce(s) illisible(s) ou non identifiée(s) : vérification requise.",
                evidence={"pieces_manquantes": rec.pieces_manquantes, "non_identifies": rec.non_identifies}))
        elif rec.pieces_manquantes:
            results.append(BusinessRuleResult(
                rule_name="REQUIRED_DOCUMENTS", status=R.DOCUMENT_MANQUANT,
                message=f"Document manquant : {' ; '.join(rec.pieces_manquantes)}.",
                evidence={"pieces_manquantes": rec.pieces_manquantes, "types_presents": rec.types_presents}))
        else:
            results.append(BusinessRuleResult(
                rule_name="REQUIRED_DOCUMENTS", status=R.PASSED,
                message="Pièces obligatoires présentes (identité et véhicule).",
                evidence={"types_presents": rec.types_presents}))

        # 1. RÈGLE 1 : informations critiques présentes (identité, châssis)
        if identity_val is None or chassis_val is None:
            results.append(BusinessRuleResult(
                rule_name="MANDATORY_FIELDS_PRESENT", status=R.REVIEW_REQUIRED, missing_data=True,
                message="Informations obligatoires incomplètes (CIN/passeport ou N° de châssis manquant).",
                evidence={"cin_passeport": identity_val, "numero_chassis": chassis_val}
            ))
        else:
            results.append(BusinessRuleResult(
                rule_name="MANDATORY_FIELDS_PRESENT", status=R.PASSED,
                message="Toutes les informations obligatoires clés sont identifiées.",
                evidence={"cin_passeport": identity_val, "numero_chassis": chassis_val}
            ))

        # 2. RÈGLE 2 : cohérence inter-documents (jamais un refus automatique : contradiction => revue humaine)
        mismatches = [m for m in matching_results if m.status == MatchingStatusEnum.MISMATCH]
        reviews = [m for m in matching_results if m.status == MatchingStatusEnum.REVIEW]
        if mismatches:
            details = " ; ".join([m.message or f"{m.field} non concordant" for m in mismatches])
            results.append(BusinessRuleResult(
                rule_name="CROSS_DOCUMENT_CONSISTENCY", status=R.REVIEW_REQUIRED,
                message=f"Contradictions entre pièces justificatives, contrôle gestionnaire requis : {details}",
                evidence={"mismatches_count": len(mismatches)}
            ))
        elif reviews:
            results.append(BusinessRuleResult(
                rule_name="CROSS_DOCUMENT_CONSISTENCY", status=R.REVIEW_REQUIRED,
                message="Rapprochement documentaire ambigu. Contrôle par gestionnaire requis.",
                evidence={"reviews_count": len(reviews)}
            ))
        else:
            results.append(BusinessRuleResult(
                rule_name="CROSS_DOCUMENT_CONSISTENCY", status=R.PASSED,
                message="Concordance validée entre les pièces du dossier.",
                evidence={"matching_items_verified": len(matching_results)}
            ))

        # 3. RÈGLE 3 : cylindrée
        cyl_conflict = any(m.field == "cylindree" and m.status != MatchingStatusEnum.MATCH for m in matching_results)
        cyl_certain = cylindree_val is not None and not cyl_conflict and not uncertain("cylindree")

        if cylindree_val is None:
            results.append(BusinessRuleResult(
                rule_name="DISPLACEMENT_CHECK", status=R.REVIEW_REQUIRED, missing_data=True,
                message="Cylindrée non renseignée ou illisible sur les documents.",
                evidence={"cylindree": None}
            ))
        elif not cyl_certain:
            # Une valeur contradictoire ou douteuse ne doit jamais fonder un rejet ni une acceptation.
            results.append(BusinessRuleResult(
                rule_name="DISPLACEMENT_CHECK", status=R.REVIEW_REQUIRED,
                message=f"Cylindrée incertaine ({cylindree_val} cc : contradiction entre pièces ou valeur douteuse) : "
                        f"aucune décision automatique sur cette valeur.",
                evidence={"cylindree": cylindree_val, "conflict": cyl_conflict}
            ))
        elif cylindree_val > MAX_ALLOWED_DISPLACEMENT:
            results.append(BusinessRuleResult(
                rule_name="DISPLACEMENT_CHECK", status=R.FAILED, code="REJ-01", rejection_type=REJECTION_TYPES["displacement_above_limit"],
                message=f"Cylindrée de {cylindree_val} cc supérieure au seuil maximal autorisé ({MAX_ALLOWED_DISPLACEMENT} cc).",
                evidence={"cylindree": cylindree_val, "max_allowed": MAX_ALLOWED_DISPLACEMENT}
            ))
        elif cylindree_val <= SMALL_ENGINE_THRESHOLD:
            results.append(BusinessRuleResult(
                rule_name="DISPLACEMENT_CHECK", status=R.PASSED,
                message=f"Petite cylindrée ({cylindree_val} cc <= {SMALL_ENGINE_THRESHOLD} cc) : condition de cylindrée satisfaite.",
                evidence={"cylindree": cylindree_val}
            ))
        else:
            results.append(BusinessRuleResult(
                rule_name="DISPLACEMENT_CHECK", status=R.PASSED,
                message=f"Cylindrée conforme au plafond ({cylindree_val} cc <= {MAX_ALLOWED_DISPLACEMENT} cc), "
                        f"sous réserve de la condition de date.",
                evidence={"cylindree": cylindree_val}
            ))

        # 4. RÈGLE 4 : condition de date pour SMALL_ENGINE_THRESHOLD < cylindrée <= MAX_ALLOWED_DISPLACEMENT
        if cyl_certain and SMALL_ENGINE_THRESHOLD < cylindree_val <= MAX_ALLOWED_DISPLACEMENT:
            results.append(cls._date_condition(RESTRICTION_DATE_FIELD, value(RESTRICTION_DATE_FIELD),
                                               uncertain(RESTRICTION_DATE_FIELD), value("date_vente"), cylindree_val))

        return results

    @classmethod
    def _date_condition(cls, field: str, date_str: Optional[str], is_uncertain: bool,
                        other_date: Optional[str], cylindree: int) -> BusinessRuleResult:
        label = "date de mise en circulation (date_mec)" if field == "date_mec" else field
        threshold = f"01/01/{MEC_RESTRICTION_YEAR}"

        if not date_str:
            return BusinessRuleResult(
                rule_name="DATE_CONDITION", status=R.REVIEW_REQUIRED, missing_data=True,
                message=f"Cylindrée de {cylindree} cc : la {label} est manquante. Elle n'est pas remplacée par une "
                        f"autre date (la date de vente n'est jamais substituée) : vérification requise.",
                evidence={"cylindree": cylindree, "date_field": field, "date_vente_observee": other_date}
            )

        parsed = cls._parse_date(date_str)
        if parsed is None or is_uncertain:
            return BusinessRuleResult(
                rule_name="DATE_CONDITION", status=R.REVIEW_REQUIRED,
                message=f"La {label} n'est pas exploitable ou reste douteuse ({date_str}) : vérification requise.",
                evidence={"date": date_str, "date_field": field}
            )

        if parsed > datetime(MEC_RESTRICTION_YEAR, 1, 1):
            return BusinessRuleResult(
                rule_name="DATE_CONDITION", status=R.FAILED, code="REJ-02", rejection_type=REJECTION_TYPES["date_condition_not_met"],
                message=f"Véhicule {cylindree} cc : {label} postérieure au {threshold} ({date_str}) : "
                        f"condition de date non satisfaite.",
                evidence={"date": date_str, "date_field": field, "year_restriction": MEC_RESTRICTION_YEAR}
            )
        return BusinessRuleResult(
            rule_name="DATE_CONDITION", status=R.PASSED,
            message=f"{label.capitalize()} antérieure ou égale au {threshold} ({date_str}) : condition de date satisfaite.",
            evidence={"date": date_str, "date_field": field}
        )

    @staticmethod
    def _parse_date(date_str: str) -> Optional[datetime]:
        """JJ/MM/AAAA -> datetime ; None si la date n'est pas exploitable (jamais présumée valide)."""
        try:
            d, m, y = (int(x) for x in str(date_str).split("/"))
            return datetime(y, m, d)
        except (ValueError, TypeError):
            return None
