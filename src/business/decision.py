"""
Moteur de décision finale de souscription d'assurance deux-roues (rapport, chapitre 8 et annexe B).
Garantit une justification explicite et explicable.

États possibles : ACCEPTED, REJECTED, REVIEW_REQUIRED.

Ordre de priorité (annexe B du rapport) :
    0. pièce obligatoire manquante (identité / véhicule) -> DOCUMENT_MANQUANT
    1. donnée critique manquante                       -> REVIEW_REQUIRED
    2. condition d'inéligibilité clairement identifiée -> REJECTED  (codes REJ-01, REJ-02, REJ-03)
    3. contradiction / ambiguïté / anomalie            -> REVIEW_REQUIRED
    4. sinon                                           -> ACCEPTED
"""

from typing import List, Optional

from src.business.codes import REJECTION_CODES
from src.validation.schemas import (
    Anomaly, BusinessRuleResult, BusinessRuleStatusEnum,
    DossierDecision, DecisionEnum
)


class UnderwritingDecisionEngine:
    """Synthèse des règles métier et arbitrage de la décision finale."""

    @classmethod
    def compute_decision(
        cls,
        rules: List[BusinessRuleResult],
        anomalies: Optional[List[Anomaly]] = None,
        review_reasons: Optional[List[str]] = None,
    ) -> DossierDecision:
        """
        `anomalies` et `review_reasons` (signaux collectés par l'orchestrateur : contradiction entre tentatives,
        valeur invalide, document inexploitable...) ne peuvent jamais produire un rejet : ils ramènent au mieux
        un dossier conforme à REVIEW_REQUIRED.
        """
        anomalies = anomalies or []
        review_reasons = review_reasons or []

        absent = [r for r in rules if r.status == BusinessRuleStatusEnum.DOCUMENT_MANQUANT]
        failed = [r for r in rules if r.status == BusinessRuleStatusEnum.FAILED]
        review = [r for r in rules if r.status == BusinessRuleStatusEnum.REVIEW_REQUIRED]
        missing = [r for r in review if r.missing_data]
        explanation = cls._explain(rules, anomalies, review_reasons)

        # 0. Pièce obligatoire absente : le dossier est incomplet, aucune autre conclusion n'est tirée
        if absent:
            return DossierDecision(
                decision=DecisionEnum.DOCUMENT_MANQUANT,
                reason=f"{absent[0].message} Merci de compléter le dossier.",
                explanation=explanation, rules=rules,
                evidence={"pieces_manquantes": absent[0].evidence.get("pieces_manquantes", [])},
            )

        # 1. Donnée critique manquante : jamais de rejet ni d'acceptation sur des données incomplètes
        if missing:
            return DossierDecision(
                decision=DecisionEnum.REVIEW_REQUIRED,
                reason=f"Vérification gestionnaire requise : {missing[0].message}",
                explanation=explanation, rules=rules,
                evidence={"review_reasons": [r.message for r in review], "missing_critical_data": True},
            )

        # 2. Rejet formel
        if failed:
            codes = sorted({r.code or "REJ-03" for r in failed})
            types = sorted({r.rejection_type for r in failed if r.rejection_type})
            labels = " ; ".join(f"{c} {REJECTION_CODES[c]}" for c in codes)
            primary = " ; ".join(r.message for r in failed)
            return DossierDecision(
                decision=DecisionEnum.REJECTED,
                reason=f"Souscription REFUSÉE [{labels}] : {primary}",
                rejection_codes=codes, rejection_type=" ; ".join(types) or None, explanation=explanation, rules=rules,
                evidence={"failed_rules_count": len(failed)},
            )

        # 3. Contrôle humain requis
        if review:
            return DossierDecision(
                decision=DecisionEnum.REVIEW_REQUIRED,
                reason=f"Vérification gestionnaire requise : {review[0].message}",
                explanation=explanation, rules=rules,
                evidence={"review_reasons": [r.message for r in review]},
            )
        if review_reasons or anomalies:
            reasons = review_reasons or [f"{a.code} {a.name} : {a.message}" for a in anomalies]
            return DossierDecision(
                decision=DecisionEnum.REVIEW_REQUIRED,
                reason=f"Contrôle requis : {' ; '.join(reasons)}",
                explanation=explanation, rules=rules,
                evidence={"review_reasons": reasons, "anomalies": [a.code for a in anomalies]},
            )

        # 4. Approbation automatique
        return DossierDecision(
            decision=DecisionEnum.ACCEPTED,
            reason="Dossier COMPLET et CONFORME. Toutes les conditions d'éligibilité deux-roues sont validées.",
            explanation=explanation, rules=rules,
            evidence={"status": "ALL_RULES_PASSED"},
        )

    @staticmethod
    def _explain(rules: List[BusinessRuleResult], anomalies: List[Anomaly], review_reasons: List[str]) -> List[str]:
        lines = [f"[{r.status.value}] {r.rule_name}{' (' + r.code + ')' if r.code else ''} : {r.message}" for r in rules]
        lines += [f"[ANOMALIE {a.severity}] {a.code} {a.name} : {a.message}" for a in anomalies]
        lines += [f"[CONTRÔLE] {reason}" for reason in review_reasons]
        return lines
