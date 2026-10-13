"""
Orchestrateur agentique (AgentOrchestrator).

Workflow, avec décisions intermédiaires :

  RECEIVED -> QUALITY_CHECK -> (qualité insuffisante ? PREPROCESSING -> QUALITY_CHECK)
           -> (toujours inexploitable ? REVIEW_REQUIRED, sans appel au VLM)
           -> CLASSIFICATION -> EXTRACTION_VLM -> (champ critique manquant ? QUALITY_CHECK -> PREPROCESSING ->
              nouvelle extraction ciblée, 2 tentatives max, contradiction => REVIEW_REQUIRED)
           -> VALIDATION -> RECONSTRUCTION -> NORMALIZATION -> MATCHING -> DATA_QUALITY (scores + anomalies)
           -> BUSINESS_RULES -> DECISION -> PERSISTENCE (SQLite)
"""

import hashlib
import logging
from pathlib import Path
from typing import Dict, List, Optional
from datetime import datetime

from src.agent.state import DossierState
from src.agent.trace import StepTimer
from src.config import QUALITY_UNUSABLE_THRESHOLD, UNREADABLE_DOCUMENT_MESSAGE
from src.documents.classifier import DocumentClassifier
from src.documents.loader import DocumentLoader
from src.preprocessing.quality import analyze_document_quality
from src.preprocessing.image_processor import preprocess_document_image
from src.preprocessing.fallback import FallbackAgent
from src.vision.extraction import VLMExtractionService
from src.matching.matcher import CrossDocumentMatcher
from src.business.rules import BusinessRulesEngine
from src.business.decision import UnderwritingDecisionEngine
from src.business.anomalies import detect_anomalies
from src.agent.reconstruction import reconstruct_dossier
from src.validation.scoring import compute_scores
from src.database.repository import SouscriptionRepository
from src.evaluation.evaluator import DossierEvaluator
from src.validation.validators import validate_extraction, assess_data_quality
from src.validation.schemas import (
    StepEnum, StepStatusEnum, ExtractionStatusEnum, DecisionEnum
)

logger = logging.getLogger(__name__)


class SouscriptionAgentOrchestrator:
    """Agent autonome de souscription d'assurance deux-roues."""

    def __init__(
        self,
        repository: Optional[SouscriptionRepository] = None,
        vlm_service: Optional[VLMExtractionService] = None,
        classifier: Optional[DocumentClassifier] = None,
        evaluator: Optional[DossierEvaluator] = None,
    ):
        self.repository = repository or SouscriptionRepository()
        self.vlm_service = vlm_service or VLMExtractionService()
        self.classifier = classifier or DocumentClassifier(self.vlm_service.client)
        self.evaluator = evaluator or DossierEvaluator()

    def process_dossier(self, file_paths: List[str], dossier_id: Optional[str] = None) -> DossierState:
        """Exécute le workflow agentique séquentiel et adaptatif."""
        d_id = dossier_id or f"DOS-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
        state = DossierState(dossier_id=d_id)
        state.file_paths = file_paths

        # --- ÉTAPE 1 : RÉCEPTION ---
        with StepTimer(state, StepEnum.RECEIVED, "Réception et inventaire des pièces justificatives") as timer:
            loaded_docs = DocumentLoader.load_documents(file_paths)
            if not loaded_docs:
                timer.fail("Aucun document recevable fourni.")
                state.mark_review("Dossier vide ou formats de fichiers non supportés.")
                self._finalize_empty_dossier(state)
                return self._persist(state)
            timer.complete(f"{len(loaded_docs)} document(s) réceptionné(s) et validé(s).")

        # --- ÉTAPE 2 : CONTRÔLE QUALITÉ -> PRÉTRAITEMENT -> NOUVEAU CONTRÔLE ---
        processed_paths = {}
        unusable = set()
        for doc in loaded_docs:
            p = doc["path"]
            with StepTimer(state, StepEnum.QUALITY_CHECK, f"Contrôle qualité OpenCV : {doc['filename']}") as timer:
                q_res = analyze_document_quality(p)
                state.quality_checks[p] = q_res
                if q_res.is_readable:
                    processed_paths[p] = p
                    timer.complete(f"Qualité satisfaisante ({q_res.quality_score*100:.1f}%, netteté: {q_res.sharpness:.1f}).")
                    continue

                timer.complete(
                    f"Qualité technique basse ({q_res.quality_score*100:.1f}%) : rehaussement de contraste nécessaire.",
                    status=StepStatusEnum.WARNING,
                    details={"issues": q_res.issues}
                )

            with StepTimer(state, StepEnum.PREPROCESSING, f"Rehaussement adaptatif CLAHE : {doc['filename']}") as prep_timer:
                enhanced_p = preprocess_document_image(p)
                processed_paths[p] = enhanced_p
                prep_timer.complete("Prétraitement appliqué.")

            with StepTimer(state, StepEnum.QUALITY_CHECK, f"Nouveau contrôle qualité : {doc['filename']}") as timer:
                q_after = analyze_document_quality(enhanced_p)
                state.quality_checks[p] = q_after
                if q_after.quality_score < QUALITY_UNUSABLE_THRESHOLD:
                    unusable.add(p)
                    state.unreadable_documents.append(doc["filename"])
                    state.mark_review(f"Document inexploitable après prétraitement : {doc['filename']}")
                    timer.complete(
                        f"Qualité toujours insuffisante ({q_after.quality_score*100:.1f}%) : envoi au VLM annulé.",
                        status=StepStatusEnum.WARNING, details={"issues": q_after.issues}
                    )
                elif q_after.is_readable:
                    timer.complete(f"Qualité satisfaisante après prétraitement ({q_after.quality_score*100:.1f}%).")
                else:
                    timer.complete(
                        f"Qualité encore limitée ({q_after.quality_score*100:.1f}%) : extraction tentée, résultats à contrôler.",
                        status=StepStatusEnum.WARNING, details={"issues": q_after.issues}
                    )

        # --- VLM disponible ? (sinon aucune donnée n'est inventée) ---
        vlm_ready = self.vlm_service.is_available()
        if not vlm_ready:
            with StepTimer(state, StepEnum.EXTRACTION_VLM, "Vérification de la disponibilité du VLM") as timer:
                timer.complete(
                    "VLM indisponible (GEMINI_API_KEY absente) : aucune extraction réalisée, aucune valeur inventée.",
                    status=StepStatusEnum.FAILED
                )
            state.mark_review("VLM indisponible : aucune clé GEMINI_API_KEY configurée.")

        # --- ÉTAPES 3 à 5 : CLASSIFICATION, EXTRACTION, FALLBACK ---
        for doc in loaded_docs:
            p = doc["path"]
            effective_path = processed_paths.get(p, p)

            if p in unusable or not vlm_ready:
                reason = UNREADABLE_DOCUMENT_MESSAGE if p in unusable else "VLM indisponible"
                state.extractions[p] = VLMExtractionService.empty_extraction(
                    p, ExtractionStatusEnum.UNREADABLE if p in unusable else ExtractionStatusEnum.FAILED, reason
                )
                continue

            # CLASSIFICATION
            with StepTimer(state, StepEnum.CLASSIFICATION, f"Classification du document : {doc['filename']}") as timer:
                classification = self.classifier.classify(effective_path)
                if classification.error:
                    timer.complete(f"Classification impossible : {classification.error}", status=StepStatusEnum.WARNING)
                else:
                    timer.complete(
                        f"Type identifié : {classification.document_type.value}",
                        details={"confidence_model": classification.confidence, "justification": classification.justification}
                    )

            # EXTRACTION VLM
            with StepTimer(state, StepEnum.EXTRACTION_VLM, f"Extraction VLM multimodale : {doc['filename']}") as timer:
                extraction = self.vlm_service.extract_document(
                    effective_path, doc_id=doc["stem"],
                    declared_type=classification.document_type,
                    classification_confidence=classification.confidence,
                )
                extraction.filename = doc["filename"]
                for fld in extraction.fields.values():
                    fld.source_document = doc["filename"]
                state.extractions[p] = extraction

                if extraction.source == "OFFLINE_MOCK" or classification.source == "OFFLINE_MOCK":
                    state.offline_mock_used = True

                if extraction.extraction_status == ExtractionStatusEnum.FAILED:
                    state.mark_review(f"Extraction impossible pour {doc['filename']} : {extraction.error}")
                    timer.complete(f"Échec de l'extraction : {extraction.error}", status=StepStatusEnum.FAILED)
                    continue

                if extraction.classification_conflict:
                    state.mark_review(f"Type de document ambigu pour {doc['filename']} (classification ≠ extraction)")

                # CONTRÔLE DE COMPLÉTUDE & FALLBACK CIBLÉ
                should_fallback, fallback_reason = FallbackAgent.should_trigger_fallback(
                    extraction.fields, extraction.document_type.value
                )
                if not should_fallback:
                    suffix = " [MODE TEST HORS-LIGNE]" if extraction.source == "OFFLINE_MOCK" else ""
                    timer.complete(f"Extraction réussie : {extraction.document_type.value} ({len(extraction.fields)} champs).{suffix}")
                    continue

                state.fallback_triggered = True
                timer.complete(
                    f"Champs critiques incomplets : {fallback_reason}. Activation du fallback ciblé.",
                    status=StepStatusEnum.FALLBACK, details={"reason": fallback_reason}
                )

            # FALLBACK : champ manquant -> Quality Check -> Prétraitement -> nouvelle extraction -> comparaison
            with StepTimer(state, StepEnum.PREPROCESSING, f"Seconde tentative (fallback ciblé) : {doc['filename']}") as fb_timer:
                extraction.first_attempt_fields = {k: v.model_copy(deep=True) for k, v in extraction.fields.items()}
                enhanced_fb_path = preprocess_document_image(effective_path)
                q_fb = analyze_document_quality(enhanced_fb_path)
                if q_fb.quality_score < QUALITY_UNUSABLE_THRESHOLD:
                    # Même après prétraitement ciblé : on n'insiste pas, aucune valeur n'est estimée.
                    state.unreadable_documents.append(doc["filename"])
                    state.mark_review(f"Document inexploitable après prétraitement : {doc['filename']}")
                    fb_timer.complete(
                        f"Qualité insuffisante après prétraitement ({q_fb.quality_score*100:.1f}%) : seconde extraction annulée.",
                        status=StepStatusEnum.WARNING, details={"issues": q_fb.issues}
                    )
                    continue

                fb_fields = self.vlm_service.targeted_fallback_extract(
                    enhanced_fb_path, FallbackAgent.CRITICAL_FIELDS, doc["stem"]
                )
                reconciled, contradiction, contra_msg = FallbackAgent.reconcile_attempts(extraction.fields, fb_fields)
                extraction.fields = reconciled
                extraction.refresh_confidence()
                extraction.attempts_count = 2
                extraction.fallback_used = True
                extraction.fallback_reason = fallback_reason
                extraction.extraction_status = ExtractionStatusEnum.FALLBACK_APPLIED
                if contradiction:
                    state.mark_review(f"Contradiction entre tentatives : {contra_msg}")
                    fb_timer.complete(f"Contradiction détectée ({contra_msg}) : contrôle requis.", status=StepStatusEnum.WARNING)
                else:
                    fb_timer.complete("Réconciliation des deux tentatives sans contradiction.")

        extractions = list(state.extractions.values())

        # --- ÉTAPE 6 : VALIDATION (Pydantic + plausibilité) ---
        with StepTimer(state, StepEnum.VALIDATION, "Validation Pydantic et plausibilité des valeurs") as timer:
            issues = [i for ext in extractions for i in validate_extraction(ext)]
            state.validation_issues = issues
            if issues:
                for i in issues:
                    state.mark_review(f"{i['document']} : {i['issue']}")
                timer.complete(f"{len(issues)} anomalie(s) de validation.", status=StepStatusEnum.WARNING,
                               details={"issues": issues})
            else:
                timer.complete("Toutes les valeurs extraites sont valides.")

        # --- ÉTAPE 7 : RECONSTRUCTION DU DOSSIER (identité / véhicule / autres) ---
        with StepTimer(state, StepEnum.RECONSTRUCTION, "Reconstruction du dossier (identité, véhicule, autres)") as timer:
            state.reconstruction = reconstruct_dossier(extractions)
            rec = state.reconstruction
            summary = (f"Identité : {len(rec.identite)} pièce(s) ; véhicule : {len(rec.vehicule)} ; "
                       f"autres : {len(rec.autres)} ; non identifiées : {len(rec.non_identifies)}.")
            if rec.pieces_manquantes:
                timer.complete(summary + f" Catégorie(s) sans pièce : {', '.join(rec.pieces_manquantes)}.",
                               status=StepStatusEnum.WARNING, details=rec.model_dump())
            else:
                timer.complete(summary, details=rec.model_dump())

        # --- ÉTAPE 8 : NORMALISATION (appliquée à l'extraction ; bilan réel) ---
        with StepTimer(state, StepEnum.NORMALIZATION, "Bilan de normalisation des champs") as timer:
            filled = [f for e in extractions for f in e.fields.values() if f.value is not None]
            changed = [f for f in filled if f.raw_value is not None and str(f.raw_value).strip() != str(f.value)]
            timer.complete(
                f"{len(filled)} champ(s) renseigné(s), dont {len(changed)} normalisé(s) "
                f"(CIN 8 chiffres, cylindrée entière, dates JJ/MM/AAAA).",
                details={"filled": len(filled), "normalized": len(changed)}
            )

        # --- ÉTAPE 9 : MATCHING INTER-DOCUMENTAIRE ---
        with StepTimer(state, StepEnum.MATCHING, "Vérification croisée inter-documents") as timer:
            state.matching_results = CrossDocumentMatcher.compare_documents(extractions)
            timer.complete(f"{len(state.matching_results)} point(s) de contrôle croisé vérifié(s).")

        # --- ÉTAPE 10 : DATA QUALITY (comptages, scores interprétables, anomalies FRA-xx) ---
        with StepTimer(state, StepEnum.DATA_QUALITY, "Contrôle de la qualité des données du dossier") as timer:
            state.data_quality = assess_data_quality(extractions, state.matching_results, state.quality_checks)
            state.scores = compute_scores(extractions, state.matching_results)
            state.duplicate_groups = self._find_duplicates(state)
            state.anomalies = detect_anomalies(
                extractions, state.matching_results, state.validation_issues, state.duplicate_groups
            )
            for a in state.anomalies:
                # Les anomalies issues du matching et de la validation ont déjà leur propre motif de contrôle ;
                # on ajoute celles qui ne sont couvertes par aucune règle.
                if a.code in ("FRA-06", "FRA-07", "FRA-08"):
                    state.mark_review(f"{a.code} {a.name} : {a.message}")

            dq = state.data_quality
            dq["anomalies"] = [a.code for a in state.anomalies]
            gdc = state.scores.global_data_confidence
            timer.complete(
                f"{dq['fields_present']}/{dq['fields_total']} champs renseignés ; "
                f"matching : {dq['matching']['MATCH']} MATCH, {dq['matching']['REVIEW']} REVIEW, {dq['matching']['MISMATCH']} MISMATCH ; "
                f"{len(state.anomalies)} anomalie(s) ; confiance globale des données : "
                f"{'n/a' if gdc is None else f'{gdc*100:.0f}%'} (indicateur non calibré).",
                status=StepStatusEnum.SUCCESS if dq["status"] == "OK" and not state.anomalies else StepStatusEnum.WARNING,
                details=dq
            )

        # --- ÉTAPE 11 : RÈGLES MÉTIER ---
        with StepTimer(state, StepEnum.BUSINESS_RULES, "Évaluation des critères d'éligibilité assurance") as timer:
            state.business_rule_results = BusinessRulesEngine.evaluate_dossier(extractions, state.matching_results)
            timer.complete(f"{len(state.business_rule_results)} règles métier appliquées.")

        # --- ÉTAPE 12 : DÉCISION ---
        with StepTimer(state, StepEnum.DECISION, "Arbitrage et décision finale") as timer:
            decision = UnderwritingDecisionEngine.compute_decision(
                state.business_rule_results, anomalies=state.anomalies, review_reasons=state.review_reasons
            )
            if state.unreadable_documents:
                decision.reason = f"{decision.reason} | {UNREADABLE_DOCUMENT_MESSAGE}"
            state.decision = decision
            state.status = decision.decision.value
            codes = f" [{', '.join(decision.rejection_codes)}]" if decision.rejection_codes else ""
            timer.complete(f"Décision finale : {decision.decision.value}{codes} ({decision.reason})")

        # --- ÉVALUATION (seulement avec vérité terrain, jamais en mode test hors-ligne) ---
        try:
            state.evaluation = self.evaluator.evaluate(state)
        except Exception:
            logger.exception("Évaluation impossible pour %s", state.dossier_id)

        # --- PERSISTANCE SQLITE (transaction unique : tout est enregistré, ou rien) ---
        return self._persist(state)

    def _persist(self, state: DossierState) -> DossierState:
        state.add_trace(StepEnum.PERSISTENCE, StepStatusEnum.SUCCESS,
                        "Enregistrement SQLite des résultats (transaction unique)")
        try:
            self.repository.save_dossier(state)
        except Exception as e:
            logger.exception("Échec de l'enregistrement SQLite pour %s", state.dossier_id)
            state.persistence_error = str(e)
            state.add_trace(StepEnum.PERSISTENCE, StepStatusEnum.FAILED, f"Échec de l'enregistrement : {e}")
        return state

    @staticmethod
    def _find_duplicates(state: DossierState) -> List[List[str]]:
        """Groupes de fichiers strictement identiques (même empreinte SHA-256) parmi les pièces du dossier."""
        by_hash: Dict[str, List[str]] = {}
        for path, ext in state.extractions.items():
            try:
                digest = hashlib.sha256(Path(path).read_bytes()).hexdigest()
            except OSError:
                continue
            by_hash.setdefault(digest, []).append(ext.filename)
        return [names for names in by_hash.values() if len(names) > 1]

    def _finalize_empty_dossier(self, state: DossierState):
        """Finalise un dossier ne contenant aucune pièce exploitable."""
        with StepTimer(state, StepEnum.DECISION, "Décision automatique") as timer:
            decision = UnderwritingDecisionEngine.compute_decision([])
            decision.decision = DecisionEnum.REVIEW_REQUIRED
            decision.reason = "Dossier incomplet : aucune pièce justificative valide n'a pu être chargée."
            state.decision = decision
            state.status = "REVIEW_REQUIRED"
            timer.complete("Dossier orienté vers contrôle gestionnaire.")
