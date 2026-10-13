"""
État du dossier supervisé par l'Orchestrateur Agentique (DossierState).
Conserve la mémoire immuable des transitions d'états, des résultats partiels et des fallbacks.
"""

from typing import Dict, List, Optional, Any
from datetime import datetime
from src.validation.schemas import (
    DocumentExtraction, DocumentQuality, MatchingResult,
    BusinessRuleResult, DossierDecision, AgentTrace,
    Anomaly, DossierScores, DossierReconstruction,
    StepEnum, StepStatusEnum
)


class DossierState:
    """
    Conteneur d'état complet d'un dossier de souscription.
    """
    def __init__(self, dossier_id: str):
        self.dossier_id = dossier_id
        self.created_at = datetime.now().isoformat()
        self.status: str = "INITIALIZED"
        self.file_paths: List[str] = []
        
        # Données collectées par étape
        self.quality_checks: Dict[str, DocumentQuality] = {}
        self.extractions: Dict[str, DocumentExtraction] = {}
        self.matching_results: List[MatchingResult] = []
        self.business_rule_results: List[BusinessRuleResult] = []
        self.decision: Optional[DossierDecision] = None
        self.reconstruction: Optional[DossierReconstruction] = None
        self.anomalies: List[Anomaly] = []
        self.scores: Optional[DossierScores] = None
        self.validation_issues: List[Dict[str, Any]] = []
        self.unreadable_documents: List[str] = []
        self.duplicate_groups: List[List[str]] = []
        
        # Flags et compteurs agentiques
        self.traces: List[AgentTrace] = []
        self.fallback_triggered: bool = False
        self.review_required: bool = False
        self.review_reasons: List[str] = []
        self.attempt_counts: Dict[str, int] = {}
        self.data_quality: Dict[str, Any] = {}
        self.evaluation: Optional[Dict[str, Any]] = None
        self.offline_mock_used: bool = False
        self.persistence_error: Optional[str] = None

    def add_trace(self, step: StepEnum, status: StepStatusEnum, message: str, duration_ms: int = 0, details: Optional[Dict[str, Any]] = None):
        """Enregistre un événement auditable dans la chronologie de l'agent."""
        trace = AgentTrace(
            step=step,
            status=status,
            timestamp=datetime.now().strftime("%H:%M:%S.%f")[:-3],
            duration_ms=duration_ms,
            message=message,
            details=details or {}
        )
        self.traces.append(trace)

    def mark_review(self, reason: str):
        """Déclenche la condition de contrôle humain (REVIEW_REQUIRED)."""
        self.review_required = True
        if reason not in self.review_reasons:
            self.review_reasons.append(reason)

    def to_dict(self) -> Dict[str, Any]:
        """Convertit l'état en dictionnaire sérialisable."""
        return {
            "dossier_id": self.dossier_id,
            "created_at": self.created_at,
            "status": self.status,
            "file_paths": self.file_paths,
            "quality_checks": {k: v.model_dump() for k, v in self.quality_checks.items()},
            "extractions": {k: v.model_dump() for k, v in self.extractions.items()},
            "matching_results": [m.model_dump() for m in self.matching_results],
            "business_rule_results": [b.model_dump() for b in self.business_rule_results],
            "decision": self.decision.model_dump() if self.decision else None,
            "reconstruction": self.reconstruction.model_dump() if self.reconstruction else None,
            "anomalies": [a.model_dump() for a in self.anomalies],
            "scores": self.scores.model_dump() if self.scores else None,
            "unreadable_documents": self.unreadable_documents,
            "traces": [t.model_dump() for t in self.traces],
            "fallback_triggered": self.fallback_triggered,
            "review_required": self.review_required,
            "review_reasons": self.review_reasons,
            "data_quality": self.data_quality,
            "offline_mock_used": self.offline_mock_used,
        }
