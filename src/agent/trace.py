"""
Gestionnaire des traces d'audit agentique.
Fournit le chronométrage précis et le formatage visuel des étapes pour la traçabilité.
"""

import time
from typing import Optional, Dict, Any
from src.agent.state import DossierState
from src.validation.schemas import StepEnum, StepStatusEnum


class StepTimer:
    """Gestionnaire de contexte pour chronométrer l'exécution de chaque étape agentique."""
    def __init__(self, state: DossierState, step: StepEnum, start_message: str):
        self.state = state
        self.step = step
        self.start_message = start_message
        self.start_time: float = 0.0

    def __enter__(self):
        self.start_time = time.time()
        self.state.add_trace(
            step=self.step,
            status=StepStatusEnum.RUNNING,
            message=self.start_message
        )
        return self

    def complete(self, message: str, status: StepStatusEnum = StepStatusEnum.SUCCESS, details: Optional[Dict[str, Any]] = None):
        duration_ms = int((time.time() - self.start_time) * 1000)
        self.state.add_trace(
            step=self.step,
            status=status,
            message=message,
            duration_ms=duration_ms,
            details=details or {}
        )

    def fail(self, error_message: str, details: Optional[Dict[str, Any]] = None):
        duration_ms = int((time.time() - self.start_time) * 1000)
        self.state.add_trace(
            step=self.step,
            status=StepStatusEnum.FAILED,
            message=f"Échec: {error_message}",
            duration_ms=duration_ms,
            details=details or {}
        )

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is not None:
            self.fail(str(exc_val))
        return False
