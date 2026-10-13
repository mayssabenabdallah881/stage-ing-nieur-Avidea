"""
Classification du type de document par le VLM (étape CLASSIFICATION).
Une requête dédiée, avant l'extraction : le type retenu sert de référence pour la suite du pipeline.
"""

from typing import Optional
from pydantic import BaseModel

from src.validation.schemas import DocumentTypeEnum
from src.vision.gemini_client import GeminiVLMClient
from src.vision.prompts import CLASSIFIER_PROMPT


class ClassificationResult(BaseModel):
    document_type: DocumentTypeEnum = DocumentTypeEnum.UNKNOWN
    confidence: Optional[float] = None  # confiance déclarée par le modèle, non calibrée
    justification: Optional[str] = None
    error: Optional[str] = None
    source: Optional[str] = None


class DocumentClassifier:
    """Identifie le type d'un document (CIN, contrat, quittance...) sans jamais forcer une réponse."""

    def __init__(self, client: Optional[GeminiVLMClient] = None):
        self.client = client or GeminiVLMClient()

    def classify(self, image_path: str) -> ClassificationResult:
        result = self.client.call_vlm(image_path, CLASSIFIER_PROMPT)

        if "error" in result:
            return ClassificationResult(error=str(result["error"]), source=result.get("_source"))

        try:
            doc_type = DocumentTypeEnum(result.get("document_type", "UNKNOWN"))
        except ValueError:
            doc_type = DocumentTypeEnum.UNKNOWN

        conf = result.get("confidence")
        return ClassificationResult(
            document_type=doc_type,
            confidence=float(conf) if isinstance(conf, (int, float)) else None,
            justification=result.get("justification"),
            source=result.get("_source"),
        )
