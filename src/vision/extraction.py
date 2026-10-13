"""
Orchestration de l'extraction documentaire par le VLM Gemini.
Assure l'étanchéité : aucune valeur n'est inventée, typage Pydantic strict et normalisation.
"""

from pathlib import Path
from typing import Dict, Any, Optional, List
from src.vision.gemini_client import GeminiVLMClient
from src.vision.prompts import EXTRACTION_PROMPT, FALLBACK_PROMPT
from src.validation.schemas import (
    DocumentExtraction, ExtractedField, DocumentTypeEnum, ExtractionStatusEnum
)
from src.validation.normalization import (
    normalize_cin_passeport, normalize_chassis, normalize_date,
    normalize_displacement, normalize_name
)

# Champs du rapport (chapitre 5) : nom, prenom, cin_passeport, nom_prenom_acheteur,
# numero_chassis, cylindree, date_vente, date_mec  (+ confidence, portée par chaque champ et par le document)
FIELD_NAMES: List[str] = [
    "nom", "prenom", "cin_passeport", "nom_prenom_acheteur",
    "cylindree", "date_vente", "date_mec", "numero_chassis",
]


def _normalize(field_name: str, value: Any, raw: Any):
    """Applique le normaliseur déterministe adapté au champ. Retourne (valeur_normalisée, valeur_brute)."""
    candidate = value if value not in (None, "") else raw
    if field_name == "cin_passeport":
        return normalize_cin_passeport(candidate)
    if field_name == "cylindree":
        return normalize_displacement(candidate)
    if field_name == "numero_chassis":
        return normalize_chassis(candidate)
    if field_name in ("date_vente", "date_mec"):
        return normalize_date(candidate)
    if field_name in ("nom", "prenom", "nom_prenom_acheteur"):
        return normalize_name(candidate)
    return candidate, raw


def _parse_visual_flags(raw: Any) -> List[str]:
    """Signes d'altération signalés par le VLM : liste de courtes phrases, vide si rien n'est visible."""
    if not raw:
        return []
    items = raw if isinstance(raw, list) else [raw]
    flags = []
    for item in items:
        text = str(item.get("description") or item) if isinstance(item, dict) else str(item)
        text = text.strip()
        if text and text.lower() not in ("null", "none", "aucun", "aucune", "[]"):
            flags.append(text)
    return flags


class VLMExtractionService:
    """Service d'analyse documentaire Vision-Language."""

    def __init__(self, gemini_client: Optional[GeminiVLMClient] = None):
        self.client = gemini_client or GeminiVLMClient()

    def is_available(self) -> bool:
        return self.client.is_available()

    @staticmethod
    def empty_extraction(image_path: str, status: ExtractionStatusEnum, error: str,
                         doc_id: Optional[str] = None) -> DocumentExtraction:
        """Extraction vide (tous les champs à null) : utilisée quand le document n'a pas pu être lu."""
        p = Path(image_path)
        return DocumentExtraction(
            document_id=doc_id or p.stem,
            filename=p.name,
            document_type=DocumentTypeEnum.UNKNOWN,
            fields={
                name: ExtractedField(value=None, raw_value=None, confidence=None,
                                     source_document=p.name, status="NULL")
                for name in FIELD_NAMES
            },
            extraction_status=status,
            error=error,
        )

    def extract_document(
        self,
        image_path: str,
        doc_id: Optional[str] = None,
        declared_type: Optional[DocumentTypeEnum] = None,
        classification_confidence: Optional[float] = None,
    ) -> DocumentExtraction:
        """
        Extrait les champs d'un document. Si `declared_type` est fourni (étape CLASSIFICATION),
        il est conservé comme type du document et tout désaccord avec la réponse d'extraction est signalé.
        """
        p = Path(image_path)
        doc_identifier = doc_id or p.stem

        prompt = EXTRACTION_PROMPT
        if declared_type and declared_type != DocumentTypeEnum.UNKNOWN:
            prompt = f"Type de document déjà identifié par la classification : {declared_type.value}.\n\n" + prompt

        result = self.client.call_vlm(image_path, prompt)

        if "error" in result:
            return self.empty_extraction(image_path, ExtractionStatusEnum.FAILED, str(result["error"]), doc_identifier)

        try:
            extracted_type = DocumentTypeEnum(result.get("document_type", "UNKNOWN"))
        except ValueError:
            extracted_type = DocumentTypeEnum.UNKNOWN

        conflict = False
        if declared_type and declared_type != DocumentTypeEnum.UNKNOWN:
            conflict = extracted_type not in (DocumentTypeEnum.UNKNOWN, declared_type)
            doc_type = declared_type
        else:
            doc_type = extracted_type

        raw_fields = result.get("fields", {}) or {}
        parsed_fields: Dict[str, ExtractedField] = {}

        for field_name in FIELD_NAMES:
            field_data = raw_fields.get(field_name) or {}
            raw_v = field_data.get("raw_value")
            norm_v, raw_v = _normalize(field_name, field_data.get("value"), raw_v)

            parsed_fields[field_name] = ExtractedField(
                value=norm_v,
                raw_value=raw_v,
                confidence=field_data.get("confidence"),
                source_document=p.name,
                status="NULL" if norm_v is None else "VALID",
            )

        extraction = DocumentExtraction(
            document_id=doc_identifier,
            filename=p.name,
            document_type=doc_type,
            fields=parsed_fields,
            raw_response=str(result),
            extraction_status=ExtractionStatusEnum.SUCCESS,
            source=result.get("_source"),
            classification_confidence=classification_confidence,
            classification_conflict=conflict,
            visual_flags=_parse_visual_flags(result.get("signes_alteration")),
        )
        extraction.refresh_confidence()
        return extraction

    # Compatibilité : extraction sans classification préalable
    def classify_and_extract(self, image_path: str, doc_id: Optional[str] = None) -> DocumentExtraction:
        return self.extract_document(image_path, doc_id)

    def targeted_fallback_extract(self, image_path: str, target_fields: list, doc_id: str) -> Dict[str, ExtractedField]:
        """
        Extraction de seconde passe après amélioration d'image sur un sous-ensemble de champs incertains.
        """
        prompt = FALLBACK_PROMPT.replace("<<TARGET_FIELDS>>", ", ".join(target_fields))
        result = self.client.call_vlm(image_path, prompt)

        if "error" in result:
            return {}

        raw_fields = result.get("fields", {}) or {}
        re_extracted: Dict[str, ExtractedField] = {}

        for field_name in target_fields:
            if field_name in raw_fields:
                f_data = raw_fields[field_name] or {}
                raw_v = f_data.get("raw_value")
                norm_v, raw_v = _normalize(field_name, f_data.get("value"), raw_v)

                re_extracted[field_name] = ExtractedField(
                    value=norm_v,
                    raw_value=raw_v,
                    confidence=f_data.get("confidence"),
                    source_document=Path(image_path).name,
                    status="RECOVERED" if norm_v is not None else "NULL",
                )

        return re_extracted
