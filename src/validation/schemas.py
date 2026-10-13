"""
Modèles Pydantic pour la validation stricte et le typage du pipeline de souscription.
Aucune donnée n'est inventée: les valeurs absentes ou illisibles valent impérativement None.
"""

from typing import List, Optional, Dict, Any, Union
from enum import Enum
from pydantic import BaseModel, Field


class DocumentTypeEnum(str, Enum):
    CIN = "CIN"
    PASSPORT = "PASSPORT"
    CONTRAT = "CONTRAT"
    FACTURE = "FACTURE"
    QUITTANCE = "QUITTANCE"
    CARTE_GRISE = "CARTE_GRISE"
    PERMIS = "PERMIS"
    UNKNOWN = "UNKNOWN"


class ExtractionStatusEnum(str, Enum):
    SUCCESS = "SUCCESS"
    PARTIAL = "PARTIAL"
    FALLBACK_APPLIED = "FALLBACK_APPLIED"
    FAILED = "FAILED"
    UNREADABLE = "UNREADABLE"


class MatchingStatusEnum(str, Enum):
    MATCH = "MATCH"
    MISMATCH = "MISMATCH"
    REVIEW = "REVIEW"
    NOT_APPLICABLE = "NOT_APPLICABLE"


class BusinessRuleStatusEnum(str, Enum):
    PASSED = "PASSED"
    FAILED = "FAILED"
    REVIEW_REQUIRED = "REVIEW_REQUIRED"
    DOCUMENT_MANQUANT = "DOCUMENT_MANQUANT"


class DecisionEnum(str, Enum):
    ACCEPTED = "ACCEPTED"
    REJECTED = "REJECTED"
    REVIEW_REQUIRED = "REVIEW_REQUIRED"
    DOCUMENT_MANQUANT = "DOCUMENT_MANQUANT"


class StepEnum(str, Enum):
    RECEIVED = "RECEIVED"
    QUALITY_CHECK = "QUALITY_CHECK"
    PREPROCESSING = "PREPROCESSING"
    CLASSIFICATION = "CLASSIFICATION"
    EXTRACTION_VLM = "EXTRACTION_VLM"
    VALIDATION = "VALIDATION"
    RECONSTRUCTION = "RECONSTRUCTION"
    NORMALIZATION = "NORMALIZATION"
    MATCHING = "MATCHING"
    DATA_QUALITY = "DATA_QUALITY"
    BUSINESS_RULES = "BUSINESS_RULES"
    DECISION = "DECISION"
    PERSISTENCE = "PERSISTENCE"


class StepStatusEnum(str, Enum):
    PENDING = "PENDING"
    RUNNING = "RUNNING"
    SUCCESS = "SUCCESS"
    WARNING = "WARNING"
    FALLBACK = "FALLBACK"
    FAILED = "FAILED"


# 1. Qualité technique du document
class DocumentQuality(BaseModel):
    quality_score: float = Field(
        ...,
        description="Indicateur technique d'exploitabilité (0.0 à 1.0), PAS une probabilité statistique."
    )
    resolution: str = Field(..., description="Résolution WxH de l'image (ex: 1920x1080)")
    sharpness: float = Field(..., description="Mesure de netteté par variance du Laplacien")
    brightness: float = Field(..., description="Luminosité moyenne sur 255")
    contrast: float = Field(..., description="Écart-type des niveaux de gris")
    orientation: int = Field(default=0, description="Angle estimé ou correction de rotation")
    is_readable: bool = Field(..., description="Indique si le document présente un minimum de lisibilité")
    issues: List[str] = Field(default_factory=list, description="Liste d'anomalies détectées (flou, sous-exposition...)")


# 2. Champ extrait individuel
class ExtractedField(BaseModel):
    value: Optional[Union[str, int, float]] = Field(None, description="Valeur normalisée ou None si absent/illisible")
    raw_value: Optional[str] = Field(None, description="Valeur brute extraite du document textuel")
    confidence: Optional[float] = Field(
        None,
        description="Confiance déclarée par le modèle VLM (information indicative du modèle, non calibrée)"
    )
    source_document: str = Field(..., description="Identifiant ou nom du document source")
    status: str = Field(default="VALID", description="VALID, NULL, AMBIGUOUS, CONTRADICTORY")


# 3. Résultat d'extraction d'un document
class DocumentExtraction(BaseModel):
    document_id: str
    filename: str
    document_type: DocumentTypeEnum
    fields: Dict[str, ExtractedField] = Field(default_factory=dict)
    raw_response: Optional[str] = None
    extraction_status: ExtractionStatusEnum = ExtractionStatusEnum.SUCCESS
    attempts_count: int = Field(default=1, description="Nombre de tentatives d'extraction (1 ou 2)")
    fallback_used: bool = False
    fallback_reason: Optional[str] = None
    source: Optional[str] = Field(None, description="Origine : GEMINI:<modèle> ou OFFLINE_MOCK (test uniquement)")
    classification_confidence: Optional[float] = Field(None, description="Confiance déclarée par le modèle (non calibrée)")
    classification_conflict: bool = Field(False, description="Classification et extraction n'ont pas donné le même type")
    confidence: Optional[float] = Field(
        None, description="Confiance moyenne déclarée par le modèle sur les champs renseignés (0 à 1, non calibrée), si disponible"
    )
    visual_flags: List[str] = Field(
        default_factory=list, description="Signes d'altération visibles signalés par le VLM (signal de risque, jamais une preuve)"
    )
    error: Optional[str] = Field(None, description="Message d'erreur si l'extraction a échoué")
    first_attempt_fields: Optional[Dict[str, "ExtractedField"]] = Field(
        None, description="Copie des champs de la tentative 1 (avant fallback), pour mesurer l'apport du fallback"
    )


    def refresh_confidence(self) -> Optional[float]:
        """Recalcule la confiance moyenne déclarée par le modèle sur les champs renseignés (None si aucune)."""
        values = [f.confidence for f in self.fields.values()
                  if f.value is not None and isinstance(f.confidence, (int, float))]
        self.confidence = round(sum(values) / len(values), 4) if values else None
        return self.confidence

    def to_report_dict(self, decision: Optional[str] = None) -> Dict[str, Any]:
        """Structure de sortie décrite en annexe du rapport : type, champs extraits (null si absent), décision."""
        names = ["nom", "prenom", "cin_passeport", "nom_prenom_acheteur",
                 "numero_chassis", "cylindree", "date_vente", "date_mec"]
        out: Dict[str, Any] = {
            "document_type": self.document_type.value,
            "extraction": {n: (self.fields[n].value if n in self.fields else None) for n in names},
            "confidence": self.confidence,
        }
        if decision is not None:
            out["decision"] = decision
        return out


# 4. Résultat de matching croisé
class MatchingResult(BaseModel):
    field: str = Field(..., description="Nom du champ comparé (ex: numero_chassis, cylindree, nom_prenom)")
    document_a: str = Field(..., description="Nom du document A")
    document_b: str = Field(..., description="Nom du document B")
    value_a: Optional[str] = None
    value_b: Optional[str] = None
    status: MatchingStatusEnum = MatchingStatusEnum.MATCH
    similarity: float = Field(default=1.0, description="Score de similarité textuelle ou égalité stricte")
    message: Optional[str] = None


# 5. Résultat d'une règle métier
class BusinessRuleResult(BaseModel):
    rule_name: str
    status: BusinessRuleStatusEnum
    message: str
    code: Optional[str] = Field(None, description="Code de rejet REJ-xx lorsque la règle échoue")
    rejection_type: Optional[str] = Field(None, description="Type de rejet métier (ex: PUISSANCE_SUP_125) lorsque la règle échoue")
    missing_data: bool = Field(False, description="REVIEW_REQUIRED dû à une donnée critique absente (prioritaire sur un rejet)")
    evidence: Dict[str, Any] = Field(default_factory=dict)


# 5 bis. Anomalie (signal de risque FRA-xx)
class Anomaly(BaseModel):
    code: str = Field(..., description="Code FRA-xx")
    name: str = Field(..., description="Nom normalisé (ex: CHASSIS_MISMATCH)")
    severity: str = Field("MEDIUM", description="HIGH (contradiction) ou MEDIUM (ambiguïté / signal à contrôler)")
    message: str
    field: Optional[str] = None
    documents: List[str] = Field(default_factory=list)


# 5 ter. Scores interprétables et variables de feature engineering (rapport, chapitre 9)
class DossierScores(BaseModel):
    """Indicateurs interprétables conçus pour le prototype : ce ne sont PAS des probabilités calibrées.
    None = non calculable (aucune comparaison possible, aucun champ renseigné...), jamais une valeur inventée."""
    missing_field_count: int = 0
    fields_total: int = 0
    extraction_score: Optional[float] = None
    completeness_score: Optional[float] = None
    validity_score: Optional[float] = None
    identity_match_score: Optional[float] = None
    chassis_match_score: Optional[float] = None
    displacement_consistency: Optional[float] = None
    date_consistency: Optional[float] = None
    global_data_confidence: Optional[float] = None
    identity_match: Optional[int] = Field(None, description="1 si toutes les comparaisons d'identité concordent, 0 sinon, None si non comparable")
    chassis_match: Optional[int] = None
    displacement_consistent: Optional[int] = None
    date_consistent: Optional[int] = None
    components_used: List[str] = Field(default_factory=list)


# 5 quater. Reconstruction du dossier (identité / véhicule / autres)
class DossierReconstruction(BaseModel):
    identite: List[str] = Field(default_factory=list, description="Fichiers CIN / passeport")
    vehicule: List[str] = Field(default_factory=list, description="Fichiers facture / contrat / carte grise")
    paiement: List[str] = Field(default_factory=list, description="Fichiers quittance (si exigée)")
    autres: List[str] = Field(default_factory=list, description="Fichiers quittance / permis")
    non_identifies: List[str] = Field(default_factory=list, description="Fichiers de type UNKNOWN ou non lus")
    types_presents: List[str] = Field(default_factory=list)
    pieces_manquantes: List[str] = Field(default_factory=list, description="Catégories sans aucune pièce (information, pas une décision)")
    consolidated: Dict[str, Any] = Field(default_factory=dict, description="Valeur retenue par champ (première valeur non nulle, pièce la plus pertinente d'abord)")
    consolidated_source: Dict[str, str] = Field(default_factory=dict, description="Fichier d'où provient chaque valeur retenue")


# 6. Décision finale du dossier
class DossierDecision(BaseModel):
    decision: DecisionEnum
    reason: str
    rejection_codes: List[str] = Field(default_factory=list, description="Codes REJ-xx lorsque la décision est REJECTED")
    rejection_type: Optional[str] = Field(None, description="Type de rejet lorsque la décision est REJECTED (configurable)")
    explanation: List[str] = Field(default_factory=list, description="Justification détaillée, une ligne par règle / anomalie pertinente")
    rules: List[BusinessRuleResult] = Field(default_factory=list)
    evidence: Dict[str, Any] = Field(default_factory=dict)


# 7. Trace d'exécution d'une étape agentique
class AgentTrace(BaseModel):
    step: StepEnum
    status: StepStatusEnum
    timestamp: str
    duration_ms: int = 0
    message: str
    details: Dict[str, Any] = Field(default_factory=dict)


# 8. Modèle global d'un dossier
class DossierResult(BaseModel):
    dossier_id: str
    created_at: str
    status: str
    documents: List[DocumentExtraction] = Field(default_factory=list)
    quality_checks: Dict[str, DocumentQuality] = Field(default_factory=dict)
    cross_matching: List[MatchingResult] = Field(default_factory=list)
    business_rules: List[BusinessRuleResult] = Field(default_factory=list)
    decision: Optional[DossierDecision] = None
    traces: List[AgentTrace] = Field(default_factory=list)
    data_quality: Dict[str, Any] = Field(default_factory=dict)
    anomalies: List[Anomaly] = Field(default_factory=list)
    scores: Optional[DossierScores] = None
    fallback_triggered: bool = False
