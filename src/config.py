"""
Configuration centrale du système Agentic AI de souscription d'assurance deux-roues.
Gère les constantes métier, les seuils techniques de vision et les chemins de données.
"""

import os
from pathlib import Path
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass
try:
    import yaml
except ImportError:  # PyYAML absent : les valeurs par défaut ci-dessous s'appliquent
    yaml = None

# Chemins de base du projet
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
INPUT_DIR = DATA_DIR / "input"
GROUND_TRUTH_DIR = DATA_DIR / "ground_truth"
SAMPLES_DIR = DATA_DIR / "samples"
MANIFEST_PATH = DATA_DIR / "inputs_manifest.json"
CONFIG_DIR = BASE_DIR / "config"
PROMPTS_DIR = BASE_DIR / "prompts"
OUTPUT_DIR = BASE_DIR / "outputs"
CLASSIFIED_DIR = OUTPUT_DIR / "classified"   # résultat du traitement en lot : DOSSIER/TYPE/fichier
DEFAULT_BATCH_INPUT = INPUT_DIR / "Deux_Roues"  # dossier d'images traité d'un seul coup par `run.py batch`
DATABASE_DIR = DATA_DIR / "database"
DB_PATH = Path(os.getenv("DIGICLAIM_DB_PATH", DATABASE_DIR / "assurance_souscription.db"))

# Backend API (FastAPI) appelé par l'interface Streamlit
API_HOST = os.getenv("API_HOST", "127.0.0.1")
API_PORT = int(os.getenv("API_PORT", "8000"))
API_BASE_URL = os.getenv("API_BASE_URL", f"http://{API_HOST}:{API_PORT}")

# Traitement en lot : nombre de pièces analysées en parallèle (appels VLM simultanés)
BATCH_MAX_WORKERS = int(os.getenv("BATCH_MAX_WORKERS", "4"))

# Création des dossiers si nécessaire
for p in [DATA_DIR, INPUT_DIR, GROUND_TRUTH_DIR, SAMPLES_DIR, OUTPUT_DIR, DATABASE_DIR]:
    p.mkdir(parents=True, exist_ok=True)

# Configuration API Gemini
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
# Mode hors-ligne de TEST uniquement : renvoie des extractions écrites à la main pour 5 dossiers d'exemple.
# Désactivé par défaut. Les résultats produits dans ce mode ne sont jamais évalués ni présentés comme une performance.
OFFLINE_MOCK_ENV = "DIGICLAIM_OFFLINE_MOCK"
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.1-flash-lite")   # Gemini 3.1 Flash-Lite (rapport, chapitre 3)

# Règles Métier Déterministes (Assurance Deux-Roues) : valeurs lues dans config/business_rules.yaml
_DEFAULT_RULES = {
    "max_allowed_displacement": 125,
    "small_engine_threshold": 49,
    "restriction_year": 2025,
    "restriction_date_field": "date_mec",
    "rejection_types": {"displacement_above_limit": "PUISSANCE_SUP_125",
                        "date_condition_not_met": "PUISSANCE_SUP_49_APRES_2025"},
    "required_documents": {"identity": ["CIN", "PASSPORT"], "vehicle": ["CONTRAT", "FACTURE"], "payment": []},
}


def load_business_rules(path=None) -> dict:
    """Charge les paramètres métier ; toute valeur absente retombe sur la valeur par défaut."""
    path = Path(path or os.getenv("BUSINESS_RULES_FILE") or CONFIG_DIR / "business_rules.yaml")
    rules = dict(_DEFAULT_RULES)
    if yaml is not None and path.exists():
        with open(path, encoding="utf-8") as f:
            rules.update({k: v for k, v in (yaml.safe_load(f) or {}).items() if k in rules})
    rules["rejection_types"] = {**_DEFAULT_RULES["rejection_types"], **(rules["rejection_types"] or {})}
    docs = rules["required_documents"]
    rules["required_documents"] = {k: [str(t).upper() for t in docs.get(k, _DEFAULT_RULES["required_documents"][k])]
                                   for k in ("identity", "vehicle", "payment")}
    from src.validation.schemas import DocumentTypeEnum
    valid = {t.value for t in DocumentTypeEnum} - {"UNKNOWN"}
    unknown = {t for lst in rules["required_documents"].values() for t in lst} - valid
    if unknown:
        raise ValueError(f"required_documents : type(s) de document inconnu(s) {sorted(unknown)} (valides : {sorted(valid)})")
    return rules


BUSINESS_RULES = load_business_rules()
MAX_ALLOWED_DISPLACEMENT = int(BUSINESS_RULES["max_allowed_displacement"])  # > 125 cc -> REJECTED (REJ-01)
SMALL_ENGINE_THRESHOLD = int(BUSINESS_RULES["small_engine_threshold"])      # <= 49 cc -> condition de cylindrée satisfaite
MEC_RESTRICTION_YEAR = int(BUSINESS_RULES["restriction_year"])              # 49 < cc <= 125 et date > 01/01/2025 -> REJECTED (REJ-02)
REJECTION_TYPES = dict(BUSINESS_RULES["rejection_types"])
REQUIRED_IDENTITY_TYPES = tuple(BUSINESS_RULES["required_documents"]["identity"])   # CIN ou PASSPORT
REQUIRED_VEHICLE_TYPES = tuple(BUSINESS_RULES["required_documents"]["vehicle"])     # CONTRAT ou FACTURE
REQUIRED_PAYMENT_TYPES = tuple(BUSINESS_RULES["required_documents"]["payment"])     # QUITTANCE (vide : non exigee)
RESTRICTION_DATE_FIELD = str(BUSINESS_RULES["restriction_date_field"])      # date évaluée : jamais remplacée par une autre

# Message affiché lorsqu'un document reste illisible après prétraitement (rapport, chapitre 6)
UNREADABLE_DOCUMENT_MESSAGE = (
    "Le document reste insuffisamment lisible après prétraitement automatique. "
    "L'extraction fiable des informations n'est pas possible. "
    "Veuillez fournir une image plus nette, correctement cadrée et suffisamment éclairée. "
    "Aucune donnée n'a été estimée ou inventée."
)

# Seuils Qualité Image (OpenCV)
# Note: quality_score est un indicateur technique d'exploitabilité, PAS une probabilité statistique
QUALITY_MIN_SHARPNESS = 60.0       # Variance du Laplacien
QUALITY_MIN_BRIGHTNESS = 40.0      # Moyenne lumineuse min
QUALITY_MAX_BRIGHTNESS = 235.0     # Moyenne lumineuse max (surexposition)
QUALITY_MIN_CONTRAST = 35.0        # Écart-type du contraste
QUALITY_THRESHOLD_ACCEPTABLE = 0.60 # Score composite minimal pour passer directement au VLM

# Paramètres de l'Orchestrateur & Fallback
MAX_FALLBACK_ATTEMPTS = 2          # Limite stricte pour éviter les boucles infinies
# Matching : une similarité approximative ne produit JAMAIS un MATCH. Elle produit au mieux un REVIEW.
NAME_REVIEW_SIMILARITY_MIN = 0.50    # En dessous : MISMATCH ; au-dessus (sans égalité stricte) : REVIEW
CHASSIS_TAIL_MIN_CHARS = 5  # Num?ro partiel : MATCH si les N derniers caract?res sont identiques
CHASSIS_REVIEW_SIMILARITY_MIN = 0.88 # Idem pour le numéro de châssis

# Seuil en dessous duquel un document reste inexploitable même après prétraitement (pas d'appel au VLM)
QUALITY_UNUSABLE_THRESHOLD = 0.20

# Plausibilité des valeurs extraites (étape VALIDATION) : un hors-plage est signalé, jamais corrigé
DISPLACEMENT_PLAUSIBLE_RANGE = (20, 2000)
CIN_LENGTH = 8
PASSPORT_LENGTH_RANGE = (6, 9)   # passeport : 6 à 9 caractères alphanumériques
