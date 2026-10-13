"""
Prompts directifs pour l'analyse Vision-Language Model (Gemini VLM).

Les textes sont versionnés dans le dossier `prompts/` à la racine du projet (un fichier par prompt) :
    classification.txt     -> CLASSIFIER_PROMPT
    extraction.txt         -> EXTRACTION_PROMPT (complété par anomaly_detection.txt)
    anomaly_detection.txt  -> consigne de signalement des signes d'altération visibles
    fallback.txt           -> FALLBACK_PROMPT (seconde passe ciblée)

Règle absolue commune : ne jamais inventer, ne jamais déduire, renvoyer null si absent ou illisible.
"""

from src.config import PROMPTS_DIR


def load_prompt(name: str) -> str:
    """Lit un prompt du dossier prompts/ (UTF-8)."""
    return (PROMPTS_DIR / f"{name}.txt").read_text(encoding="utf-8")


CLASSIFIER_PROMPT = load_prompt("classification")
ANOMALY_DETECTION_PROMPT = load_prompt("anomaly_detection")
EXTRACTION_PROMPT = load_prompt("extraction").rstrip("\n") + "\n" + ANOMALY_DETECTION_PROMPT
FALLBACK_PROMPT = load_prompt("fallback")
