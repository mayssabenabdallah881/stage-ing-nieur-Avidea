"""
Configuration Pytest.
Active le mode hors-ligne de TEST (extractions écrites à la main) pour que la suite s'exécute sans clé API.
Les tests qui vérifient le comportement SANS ce mode le désactivent explicitement.
"""

import os
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

os.environ["DIGICLAIM_OFFLINE_MOCK"] = "1"
os.environ["BUSINESS_RULES_FILE"] = str(Path(__file__).resolve().parent / "business_rules_test.yaml")
os.environ["GEMINI_API_KEY"] = ""                     # les tests ignorent la cle reelle du .env
os.environ["GEMINI_MODEL"] = "gemini-3.1-flash-lite"   # et son modele personnalise
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def make_readable_image(path: Path, size=(1000, 800), seed: int = 0) -> str:
    """Image bruitée : netteté et contraste élevés, donc jugée lisible par le contrôle qualité."""
    rng = np.random.default_rng(seed)
    arr = rng.integers(0, 256, size=(size[1], size[0], 3), dtype=np.uint8)
    Image.fromarray(arr).save(path)
    return str(path)


def make_black_image(path: Path, size=(800, 600)) -> str:
    Image.new("RGB", size, (0, 0, 0)).save(path)
    return str(path)


class FakeVLMClient:
    """Client VLM de test : répond selon le type de prompt et compte les appels."""

    def __init__(self, classification=None, extraction=None, fallback=None, available=True):
        self.classification = classification or {"document_type": "CIN", "confidence": 0.9}
        self.extraction = extraction
        self.fallback = fallback
        self.available = available
        self.calls = []

    def is_available(self):
        return self.available

    def call_vlm(self, image_path, prompt):
        if Path(image_path).name.startswith("cin_doc") and "seconde passe" not in prompt:
            # pièce d'identité factice : fournit la pièce obligatoire (identité) sans modifier les valeurs du contrat
            if "détermine son type exact" in prompt:
                self.calls.append("classification")
                return {"document_type": "CIN", "confidence": 0.9, "_source": "GEMINI:fake"}
            self.calls.append("extraction")
            return {"document_type": "CIN", "_source": "GEMINI:fake",
                    "fields": {"cin_passeport": {"raw_value": "05826391", "value": "05826391", "confidence": 0.9}}}
        if "seconde passe" in prompt:
            kind = "fallback"
            result = self.fallback if self.fallback is not None else {"fields": {}}
        elif "détermine son type exact" in prompt:
            kind = "classification"
            result = self.classification
        else:
            kind = "extraction"
            result = self.extraction if self.extraction is not None else {"document_type": "CIN", "fields": {}}
        self.calls.append(kind)
        if isinstance(result, dict) and "error" in result:
            return result
        out = dict(result)
        out["_source"] = "GEMINI:fake"
        return out


def pair(tmp_path, name="c.jpg", seed=0):
    """Dossier minimal valide : une pièce d'identité (factice, cf. FakeVLMClient) + la pièce `name` (contrat)."""
    return [make_readable_image(tmp_path / "cin_doc.jpg", seed=seed + 100), make_readable_image(tmp_path / name, seed=seed)]


def field(raw, value=None, conf=0.9):
    return {"raw_value": raw, "value": raw if value is None else value, "confidence": conf}


@pytest.fixture
def img_dir(tmp_path):
    return tmp_path
