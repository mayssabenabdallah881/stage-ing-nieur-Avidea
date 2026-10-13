"""
Client d'appel au Vision-Language Model Gemini (multimodal).
Gère la transmission des images, la validation JSON et le repli propre en cas d'erreur.
"""

import os
import json
import base64
from pathlib import Path
from typing import Dict, Any, Optional
from PIL import Image

try:
    from google import genai
    from google.genai import types
    HAS_NEW_GENAI = True
except ImportError:
    HAS_NEW_GENAI = False

try:
    import google.generativeai as legacy_genai
    HAS_LEGACY_GENAI = True
except ImportError:
    HAS_LEGACY_GENAI = False

from src.config import GEMINI_API_KEY, GEMINI_MODEL, OFFLINE_MOCK_ENV


class GeminiVLMClient:
    """Wrapper pour interagir avec le modèle multimodal Gemini."""

    def __init__(self, api_key: Optional[str] = None, model_name: Optional[str] = None):
        self.api_key = api_key or GEMINI_API_KEY or os.getenv("GEMINI_API_KEY", "")
        self.model_name = model_name or GEMINI_MODEL
        self.client = None

        if self.api_key:
            if HAS_NEW_GENAI:
                try:
                    self.client = genai.Client(api_key=self.api_key)
                except Exception:
                    self.client = None
            elif HAS_LEGACY_GENAI:
                try:
                    legacy_genai.configure(api_key=self.api_key)
                except Exception:
                    pass

    @staticmethod
    def offline_mock_enabled() -> bool:
        """Le mode hors-ligne de test n'est actif que si DIGICLAIM_OFFLINE_MOCK=1."""
        return os.getenv(OFFLINE_MOCK_ENV, "0") == "1"

    def is_available(self) -> bool:
        """True si un vrai appel Gemini est possible, ou si le mode hors-ligne de test est activé."""
        has_real = bool(self.client) or (HAS_LEGACY_GENAI and bool(self.api_key))
        return has_real or self.offline_mock_enabled()

    def call_vlm(self, image_path: str, prompt: str) -> Dict[str, Any]:
        """
        Envoie l'image et le prompt d'instruction au VLM Gemini.
        Garantit le retour d'un dictionnaire JSON structuré.
        """
        p = Path(image_path)
        if not p.exists():
            return {"error": f"Fichier introuvable: {image_path}"}

        # 1. Utilisation du SDK officiel @google/genai si disponible et configuré
        if self.client and HAS_NEW_GENAI:
            try:
                pil_img = Image.open(image_path)
                response = self.client.models.generate_content(
                    model=self.model_name,
                    contents=[pil_img, prompt],
                    config=types.GenerateContentConfig(
                        response_mime_type="application/json",
                        temperature=0.1,  # Faible température pour éviter les variations
                    )
                )
                text = response.text or "{}"
                return self._tag(self._parse_json(text), f"GEMINI:{self.model_name}")
            except Exception as e:
                # Log et fallback d'erreur
                return {"error": f"Erreur Gemini Client: {str(e)}"}

        elif HAS_LEGACY_GENAI and self.api_key:
            try:
                pil_img = Image.open(image_path)
                model = legacy_genai.GenerativeModel(self.model_name)
                response = model.generate_content(
                    [pil_img, prompt],
                    generation_config={"temperature": 0.1, "response_mime_type": "application/json"}
                )
                text = response.text or "{}"
                return self._tag(self._parse_json(text), f"GEMINI:{self.model_name}")
            except Exception as e:
                return {"error": f"Erreur Gemini Legacy: {str(e)}"}

        # 2. Aucun client Gemini disponible
        if self.offline_mock_enabled():
            return self._tag(self._local_ground_truth_mock(image_path, prompt), "OFFLINE_MOCK")

        return {
            "error": "Aucune clé GEMINI_API_KEY configurée : extraction impossible. "
                     f"(Mode test hors-ligne : {OFFLINE_MOCK_ENV}=1)"
        }

    @staticmethod
    def _tag(result: Dict[str, Any], source: str) -> Dict[str, Any]:
        """Indique l'origine de la réponse (appel Gemini réel ou mode hors-ligne de test)."""
        if isinstance(result, dict):
            result["_source"] = source
        return result

    def _parse_json(self, raw_text: str) -> Dict[str, Any]:
        """Extrait et nettoie le JSON renvoyé par le VLM."""
        text = raw_text.strip()
        if text.startswith("```json"):
            text = text[7:]
        if text.startswith("```"):
            text = text[3:]
        if text.endswith("```"):
            text = text[:-3]
        text = text.strip()

        try:
            return json.loads(text)
        except json.JSONDecodeError:
            # Recherche de la première accolade ouvrante et fermante
            start = text.find("{")
            end = text.rfind("}")
            if start != -1 and end != -1:
                try:
                    return json.loads(text[start:end+1])
                except Exception:
                    pass
            return {"raw_unparsed": text, "error": "Échec décodage JSON"}

    def _local_ground_truth_mock(self, image_path: str, prompt: str) -> Dict[str, Any]:
        """
        MODE TEST UNIQUEMENT (DIGICLAIM_OFFLINE_MOCK=1).
        Renvoie des valeurs écrites à la main pour les 5 premiers dossiers d'exemple, selon le nom du fichier.
        Ce ne sont PAS des extractions du modèle : elles ne doivent jamais servir à évaluer le système.
        """
        fname = Path(image_path).name.lower()

        # Dossier 1 : Abdelkarim Habria
        if "190558" in fname: # CIN Abdelkarim
            return {
                "document_type": "CIN",
                "fields": {
                    "nom": {"raw_value": "الحبريه", "value": "Habria", "confidence": 0.98},
                    "prenom": {"raw_value": "عبد الكريم", "value": "Abdelkarim", "confidence": 0.98},
                    "cin_passeport": {"raw_value": "05826391", "value": "05826391", "confidence": 0.99},
                    "nom_prenom_acheteur": {"raw_value": "عبد الكريم الحبريه", "value": "Abdelkarim Habria", "confidence": 0.98},
                    "cylindree": {"raw_value": None, "value": None, "confidence": None},
                    "date_vente": {"raw_value": None, "value": None, "confidence": None},
                    "date_mec": {"raw_value": None, "value": None, "confidence": None},
                    "numero_chassis": {"raw_value": None, "value": None, "confidence": None}
                }
            }
        elif "190449" in fname: # Facture vente Ariana Jaafar
            return {
                "document_type": "FACTURE",
                "fields": {
                    "nom": {"raw_value": "الحبريه", "value": "Habria", "confidence": 0.94},
                    "prenom": {"raw_value": "عبد الكريم", "value": "Abdelkarim", "confidence": 0.94},
                    "cin_passeport": {"raw_value": "05826391", "value": "05826391", "confidence": 0.96},
                    "nom_prenom_acheteur": {"raw_value": "عبد الكريم الحبريه", "value": "Abdelkarim Habria", "confidence": 0.95},
                    "cylindree": {"raw_value": "49 cc", "value": "49", "confidence": 0.93},
                    "date_vente": {"raw_value": "11/02/2021", "value": "11/02/2021", "confidence": 0.91},
                    "date_mec": {"raw_value": None, "value": None, "confidence": None},
                    "numero_chassis": {"raw_value": "9MWX05047", "value": "9MWX05047", "confidence": 0.92}
                }
            }
        elif "190508" in fname: # Quittance Raoued
            return {
                "document_type": "QUITTANCE",
                "fields": {
                    "nom": {"raw_value": "HABRIA", "value": "Habria", "confidence": 0.95},
                    "prenom": {"raw_value": "ABDELKARIM", "value": "Abdelkarim", "confidence": 0.95},
                    "cin_passeport": {"raw_value": "05826391", "value": "05826391", "confidence": 0.98},
                    "nom_prenom_acheteur": {"raw_value": "ABDELKARIM HABRIA", "value": "Abdelkarim Habria", "confidence": 0.96},
                    "cylindree": {"raw_value": "49", "value": "49", "confidence": 0.97},
                    "date_vente": {"raw_value": "12/01/2026", "value": "12/01/2026", "confidence": 0.95},
                    "date_mec": {"raw_value": None, "value": None, "confidence": None},
                    "numero_chassis": {"raw_value": "9MWX05047", "value": "9MWX05047", "confidence": 0.95}
                }
            }

        # Dossier 2 : Sami Slimani
        elif "8398662890481221" in fname or "177503967154" in fname: # CIN Sami
            return {
                "document_type": "CIN",
                "fields": {
                    "nom": {"raw_value": "سليماني", "value": "Slimani", "confidence": 0.99},
                    "prenom": {"raw_value": "سامي", "value": "Sami", "confidence": 0.99},
                    "cin_passeport": {"raw_value": "06039883", "value": "06039883", "confidence": 0.99},
                    "nom_prenom_acheteur": {"raw_value": "سامي سليماني", "value": "Sami Slimani", "confidence": 0.99},
                    "cylindree": {"raw_value": None, "value": None, "confidence": None},
                    "date_vente": {"raw_value": None, "value": None, "confidence": None},
                    "date_mec": {"raw_value": None, "value": None, "confidence": None},
                    "numero_chassis": {"raw_value": None, "value": None, "confidence": None}
                }
            }
        elif "456072213120522" in fname or "177503968860" in fname: # شهادة بيع سوق الدراجات صفاقس
            return {
                "document_type": "CONTRAT",
                "fields": {
                    "nom": {"raw_value": "سليماني", "value": "Slimani", "confidence": 0.96},
                    "prenom": {"raw_value": "سامي", "value": "Sami", "confidence": 0.96},
                    "cin_passeport": {"raw_value": "06039883", "value": "06039883", "confidence": 0.97},
                    "nom_prenom_acheteur": {"raw_value": "سامي سليماني", "value": "Sami Slimani", "confidence": 0.96},
                    "cylindree": {"raw_value": "49 cm3", "value": "49", "confidence": 0.94},
                    "date_vente": {"raw_value": "20/01/2024", "value": "20/01/2024", "confidence": 0.90},
                    "date_mec": {"raw_value": None, "value": None, "confidence": None},
                    "numero_chassis": {"raw_value": "LBAMPAB3421", "value": "LBAMPAB3421", "confidence": 0.92}
                }
            }
        elif "80117154" in fname or "177503970479" in fname: # Quittance Sfax
            return {
                "document_type": "QUITTANCE",
                "fields": {
                    "nom": {"raw_value": "SLIMANI", "value": "Slimani", "confidence": 0.97},
                    "prenom": {"raw_value": "SEMI", "value": "Sami", "confidence": 0.95},
                    "cin_passeport": {"raw_value": "06039883", "value": "06039883", "confidence": 0.98},
                    "nom_prenom_acheteur": {"raw_value": "SEMI SLIMANI", "value": "Sami Slimani", "confidence": 0.96},
                    "cylindree": {"raw_value": "49", "value": "49", "confidence": 0.98},
                    "date_vente": {"raw_value": "01/04/2026", "value": "01/04/2026", "confidence": 0.94},
                    "date_mec": {"raw_value": None, "value": None, "confidence": None},
                    "numero_chassis": {"raw_value": "LBAMPAB342", "value": "LBAMPAB342", "confidence": 0.96}
                }
            }

        # Dossier 3 : Nizar Laajmi
        elif "image (6)" in fname or "image_6" in fname: # CIN Nizar
            return {
                "document_type": "CIN",
                "fields": {
                    "nom": {"raw_value": "العجمي", "value": "Lâajmi", "confidence": 0.98},
                    "prenom": {"raw_value": "نزار", "value": "Nizar", "confidence": 0.98},
                    "cin_passeport": {"raw_value": "09502361", "value": "09502361", "confidence": 0.99},
                    "nom_prenom_acheteur": {"raw_value": "نزار العجمي", "value": "Nizar Lâajmi", "confidence": 0.98},
                    "cylindree": {"raw_value": None, "value": None, "confidence": None},
                    "date_vente": {"raw_value": None, "value": None, "confidence": None},
                    "date_mec": {"raw_value": None, "value": None, "confidence": None},
                    "numero_chassis": {"raw_value": None, "value": None, "confidence": None}
                }
            }
        elif "image (4)" in fname or "image_4" in fname: # Contrat Hammar Nizar
            return {
                "document_type": "CONTRAT",
                "fields": {
                    "nom": {"raw_value": "العجمي", "value": "Lâajmi", "confidence": 0.95},
                    "prenom": {"raw_value": "نزار", "value": "Nizar", "confidence": 0.95},
                    "cin_passeport": {"raw_value": "09502361", "value": "09502361", "confidence": 0.97},
                    "nom_prenom_acheteur": {"raw_value": "نزار العجمي", "value": "Nizar Lâajmi", "confidence": 0.95},
                    "cylindree": {"raw_value": "49 cm3", "value": "49", "confidence": 0.95},
                    "date_vente": {"raw_value": "2024", "value": "01/01/2024", "confidence": 0.88},
                    "date_mec": {"raw_value": None, "value": None, "confidence": None},
                    "numero_chassis": {"raw_value": "LB412PEFYPC228466", "value": "LB412PEFYPC228466", "confidence": 0.96}
                }
            }

        # Dossier 4 : Walid Araki
        elif "083747" in fname: # CIN Walid
            return {
                "document_type": "CIN",
                "fields": {
                    "nom": {"raw_value": "عراكي", "value": "Araki", "confidence": 0.97},
                    "prenom": {"raw_value": "وليد", "value": "Walid", "confidence": 0.97},
                    "cin_passeport": {"raw_value": "04678976", "value": "04678976", "confidence": 0.99},
                    "nom_prenom_acheteur": {"raw_value": "وليد عراكي", "value": "Walid Araki", "confidence": 0.97},
                    "cylindree": {"raw_value": None, "value": None, "confidence": None},
                    "date_vente": {"raw_value": None, "value": None, "confidence": None},
                    "date_mec": {"raw_value": None, "value": None, "confidence": None},
                    "numero_chassis": {"raw_value": None, "value": None, "confidence": None}
                }
            }
        elif "171703" in fname: # Attestation Affaires Sociales
            return {
                "document_type": "CONTRAT",
                "fields": {
                    "nom": {"raw_value": "عراكي", "value": "Araki", "confidence": 0.94},
                    "prenom": {"raw_value": "وليد", "value": "Walid", "confidence": 0.94},
                    "cin_passeport": {"raw_value": "04678976", "value": "04678976", "confidence": 0.96},
                    "nom_prenom_acheteur": {"raw_value": "وليد عراكي", "value": "Walid Araki", "confidence": 0.95},
                    "cylindree": {"raw_value": "49 cm3", "value": "49", "confidence": 0.95},
                    "date_vente": {"raw_value": "25/10/2023", "value": "25/10/2023", "confidence": 0.92},
                    "date_mec": {"raw_value": None, "value": None, "confidence": None},
                    "numero_chassis": {"raw_value": "AZ2788", "value": "AZ2788", "confidence": 0.94}
                }
            }
        elif "171958" in fname: # Quittance Manouba
            return {
                "document_type": "QUITTANCE",
                "fields": {
                    "nom": {"raw_value": "عراكي", "value": "Araki", "confidence": 0.96},
                    "prenom": {"raw_value": "وليد", "value": "Walid", "confidence": 0.96},
                    "cin_passeport": {"raw_value": "04678976", "value": "04678976", "confidence": 0.98},
                    "nom_prenom_acheteur": {"raw_value": "وليد بن عمر عراكي", "value": "Walid Araki", "confidence": 0.96},
                    "cylindree": {"raw_value": "49", "value": "49", "confidence": 0.98},
                    "date_vente": {"raw_value": "23/03/2026", "value": "23/03/2026", "confidence": 0.95},
                    "date_mec": {"raw_value": None, "value": None, "confidence": None},
                    "numero_chassis": {"raw_value": "AZ2788", "value": "AZ2788", "confidence": 0.97}
                }
            }

        # Dossier 5 : Salima Chiouchiou
        elif "143923" in fname: # CIN Salima
            return {
                "document_type": "CIN",
                "fields": {
                    "nom": {"raw_value": "شيوشيو", "value": "Chiouchiou", "confidence": 0.98},
                    "prenom": {"raw_value": "سليمة", "value": "Salima", "confidence": 0.98},
                    "cin_passeport": {"raw_value": "09146769", "value": "09146769", "confidence": 0.99},
                    "nom_prenom_acheteur": {"raw_value": "سليمة شيوشيو", "value": "Salima Chiouchiou", "confidence": 0.98},
                    "cylindree": {"raw_value": None, "value": None, "confidence": None},
                    "date_vente": {"raw_value": None, "value": None, "confidence": None},
                    "date_mec": {"raw_value": None, "value": None, "confidence": None},
                    "numero_chassis": {"raw_value": None, "value": None, "confidence": None}
                }
            }
        elif "144010" in fname: # Contrat Rymco Like 50
            return {
                "document_type": "CONTRAT",
                "fields": {
                    "nom": {"raw_value": "شيوشيو", "value": "Chiouchiou", "confidence": 0.95},
                    "prenom": {"raw_value": "سليمة", "value": "Salima", "confidence": 0.95},
                    "cin_passeport": {"raw_value": "09146769", "value": "09146769", "confidence": 0.96},
                    "nom_prenom_acheteur": {"raw_value": "سليمة شيوشيو", "value": "Salima Chiouchiou", "confidence": 0.95},
                    "cylindree": {"raw_value": "49 سم3", "value": "49", "confidence": 0.93},
                    "date_vente": {"raw_value": "01/04/2026", "value": "01/04/2026", "confidence": 0.96},
                    "date_mec": {"raw_value": None, "value": None, "confidence": None},
                    "numero_chassis": {"raw_value": "LC2D10000C1100897", "value": "LC2D10000C1100897", "confidence": 0.94}
                }
            }

        # Document générique
        return {
            "document_type": "UNKNOWN",
            "fields": {
                "nom": {"raw_value": None, "value": None, "confidence": None},
                "prenom": {"raw_value": None, "value": None, "confidence": None},
                "cin_passeport": {"raw_value": None, "value": None, "confidence": None},
                "nom_prenom_acheteur": {"raw_value": None, "value": None, "confidence": None},
                "cylindree": {"raw_value": None, "value": None, "confidence": None},
                "date_vente": {"raw_value": None, "value": None, "confidence": None},
                "date_mec": {"raw_value": None, "value": None, "confidence": None},
                "numero_chassis": {"raw_value": None, "value": None, "confidence": None}
            }
        }
