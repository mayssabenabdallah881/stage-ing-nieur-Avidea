"""
Module d'analyse de la qualité technique des documents d'assurance.
Calcul de netteté, luminosité, contraste et lisibilité via OpenCV / PIL.
Le quality_score est un indicateur technique d'exploitabilité, PAS une probabilité.
"""

import os
from pathlib import Path
from typing import Tuple, List, Optional
import numpy as np

try:
    import cv2
    OPENCV_AVAILABLE = True
except ImportError:
    OPENCV_AVAILABLE = False

from PIL import Image
from src.validation.schemas import DocumentQuality
from src.config import (
    QUALITY_MIN_SHARPNESS, QUALITY_MIN_BRIGHTNESS,
    QUALITY_MAX_BRIGHTNESS, QUALITY_MIN_CONTRAST,
    QUALITY_THRESHOLD_ACCEPTABLE
)


def analyze_document_quality(image_path: str) -> DocumentQuality:
    """
    Évalue la qualité technique d'une image de document.
    Détecte le flou (Laplacien), le contraste, la luminosité et la résolution.
    """
    p = Path(image_path)
    if not p.exists():
        return DocumentQuality(
            quality_score=0.0,
            resolution="0x0",
            sharpness=0.0,
            brightness=0.0,
            contrast=0.0,
            orientation=0,
            is_readable=False,
            issues=["Fichier introuvable"]
        )

    issues: List[str] = []

    try:
        # Chargement image
        pil_img = Image.open(image_path).convert("RGB")
        width, height = pil_img.size
        resolution_str = f"{width}x{height}"

        # Vérification taille minimale
        if width < 600 or height < 600:
            issues.append(f"Résolution faible ({resolution_str}), risque de perte de caractères")

        img_np = np.array(pil_img)
        # Niveaux de gris
        if OPENCV_AVAILABLE:
            gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
            # Netteté par variance du Laplacien
            sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        else:
            # Fallback numpy pur si OpenCV non disponible
            gray = np.mean(img_np, axis=2).astype(np.uint8)
            gy, gx = np.gradient(gray.astype(float))
            sharpness = float(np.var(gx) + np.var(gy))

        brightness = float(np.mean(gray))
        contrast = float(np.std(gray))

        # Évaluation des seuils
        if sharpness < QUALITY_MIN_SHARPNESS:
            issues.append(f"Image floue (netteté: {sharpness:.1f} < seuil: {QUALITY_MIN_SHARPNESS})")

        if brightness < QUALITY_MIN_BRIGHTNESS:
            issues.append(f"Image sous-exposée (luminosité: {brightness:.1f} < seuil: {QUALITY_MIN_BRIGHTNESS})")
        elif brightness > QUALITY_MAX_BRIGHTNESS:
            issues.append(f"Image surexposée / reflet flash (luminosité: {brightness:.1f})")

        if contrast < QUALITY_MIN_CONTRAST:
            issues.append(f"Contraste insuffisant (écart-type: {contrast:.1f} < seuil: {QUALITY_MIN_CONTRAST})")

        # Calcul du score technique d'exploitabilité (0.0 à 1.0)
        # Normalisation pondérée
        norm_sharpness = min(1.0, sharpness / 200.0)
        norm_contrast = min(1.0, contrast / 70.0)
        norm_brightness = 1.0 - (abs(brightness - 128.0) / 128.0)
        norm_res = min(1.0, (width * height) / (1200 * 900))

        quality_score = float(
            (norm_sharpness * 0.40) +
            (norm_contrast * 0.25) +
            (norm_brightness * 0.20) +
            (norm_res * 0.15)
        )
        quality_score = max(0.0, min(1.0, round(quality_score, 3)))

        is_readable = quality_score >= QUALITY_THRESHOLD_ACCEPTABLE and sharpness >= (QUALITY_MIN_SHARPNESS * 0.6)

        return DocumentQuality(
            quality_score=quality_score,
            resolution=resolution_str,
            sharpness=round(sharpness, 2),
            brightness=round(brightness, 2),
            contrast=round(contrast, 2),
            orientation=0,
            is_readable=is_readable,
            issues=issues
        )

    except Exception as e:
        return DocumentQuality(
            quality_score=0.1,
            resolution="inconnue",
            sharpness=0.0,
            brightness=0.0,
            contrast=0.0,
            orientation=0,
            is_readable=False,
            issues=[f"Erreur d'analyse d'image: {str(e)}"]
        )
