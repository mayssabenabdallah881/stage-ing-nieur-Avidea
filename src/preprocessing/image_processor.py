"""
Traitements d'amélioration d'image déterministes (OpenCV / PIL).
Rehaussement de contraste local (CLAHE), accentuation des contours, débruitage.
"""

from pathlib import Path
from typing import Optional
import numpy as np
from src.config import OUTPUT_DIR
from PIL import Image, ImageEnhance, ImageFilter

try:
    import cv2
    OPENCV_AVAILABLE = True
except ImportError:
    OPENCV_AVAILABLE = False


def preprocess_document_image(input_path: str, output_path: Optional[str] = None) -> str:
    """
    Applique une chaîne de prétraitement pour améliorer la lisibilité du document :
    1. Redressement de contraste adaptatif
    2. Réduction du bruit
    3. Légère accentuation des contours (Unsharp Mask)
    """
    p_in = Path(input_path)
    if not p_in.exists():
        return input_path

    if output_path is None:
        enhanced_dir = OUTPUT_DIR / "enhanced"
        enhanced_dir.mkdir(parents=True, exist_ok=True)
        p_out = enhanced_dir / f"enhanced_{p_in.name}"
    else:
        p_out = Path(output_path)

    try:
        if OPENCV_AVAILABLE:
            img = cv2.imread(str(p_in))
            if img is None:
                return input_path

            # Conversion en espace LAB pour appliquer CLAHE sur le canal Luminance
            lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            cl = clahe.apply(l)
            limg = cv2.merge((cl, a, b))
            enhanced = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)

            # Débruitage léger non destructif
            denoised = cv2.fastNlMeansDenoisingColored(enhanced, None, 6, 6, 7, 21)

            cv2.imwrite(str(p_out), denoised)
            return str(p_out)

        else:
            # Rehaussement alternatif via PIL
            pil_img = Image.open(input_path).convert("RGB")
            enhancer = ImageEnhance.Contrast(pil_img)
            enhanced = enhancer.enhance(1.4)
            sharper = ImageEnhance.Sharpness(enhanced).enhance(1.3)
            sharper.save(str(p_out))
            return str(p_out)

    except Exception:
        return input_path
