"""
Module de chargement et d'ingestion documentaire (Document Loader).
Valide l'existence, le format des fichiers et prépare les descripteurs pour l'agent.
Les PDF sont convertis page par page en PNG pour passer par la même chaîne que les images.
"""

from pathlib import Path
from typing import List, Dict, Any

from src.config import OUTPUT_DIR

PDF_RENDER_DPI = 200


def render_pdf_pages(pdf_path: Path, dpi: int = PDF_RENDER_DPI) -> List[Path]:
    """Convertit chaque page d'un PDF en image PNG et retourne les chemins."""
    import pymupdf

    pages_dir = OUTPUT_DIR / "pages"
    pages_dir.mkdir(parents=True, exist_ok=True)
    out: List[Path] = []
    with pymupdf.open(str(pdf_path)) as pdf:
        for i, page in enumerate(pdf, start=1):
            target = pages_dir / f"{pdf_path.stem}_p{i}.png"
            page.get_pixmap(dpi=dpi).save(str(target))
            out.append(target)
    return out


class DocumentLoader:
    """Chargeur de fichiers sécurisé pour l'orchestrateur."""

    SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".pdf", ".tif", ".tiff"}

    @classmethod
    def load_documents(cls, paths: List[str]) -> List[Dict[str, Any]]:
        """
        Valide une liste de chemins de fichiers et retourne les métadonnées de base.
        Un PDF de N pages produit N documents (une image par page).
        """
        documents = []
        for raw_path in paths:
            p = Path(raw_path)
            if not p.exists():
                continue

            ext = p.suffix.lower()
            if ext not in cls.SUPPORTED_EXTENSIONS:
                continue

            if ext == ".pdf":
                for page_path in render_pdf_pages(p):
                    documents.append(cls._describe(page_path, source=p.name))
                continue

            documents.append(cls._describe(p))

        return documents

    @staticmethod
    def _describe(p: Path, source: str = "") -> Dict[str, Any]:
        return {
            "path": str(p.resolve()),
            "filename": p.name,
            "stem": p.stem,
            "extension": p.suffix.lower(),
            "size_bytes": p.stat().st_size,
            "source_pdf": source or None,
        }
