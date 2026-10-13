"""Tests du contrôle qualité et du prétraitement."""

from pathlib import Path

from conftest import make_black_image, make_readable_image
from src.config import OUTPUT_DIR
from src.preprocessing.quality import analyze_document_quality
from src.preprocessing.image_processor import preprocess_document_image


def test_sharp_contrasted_image_is_readable(tmp_path):
    q = analyze_document_quality(make_readable_image(tmp_path / "a.jpg"))
    assert q.is_readable is True
    assert 0.0 <= q.quality_score <= 1.0
    assert q.resolution == "1000x800"


def test_black_image_is_not_readable_and_reports_issues(tmp_path):
    q = analyze_document_quality(make_black_image(tmp_path / "b.jpg"))
    assert q.is_readable is False
    assert q.quality_score < 0.2
    assert any("sous-exposée" in i or "floue" in i for i in q.issues)


def test_missing_file_is_reported(tmp_path):
    q = analyze_document_quality(str(tmp_path / "absent.jpg"))
    assert q.is_readable is False and "Fichier introuvable" in q.issues


def test_preprocessing_does_not_write_next_to_source(tmp_path):
    src = make_readable_image(tmp_path / "scan.jpg", size=(400, 300))
    out = Path(preprocess_document_image(src))
    assert out.exists()
    assert out.parent == OUTPUT_DIR / "enhanced"
    assert not (tmp_path / "enhanced_scan.jpg").exists()
