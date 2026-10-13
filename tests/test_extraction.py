"""Tests de l'extraction VLM : normalisation, erreurs, mode hors-ligne explicite, fallback."""

import pytest

from conftest import FakeVLMClient, field, make_readable_image
from src.vision import gemini_client
from src.vision.extraction import VLMExtractionService
from src.documents.classifier import DocumentClassifier
from src.validation.schemas import DocumentTypeEnum, ExtractionStatusEnum


def test_extraction_normalizes_values(tmp_path):
    img = make_readable_image(tmp_path / "doc.jpg")
    client = FakeVLMClient(extraction={
        "document_type": "CONTRAT",
        "fields": {
            "cin_passeport": field("5826391"),                    # 7 chiffres -> 05826391
            "cylindree": field("49 cc", "49 cc"),       # -> 49
            "numero_chassis": field("9mwx 0504-7"),      # -> 9MWX05047
            "nom": field(None),
        },
    })
    ext = VLMExtractionService(client).extract_document(img)
    assert ext.fields["cin_passeport"].value == "05826391"
    assert ext.fields["cylindree"].value == 49
    assert ext.fields["numero_chassis"].value == "9MWX05047"
    assert ext.fields["nom"].value is None and ext.fields["nom"].status == "NULL"
    assert ext.fields["cin_passeport"].raw_value == "5826391"          # la valeur brute est conservée


def test_missing_fields_stay_null(tmp_path):
    img = make_readable_image(tmp_path / "doc.jpg")
    client = FakeVLMClient(extraction={"document_type": "CIN", "fields": {}})
    ext = VLMExtractionService(client).extract_document(img)
    assert all(f.value is None for f in ext.fields.values())


def test_vlm_error_gives_failed_extraction_without_values(tmp_path):
    img = make_readable_image(tmp_path / "doc.jpg")
    client = FakeVLMClient(extraction={"error": "quota dépassé"})
    ext = VLMExtractionService(client).extract_document(img)
    assert ext.extraction_status == ExtractionStatusEnum.FAILED
    assert ext.error == "quota dépassé"
    assert all(f.value is None for f in ext.fields.values())


def test_no_api_key_and_no_offline_mode_returns_error_not_data(tmp_path, monkeypatch):
    monkeypatch.delenv("DIGICLAIM_OFFLINE_MOCK", raising=False)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.setattr(gemini_client, "GEMINI_API_KEY", "")
    client = gemini_client.GeminiVLMClient(api_key="")
    img = make_readable_image(tmp_path / "20260331_190558.jpeg")   # nom d'un dossier d'exemple

    assert client.is_available() is False
    result = client.call_vlm(img, "prompt")
    assert "error" in result and "fields" not in result

    ext = VLMExtractionService(client).extract_document(img)
    assert ext.extraction_status == ExtractionStatusEnum.FAILED
    assert all(f.value is None for f in ext.fields.values())


def test_offline_mode_is_tagged_as_mock(tmp_path):
    client = gemini_client.GeminiVLMClient(api_key="")
    img = make_readable_image(tmp_path / "20260331_190558.jpeg")
    ext = VLMExtractionService(client).extract_document(img)
    assert ext.source == "OFFLINE_MOCK"


def test_classification_conflict_is_flagged(tmp_path):
    img = make_readable_image(tmp_path / "doc.jpg")
    client = FakeVLMClient(extraction={"document_type": "QUITTANCE", "fields": {}})
    ext = VLMExtractionService(client).extract_document(img, declared_type=DocumentTypeEnum.CONTRAT)
    assert ext.classification_conflict is True
    assert ext.document_type == DocumentTypeEnum.CONTRAT       # la classification fait référence


def test_classifier_reads_type_and_handles_errors(tmp_path):
    img = make_readable_image(tmp_path / "doc.jpg")
    ok = DocumentClassifier(FakeVLMClient(classification={"document_type": "QUITTANCE", "confidence": 0.8})).classify(img)
    assert ok.document_type == DocumentTypeEnum.QUITTANCE and ok.confidence == 0.8

    bad = DocumentClassifier(FakeVLMClient(classification={"document_type": "N_IMPORTE_QUOI"})).classify(img)
    assert bad.document_type == DocumentTypeEnum.UNKNOWN

    err = DocumentClassifier(FakeVLMClient(classification={"error": "panne"})).classify(img)
    assert err.error == "panne" and err.document_type == DocumentTypeEnum.UNKNOWN


def test_targeted_fallback_prompt_is_formatted_without_error(tmp_path):
    """Régression : le prompt de fallback contient des accolades JSON qui faisaient planter str.format()."""
    img = make_readable_image(tmp_path / "doc.jpg")
    client = FakeVLMClient(fallback={"fields": {"cin_passeport": field("05826391")}})
    fields = VLMExtractionService(client).targeted_fallback_extract(img, ["cin_passeport", "numero_chassis"], "doc")
    assert fields["cin_passeport"].value == "05826391" and fields["cin_passeport"].status == "RECOVERED"
    assert "numero_chassis" not in fields
