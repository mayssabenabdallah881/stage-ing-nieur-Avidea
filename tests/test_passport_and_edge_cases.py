"""Passeport (type PASSPORT, champ cin_passeport) et cas limites listés au chapitre 13 du rapport."""

from conftest import FakeVLMClient, field, make_black_image, make_readable_image, pair
from src.agent.orchestrator import SouscriptionAgentOrchestrator
from src.config import UNREADABLE_DOCUMENT_MESSAGE
from src.database.repository import SouscriptionRepository
from src.evaluation.evaluator import DossierEvaluator
from src.evaluation.ground_truth import GroundTruthManager
from src.matching.matcher import CrossDocumentMatcher
from src.validation.normalization import normalize_cin_passeport
from src.validation.schemas import DocumentExtraction, ExtractedField, DocumentTypeEnum as T, MatchingStatusEnum as M
from src.validation.validators import validate_extraction
from src.vision.extraction import VLMExtractionService


def _orch(tmp_path, client):
    return SouscriptionAgentOrchestrator(
        repository=SouscriptionRepository(db_path=tmp_path / "t.db"), vlm_service=VLMExtractionService(client),
        evaluator=DossierEvaluator(GroundTruthManager(tmp_path / "absent.json")))


def _doc(name, t, **v):
    return DocumentExtraction(document_id=name, filename=name, document_type=t,
                              fields={k: ExtractedField(value=x, source_document=name) for k, x in v.items()})


# --- passeport ---

def test_normalization_of_cin_and_passport():
    assert normalize_cin_passeport("5826391")[0] == "05826391"
    assert normalize_cin_passeport("p 123-456")[0] == "P123456"
    assert normalize_cin_passeport(None) == (None, None) and normalize_cin_passeport("  ")[0] is None


def test_validation_depends_on_document_type():
    assert validate_extraction(_doc("p", T.PASSPORT, cin_passeport="P1234567")) == []
    assert validate_extraction(_doc("p", T.PASSPORT, cin_passeport="05826391"))        # un passeport sans lettres : douteux
    assert validate_extraction(_doc("c", T.CIN, cin_passeport="P1234567"))              # une CIN avec lettres : douteuse
    assert validate_extraction(_doc("k", T.CONTRAT, cin_passeport="P1234567")) == []    # un contrat peut citer l'un ou l'autre


def test_cin_is_never_compared_with_a_passport_number():
    a, b = _doc("a", T.PASSPORT, cin_passeport="P1234567"), _doc("b", T.CONTRAT, cin_passeport="05826391")
    assert [r for r in CrossDocumentMatcher.compare_documents([a, b]) if r.field == "cin_passeport"] == []
    c = _doc("c", T.CONTRAT, cin_passeport="P1234567")
    res = [r for r in CrossDocumentMatcher.compare_documents([a, c]) if r.field == "cin_passeport"]
    assert [r.status for r in res] == [M.MATCH]


def test_passport_dossier_is_processed(tmp_path):
    client = FakeVLMClient(classification={"document_type": "PASSPORT"}, extraction={"document_type": "PASSPORT", "fields": {
        "cin_passeport": field("p 1234567"), "nom_prenom_acheteur": field("John Doe"), "numero_chassis": field("ABC12345"), "cylindree": field("49")}})
    state = _orch(tmp_path, client).process_dossier([make_readable_image(tmp_path / "p.jpg")], "DOS-PP")
    ext = next(iter(state.extractions.values()))
    assert ext.document_type == T.PASSPORT and ext.fields["cin_passeport"].value == "P1234567"
    assert state.reconstruction.identite == [ext.filename]


# --- cas limites du chapitre 13 ---

def test_unreadable_image_gives_the_report_message_and_no_data(tmp_path):
    client = FakeVLMClient()
    state = _orch(tmp_path, client).process_dossier([make_black_image(tmp_path / "noir.jpg")], "DOS-BLACK")
    assert client.calls == [] and state.status == "REVIEW_REQUIRED"
    assert UNREADABLE_DOCUMENT_MESSAGE in state.decision.reason
    assert "Aucune donnée n'a été estimée ou inventée." in state.decision.reason
    ext = next(iter(state.extractions.values()))
    assert ext.error == UNREADABLE_DOCUMENT_MESSAGE and all(f.value is None for f in ext.fields.values())


def test_unknown_document_requires_review_with_fra_08(tmp_path):
    client = FakeVLMClient(classification={"document_type": "UNKNOWN"}, extraction={"document_type": "UNKNOWN", "fields": {}})
    state = _orch(tmp_path, client).process_dossier([make_readable_image(tmp_path / "x.jpg")], "DOS-UNK")
    assert state.status == "REVIEW_REQUIRED" and "FRA-08" in [a.code for a in state.anomalies]


def test_all_null_extraction_is_incomplete_and_scores_stay_honest(tmp_path):
    client = FakeVLMClient(classification={"document_type": "CIN"}, extraction={"document_type": "CIN", "fields": {}})
    state = _orch(tmp_path, client).process_dossier([make_readable_image(tmp_path / "x.jpg")], "DOS-NULL")
    assert state.status == "DOCUMENT_MANQUANT" and state.scores.extraction_score is None   # une CIN seule : pas de contrat/facture
    assert state.scores.missing_field_count == 8


def test_different_chassis_and_inconsistent_identity_go_to_review(tmp_path):
    class TwoDocs(FakeVLMClient):
        def call_vlm(self, image_path, prompt):
            name = str(image_path)
            self.calls.append("x")
            if "détermine son type exact" in prompt:
                return {"document_type": "CIN" if "cin" in name else "CONTRAT", "_source": "GEMINI:fake"}
            if "cin" in name:
                f = {"cin_passeport": field("05826391"), "nom": field("Habria"), "prenom": field("Abdelkarim")}
                return {"document_type": "CIN", "fields": f, "_source": "GEMINI:fake"}
            f = {"cin_passeport": field("05826391"), "nom": field("Slimani"), "prenom": field("Sami"),
                 "numero_chassis": field("AAA11111"), "cylindree": field("49")}
            return {"document_type": "CONTRAT", "fields": f, "_source": "GEMINI:fake"}
    state = _orch(tmp_path, TwoDocs()).process_dossier(
        [make_readable_image(tmp_path / "cin.jpg", seed=1), make_readable_image(tmp_path / "contrat.jpg", seed=2)], "DOS-ID")
    assert state.status == "REVIEW_REQUIRED" and "FRA-02" in [a.code for a in state.anomalies]
    assert state.scores.identity_match == 0


def test_displacement_above_125_is_rejected_with_code(tmp_path):
    client = FakeVLMClient(classification={"document_type": "CONTRAT"}, extraction={"document_type": "CONTRAT", "fields": {
        "cin_passeport": field("05826391"), "nom_prenom_acheteur": field("A B"), "numero_chassis": field("9MWX05047"), "cylindree": field("150")}})
    state = _orch(tmp_path, client).process_dossier(pair(tmp_path), "DOS-BIG")
    assert state.status == "REJECTED" and state.decision.rejection_codes == ["REJ-01"]


def test_missing_date_mec_for_125cc_goes_to_review(tmp_path):
    client = FakeVLMClient(classification={"document_type": "CONTRAT"}, extraction={"document_type": "CONTRAT", "fields": {
        "cin_passeport": field("05826391"), "nom_prenom_acheteur": field("A B"), "numero_chassis": field("9MWX05047"),
        "cylindree": field("125"), "date_vente": field("01/04/2026")}})
    state = _orch(tmp_path, client).process_dossier(pair(tmp_path), "DOS-MEC")
    assert state.status == "REVIEW_REQUIRED" and "date_mec" in state.decision.reason


def test_report_output_structure():
    ext = _doc("f.jpg", T.FACTURE, numero_chassis="ABC12345", cylindree=125)
    out = ext.to_report_dict("REVIEW_REQUIRED")
    assert out["document_type"] == "FACTURE" and out["decision"] == "REVIEW_REQUIRED"
    assert set(out["extraction"]) == {"nom", "prenom", "cin_passeport", "nom_prenom_acheteur",
                                      "numero_chassis", "cylindree", "date_vente", "date_mec"}
    assert out["extraction"]["nom"] is None and out["extraction"]["cylindree"] == 125
