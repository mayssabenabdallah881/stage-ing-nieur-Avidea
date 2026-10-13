"""Traitement en lot : analyse simultanée, regroupement par dossier client, rangement par type de pièce."""

from conftest import FakeVLMClient, field, make_black_image, make_readable_image
from src.documents.batch import BatchIntake, list_input_files
from src.vision.extraction import VLMExtractionService


class ScriptedVLM(FakeVLMClient):
    """Répond selon le NOM du fichier : {nom_fichier: (type, cin, nom, prenom)}."""

    def __init__(self, script, available=True):
        super().__init__(available=available)
        self.script = script

    def call_vlm(self, image_path, prompt):
        from pathlib import Path
        name = Path(image_path).name
        doc_type, cin, nom, prenom = self.script[name]
        if "détermine son type exact" in prompt:
            self.calls.append("classification")
            return {"document_type": doc_type, "confidence": 0.9, "_source": "GEMINI:fake"}
        self.calls.append("extraction")
        fields = {
            "cin_passeport": field(cin), "nom": field(nom), "prenom": field(prenom),
            "nom_prenom_acheteur": field(f"{prenom} {nom}" if nom and prenom else None),
        }
        return {"document_type": doc_type, "fields": fields, "_source": "GEMINI:fake"}


def _intake(client):
    service = VLMExtractionService(client)
    return BatchIntake(vlm_service=service, max_workers=4)


def _make_inputs(folder, names):
    folder.mkdir()
    for i, n in enumerate(names):
        make_readable_image(folder / n, seed=i)


def test_groups_by_cin_and_sorts_by_type(tmp_path):
    names = ["a_cin.jpg", "b_contrat.jpg", "c_quittance.jpg", "d_cin.jpg", "e_contrat.jpg"]
    _make_inputs(tmp_path / "in", names)
    client = ScriptedVLM({
        "a_cin.jpg": ("CIN", "05826391", "Habria", "Abdelkarim"),
        "b_contrat.jpg": ("CONTRAT", "05826391", "Habria", "Abdelkarim"),
        # nom mal lu sur la quittance : le CIN identique suffit à la rattacher au même dossier
        "c_quittance.jpg": ("QUITTANCE", "05826391", "Habrya", "Abdelkarim"),
        "d_cin.jpg": ("CIN", "06039883", "Slimani", "Sami"),
        "e_contrat.jpg": ("CONTRAT", "06039883", "Slimani", "Sami"),
    })
    out = tmp_path / "out"
    report = _intake(client).process_folder(tmp_path / "in", out)

    assert len(report.dossiers) == 2
    habria = next(d for d in report.dossiers if d.endswith("HABRIA"))
    assert sorted(p.name for p in (out / habria).rglob("*.jpg")) == ["a_cin.jpg", "b_contrat.jpg", "c_quittance.jpg"]
    assert (out / habria / "CIN" / "a_cin.jpg").exists()
    assert (out / habria / "QUITTANCE" / "c_quittance.jpg").exists()
    assert (out / "rapport_classification.json").exists() and (out / "rapport_classification.csv").exists()
    # les originaux ne sont ni déplacés ni modifiés
    assert sorted(p.name for p in (tmp_path / "in").iterdir()) == sorted(names)


def test_different_cin_never_merged_even_with_same_name(tmp_path):
    _make_inputs(tmp_path / "in", ["x.jpg", "y.jpg"])
    client = ScriptedVLM({
        "x.jpg": ("CIN", "11111111", "Ben Ali", "Karim"),
        "y.jpg": ("CONTRAT", "22222222", "Ben Ali", "Karim"),
    })
    report = _intake(client).process_folder(tmp_path / "in", tmp_path / "out")
    assert report.dossiers == {}
    assert sorted(report.unassigned) == ["x.jpg", "y.jpg"]


def test_same_full_name_without_cin_groups(tmp_path):
    _make_inputs(tmp_path / "in", ["x.jpg", "y.jpg"])
    client = ScriptedVLM({
        "x.jpg": ("CIN", "11111111", "Ben Ali", "Karim"),
        "y.jpg": ("QUITTANCE", None, "BEN ALI", "karim"),
    })
    report = _intake(client).process_folder(tmp_path / "in", tmp_path / "out")
    assert len(report.dossiers) == 1


def test_lone_document_goes_to_unassigned_and_similar_name_is_only_flagged(tmp_path):
    _make_inputs(tmp_path / "in", ["cin.jpg", "contrat.jpg", "permis.jpg"])
    client = ScriptedVLM({
        "cin.jpg": ("CIN", "05826391", "Habria", "Abdelkarim"),
        "contrat.jpg": ("CONTRAT", "05826391", "Habria", "Abdelkarim"),
        "permis.jpg": ("PERMIS", None, "Habrya", "Abdelkarim"),   # nom proche : jamais fusionné
    })
    out = tmp_path / "out"
    report = _intake(client).process_folder(tmp_path / "in", out)
    assert report.unassigned == ["permis.jpg"]
    assert (out / "NON_AFFECTES" / "PERMIS" / "permis.jpg").exists()
    assert any("permis.jpg ressemble au dossier" in w for w in report.warnings)


def test_without_vlm_nothing_is_invented_and_no_call_is_made(tmp_path):
    _make_inputs(tmp_path / "in", ["a.jpg", "b.jpg"])
    client = ScriptedVLM({}, available=False)
    out = tmp_path / "out"
    report = _intake(client).process_folder(tmp_path / "in", out)
    assert client.calls == []
    assert report.dossiers == {} and sorted(report.to_check) == ["a.jpg", "b.jpg"]
    assert sorted(p.name for p in (out / "A_VERIFIER").iterdir()) == ["a.jpg", "b.jpg"]
    assert any("VLM indisponible" in w for w in report.warnings)


def test_unreadable_image_is_not_sent_to_vlm(tmp_path):
    (tmp_path / "in").mkdir()
    make_black_image(tmp_path / "in" / "noire.jpg")
    client = ScriptedVLM({})
    report = _intake(client).process_folder(tmp_path / "in", tmp_path / "out")
    assert client.calls == []
    assert report.to_check == ["noire.jpg"]
    assert report.items[0].status == "UNREADABLE"


def test_ignores_macos_and_hidden_files_and_scans_subfolders(tmp_path):
    inp = tmp_path / "in"
    (inp / "__MACOSX").mkdir(parents=True)
    (inp / "sous_dossier").mkdir()
    make_readable_image(inp / "__MACOSX" / "._a.jpg")
    make_readable_image(inp / ".cache.jpg")
    make_readable_image(inp / "sous_dossier" / "b.jpg")
    (inp / "notes.txt").write_text("x")
    assert [p.name for p in list_input_files(inp)] == ["b.jpg"]


def test_same_filename_in_two_subfolders_is_not_overwritten(tmp_path):
    inp = tmp_path / "in"
    (inp / "p1").mkdir(parents=True)
    (inp / "p2").mkdir()
    make_readable_image(inp / "p1" / "scan.jpg", seed=1)
    make_readable_image(inp / "p2" / "scan.jpg", seed=2)
    client = ScriptedVLM({"scan.jpg": ("CIN", "05826391", "Habria", "Abdelkarim")})
    out = tmp_path / "out"
    report = _intake(client).process_folder(inp, out)
    assert len(list(out.rglob("scan*.jpg"))) == 2
    assert report.total_items == 2
