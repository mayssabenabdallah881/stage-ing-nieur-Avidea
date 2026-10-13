"""
Traitement en lot d'un dossier d'images (ex. data/input/Deux_Roues).

Toutes les pièces du dossier sont analysées EN MÊME TEMPS (pool de threads : contrôle qualité,
classification VLM du type de pièce, lecture de l'identité), puis regroupées par dossier client
et rangées dans une arborescence :

    outputs/classified/<nom du lot>/
        DOS-001-HABRIA/CIN/...      <- une pièce par sous-dossier de type
        DOS-001-HABRIA/QUITTANCE/...
        NON_AFFECTES/PERMIS/...     <- pièce identifiée mais sans autre pièce correspondante
        A_VERIFIER/...              <- illisible, non classée, VLM indisponible ou en erreur
        rapport_classification.json / .csv

Principes (identiques au reste du projet) :
- aucune valeur n'est inventée : sans lecture fiable, la pièce va dans A_VERIFIER ;
- le regroupement n'utilise que des ÉGALITÉS (même CIN, ou même nom+prénom sans CIN contradictoire) ;
  une simple ressemblance ne fusionne jamais deux dossiers, elle est seulement signalée dans le rapport ;
- les fichiers d'origine ne sont jamais déplacés ni modifiés : le rangement se fait par copie.
"""

import csv
import difflib
import json
import logging
import re
import shutil
import unicodedata
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Callable, Dict, List, Optional

from pydantic import BaseModel, Field

from src.config import (
    BATCH_MAX_WORKERS, CLASSIFIED_DIR, NAME_REVIEW_SIMILARITY_MIN, QUALITY_UNUSABLE_THRESHOLD,
)
from src.documents.classifier import DocumentClassifier
from src.documents.loader import DocumentLoader
from src.preprocessing.image_processor import preprocess_document_image
from src.preprocessing.quality import analyze_document_quality
from src.validation.schemas import DocumentTypeEnum
from src.vision.extraction import VLMExtractionService

logger = logging.getLogger(__name__)

UNASSIGNED_DIR = "NON_AFFECTES"
TO_CHECK_DIR = "A_VERIFIER"
IGNORED_DIR_NAMES = {"__MACOSX", ".git", "__pycache__"}


class BatchItem(BaseModel):
    """Résultat de l'analyse d'une pièce (une image, ou une page de PDF)."""
    path: str                                   # fichier analysé (image, ou PNG d'une page de PDF)
    original_path: str                          # fichier d'origine du lot
    filename: str
    source_pdf: Optional[str] = None
    multi_page_pdf: bool = False
    document_type: DocumentTypeEnum = DocumentTypeEnum.UNKNOWN
    classification_confidence: Optional[float] = None   # déclarée par le modèle, non calibrée
    quality_score: Optional[float] = None
    cin_passeport: Optional[str] = None
    nom: Optional[str] = None
    prenom: Optional[str] = None
    nom_prenom_acheteur: Optional[str] = None
    status: str = "CLASSIFIED"                  # CLASSIFIED | UNREADABLE | FAILED | UNSUPPORTED
    reason: Optional[str] = None
    vlm_source: Optional[str] = None
    dossier_id: Optional[str] = None            # DOS-xxx, NON_AFFECTES ou A_VERIFIER
    destination: Optional[str] = None           # chemin relatif du fichier rangé


class BatchReport(BaseModel):
    input_dir: str
    output_dir: str
    total_files: int = 0
    total_items: int = 0
    dossiers: Dict[str, List[str]] = Field(default_factory=dict)
    unassigned: List[str] = Field(default_factory=list)
    to_check: List[str] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)
    offline_mock_used: bool = False
    items: List[BatchItem] = Field(default_factory=list)


# --------------------------------------------------------------------------- utilitaires

def _strip_accents(text: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFKD", text) if not unicodedata.combining(c))


def _name_tokens(*parts: Optional[str]) -> frozenset:
    text = " ".join(p for p in parts if p)
    cleaned = re.sub(r"[^\w\s]|_", " ", _strip_accents(text).casefold())
    return frozenset(t for t in cleaned.split() if t)


def _identity_key(item: BatchItem) -> frozenset:
    """Jeu de mots du nom complet ; vide si l'identité n'est pas lisible."""
    if item.nom and item.prenom:
        return _name_tokens(item.nom, item.prenom)
    return _name_tokens(item.nom_prenom_acheteur or item.nom or item.prenom)


def _id_kind(value: str) -> str:
    """CIN (chiffres seuls) ou passeport (avec lettres) : seuls des identifiants de même nature sont comparables."""
    return "CIN" if re.sub(r"\W", "", value).isdigit() else "PASSEPORT"


def _safe_label(text: str) -> str:
    ascii_txt = _strip_accents(text).upper()
    return re.sub(r"[^A-Z0-9]+", "-", ascii_txt).strip("-") or "INCONNU"


def _unique_target(folder: Path, filename: str) -> Path:
    target = folder / filename
    n = 1
    while target.exists():
        target = folder / f"{Path(filename).stem}_{n}{Path(filename).suffix}"
        n += 1
    return target


def list_input_files(input_dir: Path) -> List[Path]:
    """Toutes les pièces recevables du dossier (récursif), triées par nom pour un résultat reproductible."""
    files = []
    for p in sorted(input_dir.rglob("*")):
        if not p.is_file() or p.name.startswith("."):
            continue
        if any(part in IGNORED_DIR_NAMES for part in p.relative_to(input_dir).parts):
            continue
        if p.suffix.lower() in DocumentLoader.SUPPORTED_EXTENSIONS:
            files.append(p)
    return files


class _UnionFind:
    def __init__(self, n: int):
        self.parent = list(range(n))

    def find(self, i: int) -> int:
        while self.parent[i] != i:
            self.parent[i] = self.parent[self.parent[i]]
            i = self.parent[i]
        return i

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[max(ra, rb)] = min(ra, rb)


# --------------------------------------------------------------------------- traitement en lot

class BatchIntake:
    """Ingestion simultanée d'un dossier d'images puis classement par dossier client et par type de pièce."""

    def __init__(
        self,
        vlm_service: Optional[VLMExtractionService] = None,
        classifier: Optional[DocumentClassifier] = None,
        max_workers: int = BATCH_MAX_WORKERS,
    ):
        self.vlm_service = vlm_service or VLMExtractionService()
        self.classifier = classifier or DocumentClassifier(self.vlm_service.client)
        self.max_workers = max(1, max_workers)

    # ---- analyse d'une pièce (exécutée dans un thread) ----
    def _analyze_one(self, doc: Dict, original: Path, multi_page: bool, vlm_ready: bool) -> BatchItem:
        item = BatchItem(
            path=doc["path"], original_path=str(original), filename=doc["filename"],
            source_pdf=doc.get("source_pdf"), multi_page_pdf=multi_page,
        )
        try:
            quality = analyze_document_quality(item.path)
            effective = item.path
            if not quality.is_readable:
                effective = preprocess_document_image(item.path)
                quality = analyze_document_quality(effective)
            item.quality_score = round(quality.quality_score, 3)

            if quality.quality_score < QUALITY_UNUSABLE_THRESHOLD:
                item.status, item.reason = "UNREADABLE", "Document inexploitable (qualité insuffisante après prétraitement)."
                return item
            if not vlm_ready:
                item.status = "FAILED"
                item.reason = "VLM indisponible (GEMINI_API_KEY absente) : pièce non classée, rien n'est inventé."
                return item

            classification = self.classifier.classify(effective)
            item.vlm_source = classification.source
            if classification.error:
                item.status, item.reason = "FAILED", f"Classification impossible : {classification.error}"
                return item
            item.document_type = classification.document_type
            item.classification_confidence = classification.confidence

            extraction = self.vlm_service.extract_document(
                effective, doc_id=doc["stem"], declared_type=classification.document_type,
                classification_confidence=classification.confidence,
            )
            item.vlm_source = extraction.source or item.vlm_source
            if extraction.error:
                item.reason = f"Identité non lue : {extraction.error}"
                return item

            def val(name: str) -> Optional[str]:
                f = extraction.fields.get(name)
                return str(f.value).strip() if f is not None and f.value not in (None, "") else None

            item.cin_passeport, item.nom, item.prenom, item.nom_prenom_acheteur = (
                val("cin_passeport"), val("nom"), val("prenom"), val("nom_prenom_acheteur"))
            if extraction.classification_conflict:
                item.reason = "Type ambigu : la classification et l'extraction ne concordent pas."
        except Exception as e:  # une pièce défaillante ne doit jamais arrêter le lot
            logger.exception("Analyse impossible : %s", doc.get("filename"))
            item.status, item.reason = "FAILED", f"Erreur pendant l'analyse : {e}"
        return item

    # ---- regroupement par dossier client ----
    @staticmethod
    def _group(items: List[BatchItem]) -> Dict[str, List[BatchItem]]:
        """Groupes de pièces d'un même client. Égalités seulement : même CIN, ou même nom complet sans CIN contradictoire."""
        candidates = [it for it in items if it.status == "CLASSIFIED" and it.document_type != DocumentTypeEnum.UNKNOWN
                      and (it.cin_passeport or _identity_key(it))]
        uf = _UnionFind(len(candidates))
        for i in range(len(candidates)):
            for j in range(i + 1, len(candidates)):
                a, b = candidates[i], candidates[j]
                if a.cin_passeport and b.cin_passeport and _id_kind(a.cin_passeport) == _id_kind(b.cin_passeport):
                    if a.cin_passeport == b.cin_passeport:
                        uf.union(i, j)
                    continue                                    # identifiants différents : jamais fusionnés
                ka, kb = _identity_key(a), _identity_key(b)
                if ka and ka == kb:
                    uf.union(i, j)
        groups: Dict[int, List[BatchItem]] = {}
        for idx, it in enumerate(candidates):
            groups.setdefault(uf.find(idx), []).append(it)
        return groups

    def _assign_dossiers(self, items: List[BatchItem], report: BatchReport) -> None:
        groups = self._group(items)
        ordered = sorted(groups.values(), key=lambda g: min(it.filename for it in g))
        dossier_no = 0
        group_keys: Dict[str, frozenset] = {}
        for members in ordered:
            if len(members) < 2:
                continue                                        # pièce isolée -> NON_AFFECTES
            dossier_no += 1
            holder = next((m for m in members if m.document_type in (DocumentTypeEnum.CIN, DocumentTypeEnum.PASSPORT) and m.nom), None) \
                or next((m for m in members if m.nom), None)
            label = _safe_label(holder.nom if holder else (members[0].nom_prenom_acheteur or "INCONNU"))
            dossier_id = f"DOS-{dossier_no:03d}-{label}"
            for m in members:
                m.dossier_id = dossier_id
            report.dossiers[dossier_id] = [m.filename for m in members]
            group_keys[dossier_id] = _identity_key(holder or members[0])
            if len({m.cin_passeport for m in members if m.cin_passeport}) > 1:
                report.warnings.append(f"{dossier_id} : numéros CIN/passeport différents entre pièces, à contrôler.")

        for it in items:
            if it.dossier_id:
                continue
            if it.status == "CLASSIFIED" and it.document_type != DocumentTypeEnum.UNKNOWN:
                it.dossier_id = UNASSIGNED_DIR
                it.reason = it.reason or "Aucune autre pièce correspondante (même CIN/passeport ou même nom) dans le lot."
                report.unassigned.append(it.filename)
                self._warn_near_names(it, group_keys, report)
            else:
                it.dossier_id = TO_CHECK_DIR
                it.reason = it.reason or "Type de pièce non déterminé."
                report.to_check.append(it.filename)

    @staticmethod
    def _warn_near_names(item: BatchItem, group_keys: Dict[str, frozenset], report: BatchReport) -> None:
        """Ressemblance de nom avec un dossier existant : signalée, jamais fusionnée automatiquement."""
        key = " ".join(sorted(_identity_key(item)))
        if not key:
            return
        for dossier_id, gk in group_keys.items():
            ratio = difflib.SequenceMatcher(None, key, " ".join(sorted(gk))).ratio()
            if ratio >= NAME_REVIEW_SIMILARITY_MIN:
                report.warnings.append(
                    f"{item.filename} ressemble au dossier {dossier_id} (similarité {ratio:.2f}) : rapprochement à valider à la main."
                )

    # ---- rangement ----
    @staticmethod
    def _copy_items(items: List[BatchItem], out_dir: Path) -> None:
        for it in items:
            if it.dossier_id == TO_CHECK_DIR:
                folder = out_dir / TO_CHECK_DIR
            else:
                type_name = it.document_type.value if it.document_type != DocumentTypeEnum.UNKNOWN else "UNKNOWN"
                folder = out_dir / it.dossier_id / type_name
            folder.mkdir(parents=True, exist_ok=True)
            # PDF à une page ou image : on copie le fichier d'origine ; PDF multi-pages : l'image de la page
            source = Path(it.path) if it.multi_page_pdf else Path(it.original_path)
            target = _unique_target(folder, source.name)
            shutil.copy2(source, target)
            it.destination = str(target.relative_to(out_dir))

    @staticmethod
    def _write_reports(report: BatchReport, out_dir: Path) -> None:
        (out_dir / "rapport_classification.json").write_text(
            report.model_dump_json(indent=2), encoding="utf-8")
        columns = ["fichier", "source_pdf", "dossier", "type", "statut", "confiance_modele", "qualite",
                   "cin_passeport", "nom", "prenom", "motif", "destination"]
        with open(out_dir / "rapport_classification.csv", "w", newline="", encoding="utf-8-sig") as f:
            w = csv.writer(f, delimiter=";")
            w.writerow(columns)
            for it in report.items:
                w.writerow([it.filename, it.source_pdf or "", it.dossier_id, it.document_type.value, it.status,
                            it.classification_confidence if it.classification_confidence is not None else "",
                            it.quality_score if it.quality_score is not None else "",
                            it.cin_passeport or "", it.nom or "", it.prenom or "", it.reason or "", it.destination or ""])

    # ---- point d'entrée ----
    def process_folder(
        self,
        input_dir: Path,
        output_dir: Optional[Path] = None,
        progress: Optional[Callable[[BatchItem, int, int], None]] = None,
    ) -> BatchReport:
        input_dir = Path(input_dir)
        if not input_dir.is_dir():
            raise FileNotFoundError(f"Dossier d'entrée introuvable : {input_dir}")
        out_dir = Path(output_dir) if output_dir else CLASSIFIED_DIR / _safe_label(input_dir.name).title().replace("-", "_")
        if out_dir.exists():
            # le rangement est un résultat dérivé, régénéré à chaque lot ; on ne supprime qu'un sous-dossier de classified/
            if CLASSIFIED_DIR.resolve() not in out_dir.resolve().parents:
                raise ValueError(f"Dossier de sortie existant hors de {CLASSIFIED_DIR} : {out_dir} (choisissez un dossier vide).")
            shutil.rmtree(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        files = list_input_files(input_dir)
        report = BatchReport(input_dir=str(input_dir), output_dir=str(out_dir), total_files=len(files))

        # Chargement : un PDF de N pages donne N pièces
        tasks = []
        items: List[BatchItem] = []
        for f in files:
            try:
                docs = DocumentLoader.load_documents([str(f)])
            except Exception as e:
                items.append(BatchItem(path=str(f), original_path=str(f), filename=f.name, status="FAILED",
                                       reason=f"Fichier non lisible (PDF/image) : {e}"))
                continue
            for d in docs:
                tasks.append((d, f, len(docs) > 1))

        vlm_ready = self.vlm_service.is_available()
        if not vlm_ready:
            report.warnings.append("VLM indisponible (GEMINI_API_KEY absente) : aucune pièce n'a pu être classée.")

        # Analyse simultanée de toutes les pièces
        total = len(tasks) + len(items)
        done = len(items)
        with ThreadPoolExecutor(max_workers=self.max_workers) as pool:
            futures = [pool.submit(self._analyze_one, d, f, multi, vlm_ready) for d, f, multi in tasks]
            for fut in as_completed(futures):
                item = fut.result()
                items.append(item)
                done += 1
                if progress:
                    progress(item, done, total)

        items.sort(key=lambda it: (it.filename, it.path))
        report.total_items = len(items)
        report.offline_mock_used = any(it.vlm_source == "OFFLINE_MOCK" for it in items)
        if report.offline_mock_used:
            report.warnings.append("MODE TEST HORS-LIGNE : valeurs écrites à la main, non produites par le modèle.")

        self._assign_dossiers(items, report)
        self._copy_items(items, out_dir)
        report.items = items
        self._write_reports(report, out_dir)
        return report
