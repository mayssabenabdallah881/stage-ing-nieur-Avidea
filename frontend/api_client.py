"""Client HTTP de l'interface Streamlit vers le backend API (aucune logique métier ici)."""

import os
from typing import Any, Dict, List, Optional, Tuple

import requests

API_BASE_URL = os.getenv("API_BASE_URL", f"http://{os.getenv('API_HOST', '127.0.0.1')}:{os.getenv('API_PORT', '8000')}")


class ApiUnavailable(Exception):
    """Le backend ne répond pas (non lancé ?)."""


class ApiError(Exception):
    """Le backend a répondu par une erreur."""


def _request(method: str, path: str, timeout: int = 30, **kwargs) -> Any:
    try:
        resp = requests.request(method, f"{API_BASE_URL}{path}", timeout=timeout, **kwargs)
    except requests.ConnectionError as e:
        raise ApiUnavailable(f"API injoignable ({API_BASE_URL}). Lancez-la avec : python run.py api") from e
    except requests.Timeout as e:
        raise ApiError("Délai dépassé : l'analyse prend trop de temps.") from e
    if resp.status_code >= 400:
        try:
            detail = resp.json().get("detail", resp.text)
        except ValueError:
            detail = resp.text
        raise ApiError(f"{resp.status_code} : {detail}")
    return resp.json()


def health() -> Dict[str, Any]:
    return _request("GET", "/health", timeout=5)


def stats() -> Dict[str, Any]:
    return _request("GET", "/stats")


def samples() -> List[Dict[str, Any]]:
    return _request("GET", "/samples")


def list_dossiers() -> List[Dict[str, Any]]:
    return _request("GET", "/dossiers")


def get_dossier(dossier_id: str) -> Dict[str, Any]:
    return _request("GET", f"/dossiers/{dossier_id}")


def analyze_uploads(files: List[Tuple[str, bytes]]) -> Dict[str, Any]:
    payload = [("files", (name, content)) for name, content in files]
    return _request("POST", "/dossiers", timeout=600, files=payload)


def analyze_sample(dossier_id: str) -> Dict[str, Any]:
    return _request("POST", f"/dossiers/samples/{dossier_id}", timeout=600)
