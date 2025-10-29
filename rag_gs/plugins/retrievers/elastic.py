from __future__ import annotations

from typing import Any, Dict

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

_session = requests.Session()
_retry = Retry(total=5, backoff_factor=0.5, status_forcelist=(429, 500, 502, 503, 504))
_session.mount("http://", HTTPAdapter(max_retries=_retry))
_session.mount("https://", HTTPAdapter(max_retries=_retry))


def search(es_url: str, index: str, body: Dict[str, Any], *, timeout_s: int = 60) -> Dict[str, Any]:
    resp = _session.post(f"{es_url.rstrip('/')}/{index}/_search", json=body, timeout=timeout_s)
    resp.raise_for_status()
    return resp.json()
