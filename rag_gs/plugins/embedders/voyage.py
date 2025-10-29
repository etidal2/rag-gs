from __future__ import annotations

from typing import Dict, List

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

_session = requests.Session()
_retry = Retry(total=5, backoff_factor=0.5, status_forcelist=(429, 500, 502, 503, 504))
_session.mount("https://", HTTPAdapter(max_retries=_retry))


def request_embeddings(
    api_key: str,
    model: str,
    input_type: str,
    output_dimension: int,
    inputs: List[str],
    *,
    truncation: bool = True,
    timeout_s: int = 60,
) -> Dict[int, List[float]]:
    if not inputs:
        return {}
    url = "https://api.voyageai.com/v1/embeddings"
    payload: Dict[str, object] = {
        "model": model,
        "input_type": input_type,
        "output_dimension": output_dimension,
        "input": inputs,
    }
    if truncation:
        payload["truncation"] = True
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    resp = _session.post(url, json=payload, headers=headers, timeout=timeout_s)
    resp.raise_for_status()
    data = resp.json()
    out: Dict[int, List[float]] = {}
    for idx, row in enumerate(data.get("data", [])):
        vec = row.get("embedding")
        if isinstance(vec, list):
            out[idx] = [float(v) for v in vec]
    return out
