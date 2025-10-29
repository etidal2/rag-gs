from __future__ import annotations

import json
import random
import time
from typing import Any, Dict, List

from openai import APIError, OpenAI, RateLimitError


def _sleep(attempt: int) -> None:
    time.sleep(min(8.0, 0.5 * (2 ** attempt) + random.random() * 0.25))


def chat_json(
    client: OpenAI,
    model: str,
    messages: List[Dict[str, str]],
    *,
    temperature: float | None = None,
    top_p: float | None = None,
) -> Dict[str, Any]:
    kwargs: Dict[str, Any] = {
        "model": model,
        "messages": messages,
        "response_format": {"type": "json_object"},
    }
    if temperature is not None:
        kwargs["temperature"] = temperature
    if top_p is not None:
        kwargs["top_p"] = top_p

    last_err: Exception | None = None
    for attempt in range(6):
        try:
            completion = client.chat.completions.create(**kwargs)
            content = completion.choices[0].message.content or "{}"
            return json.loads(content)
        except (APIError, RateLimitError, TimeoutError) as e:  # type: ignore[name-defined]
            last_err = e
            if attempt == 5:
                break
            _sleep(attempt)
        except json.JSONDecodeError as e:  # unexpected non-JSON
            last_err = e
            if attempt == 5:
                break
            _sleep(attempt)
    if last_err:
        raise last_err
    return {}
