from __future__ import annotations

from pydantic import BaseModel, Field
from typing import List


class S1Rewrite(BaseModel):
    qid: str
    text: str
    text_rewrite: str
    embedding: List[float] = Field(min_length=1024, max_length=1024)
    model: str | None = None
    input_type: str | None = None
    output_dimension: int | None = None
    truncation: bool | None = None

