"""Lightweight evaluation: run sample questions and rate the answers."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Any

import pandas as pd

SAMPLE_QUESTIONS: list[str] = [
    "What is the main topic of this document?",
    "Summarise the key findings in three sentences.",
    "What methods, data or sources does the document rely on?",
    "What limitations, risks or open questions are mentioned?",
    "What does the document recommend or conclude?",
]


@dataclass
class Rating:
    question: str
    answer: str
    model: str
    accuracy: int
    grounded: bool
    citations: int
    latency_s: float
    note: str = ""
    at: str = field(default_factory=lambda: datetime.now().isoformat(timespec="seconds"))


def record_rating(store: list[dict[str, Any]], rating: Rating) -> None:
    store.append(asdict(rating))


def ratings_frame(store: list[dict[str, Any]]) -> pd.DataFrame:
    if not store:
        return pd.DataFrame(
            columns=["at", "question", "model", "accuracy", "grounded", "citations", "latency_s", "note"]
        )
    frame = pd.DataFrame(store)
    return frame[["at", "question", "model", "accuracy", "grounded", "citations", "latency_s", "note"]]


def summarise(store: list[dict[str, Any]]) -> dict[str, Any]:
    """Headline numbers for the evaluation tab."""
    if not store:
        return {"rated": 0, "avg_accuracy": 0.0, "grounded_rate": 0.0, "avg_latency": 0.0}
    frame = pd.DataFrame(store)
    return {
        "rated": len(frame),
        "avg_accuracy": round(float(frame["accuracy"].mean()), 2),
        "grounded_rate": round(float(frame["grounded"].mean()) * 100, 1),
        "avg_latency": round(float(frame["latency_s"].mean()), 2),
    }
