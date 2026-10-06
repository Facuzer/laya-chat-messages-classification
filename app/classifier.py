from __future__ import annotations

import logging
import os
import threading
import time
from dataclasses import dataclass
from pathlib import Path

from app.checkpoints import ActiveCheckpoint, questions_match

log = logging.getLogger(__name__)

STATE_KEY = "message"

# Two-option `choice` with semantic labels rather than `noul`: Laya's `noul` can follow its
# true/false labels instead of the text. Both questions share one forward pass.
QUESTIONS = {
    "insult": {
        "type": "choice",
        "instructions": "Does `message` insult or threaten someone?",
        "criteria": {
            "insult": "insults, name-calling, slurs or threats aimed at a person",
            "clean": "no insult or threat aimed at a person; includes sarcasm, swearing at no one and friendly banter",
        },
    },
    "sentiment": {
        "type": "choice",
        "instructions": "What is the overall sentiment of `message`?",
        "criteria": {
            "positive": "favorable, happy, grateful, satisfied",
            "negative": "unhappy, angry, disappointed, critical",
            "neutral": "factual, or no clear sentiment",
        },
    },
}


@dataclass(frozen=True)
class Verdict:
    label: str
    # Laya's `answer_confidence`: the calibrated probability of `label`. Its `confidence` field
    # is normalised entropy, which is not a probability and is not meant for thresholds.
    confidence: float
    probabilities: dict[str, float]


@dataclass(frozen=True)
class Classification:
    flagged: bool
    insult: Verdict
    sentiment: Verdict
    model: str
    latency_ms: int


def _verdict(answer: dict) -> Verdict:
    return Verdict(label=answer["choice"], confidence=answer["answer_confidence"], probabilities=answer["probabilities"])


def to_classification(result: dict, threshold: float, latency_ms: int) -> Classification:
    answers = result["answers"]
    insult = _verdict(answers["insult"])
    return Classification(
        flagged=insult.label == "insult" and insult.confidence >= threshold,
        insult=insult,
        sentiment=_verdict(answers["sentiment"]),
        model=result["routing"]["model"],
        latency_ms=latency_ms,
    )


class LayaClassifier:
    def __init__(self, router, threshold: float, checkpoint: dict | None = None):
        self.router = router
        self.threshold = threshold
        # The fine-tuned checkpoint answering for `multilingual` (see app.checkpoints), or None for the base model.
        self.checkpoint = checkpoint
        # One inference at a time: the endpoint runs in a thread pool and every request shares the model.
        self._lock = threading.Lock()

    def classify(self, text: str) -> Classification:
        with self._lock:
            start = time.perf_counter()
            result = self.router.predict({STATE_KEY: text}, QUESTIONS)
            latency_ms = round((time.perf_counter() - start) * 1000)
        return to_classification(result, self.threshold, latency_ms)


def build_router(checkpoint: Path | None = None, device: str | None = None):
    """A Router with no checkpoint loaded yet; routing decisions work without loading one.

    `checkpoint` is a local fine-tuned directory that answers in place of `multilingual`.
    `device` overrides LAYA_DEVICE (the benchmark picks it per run).
    """
    # Imported here so that importing this module (e.g. in tests) does not pull in torch.
    from laya import Router

    models = {"multilingual": str(checkpoint)} if checkpoint else None
    # Laya names a Latin-script language only from function words exclusive to it, so short or
    # unaccented Spanish often stays unidentified. Unidentified text takes `default`: sending it
    # to the multilingual checkpoint keeps Spanish off the English one, while text Laya does
    # identify as English still goes there.
    return Router(models=models, device=device or os.environ.get("LAYA_DEVICE") or None, default="multilingual")


def load_classifier(active: ActiveCheckpoint | None = None) -> LayaClassifier:
    checkpoint = (active or ActiveCheckpoint()).get()
    if checkpoint and not questions_match(checkpoint["path"], QUESTIONS):
        log.warning(
            "The active checkpoint %s was trained with different QUESTIONS; its answers may be off. "
            "Fine-tune again or run scripts/activate_model.py --base.",
            checkpoint["path"],
        )
    router = build_router(checkpoint["path"] if checkpoint else None)
    router.preload(["english", "multilingual"])
    return LayaClassifier(router, threshold=float(os.environ.get("FLAG_THRESHOLD", "0.7")), checkpoint=checkpoint)
