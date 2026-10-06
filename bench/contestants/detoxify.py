"""Detoxify's multilingual model: XLM-R trained on Jigsaw 2020, seven outputs."""

from __future__ import annotations

from importlib import metadata

from bench.contestants.base import free_gpu, map_labels
from bench.taxonomy import LANGS

# Detoxify-style label names → taxonomy. `toxicity` and `severe_toxicity` stay out: they include profanity.
DETOXIFY_STYLE = {
    "insult": "insult",
    "threat": "threat",
    "identity_attack": "identity_hate",
    "sexual_explicit": "sexual_harassment",
    "obscene": "profanity",
}


class DetoxifyContestant:
    name = "detoxify"
    languages = frozenset(LANGS)
    license = "Apache-2.0"
    notes = (
        "XLM-R entrenado con Jigsaw 2020. Fuera de `toxicity`, sus categorías se aprendieron de etiquetas en "
        "inglés traducidas, así que en español y portugués son débiles. `sexual_explicit` detecta contenido "
        "sexual, no acoso."
    )

    def __init__(self, device: str | None = None, model=None):
        self.device = device
        self._model = model
        self.version = f"multilingual-{metadata.version('detoxify')}"

    def _get_model(self):
        if self._model is None:
            import torch
            from detoxify import Detoxify

            self._model = Detoxify("multilingual", device=self.device or ("cuda" if torch.cuda.is_available() else "cpu"))
        return self._model

    def predict(self, texts: list[str], lang: str) -> list[dict[str, float]]:
        out = self._get_model().predict(texts)
        # A one-text batch can come back as scalars instead of one-item lists.
        columns = {label: list(values) if hasattr(values, "__len__") else [values] for label, values in out.items()}
        return [{label: float(values[i]) for label, values in columns.items()} for i in range(len(texts))]

    def to_scores(self, raw: dict[str, float]) -> dict[str, float]:
        return map_labels(raw, DETOXIFY_STYLE)

    def unload(self) -> None:
        self._model = None
        free_gpu()
