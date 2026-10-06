"""Any Hugging Face multi-label classifier with sigmoid outputs, loaded at a pinned revision."""

from __future__ import annotations

from bench.contestants.base import free_gpu, map_labels
from bench.contestants.detoxify import DETOXIFY_STYLE
from bench.taxonomy import LANGS

HORIZON_REVISION = "dbf12a9915275078e580707bc2396b07585da31e"  # checked on 2026-10-06 (Task 1)
HORIZON_NOTES = (
    "mmBERT-base ajustado con Civil Comments traducido por un LLM. Lo publicó una cuenta creada en "
    "septiembre de 2026, sin trayectoria, y no sabemos con qué datos exactos se entrenó: si vio ToLD-Br, "
    "OLID-BR o HateCheck, sus números en esas fuentes están inflados. Se carga solo en safetensors y sin "
    "código remoto."
)


class HFContestant:
    def __init__(self, name: str, model_id: str, revision: str, label_map: dict[str, str], notes: str, license: str,
                 languages=LANGS, max_length: int = 256, device: str | None = None, loader=None):
        self.name = name
        self.model_id = model_id
        self.revision = revision
        self.label_map = label_map
        self.notes = notes
        self.license = license
        self.languages = frozenset(languages)
        self.max_length = max_length
        self.device = device
        self.version = revision[:12]
        self._loader = loader or self._from_hub
        self._loaded = None

    def _from_hub(self):
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(self.model_id, revision=self.revision)
        # Safetensors only and no remote code: the weights come from a publisher nobody has vetted.
        model = AutoModelForSequenceClassification.from_pretrained(self.model_id, revision=self.revision, use_safetensors=True)
        return tokenizer, model

    def _get(self):
        if self._loaded is None:
            import torch

            tokenizer, model = self._loader()
            device = self.device or ("cuda" if torch.cuda.is_available() else "cpu")
            self._loaded = (tokenizer, model.to(device).eval(), device)
        return self._loaded

    def predict(self, texts: list[str], lang: str) -> list[dict[str, float]]:
        import torch

        tokenizer, model, device = self._get()
        encoded = tokenizer(texts, padding=True, truncation=True, max_length=self.max_length, return_tensors="pt")
        encoded = {k: v.to(device) for k, v in encoded.items()}
        with torch.inference_mode():
            probabilities = torch.sigmoid(model(**encoded).logits.float()).cpu().tolist()
        id2label = model.config.id2label
        return [{id2label[i]: p for i, p in enumerate(row)} for row in probabilities]

    def to_scores(self, raw: dict[str, float]) -> dict[str, float]:
        return map_labels(raw, self.label_map)

    def unload(self) -> None:
        self._loaded = None
        free_gpu()


def horizon_mmbert(device: str | None = None) -> HFContestant:
    return HFContestant("horizon-mmbert", "Horizon-Labs/multilingual-toxicity-base", HORIZON_REVISION,
                        DETOXIFY_STYLE, HORIZON_NOTES, "Apache-2.0", device=device)
