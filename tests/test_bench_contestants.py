import re
from types import SimpleNamespace

import pytest
import torch

from bench.contestants import DEFAULT_CONTESTANTS, FACTORIES, OPEN_LICENSES, build_contestants
from bench.contestants.base import effective_scores
from bench.contestants.detoxify import DETOXIFY_STYLE, DetoxifyContestant
from bench.contestants.hf import HORIZON_REVISION, HFContestant
from bench.taxonomy import FLAG

DETOX_OUT = {"toxicity": [0.99, 0.1], "severe_toxicity": [0.5, 0.0], "obscene": [0.9, 0.0], "identity_attack": [0.2, 0.0],
             "insult": [0.7, 0.05], "threat": [0.1, 0.0], "sexual_explicit": [0.3, 0.0]}


class FakeDetoxify:
    def __init__(self, out):
        self.out = out

    def predict(self, texts):
        return self.out


def test_detoxify_returns_one_raw_dict_per_text_and_flag_ignores_toxicity():
    contestant = DetoxifyContestant(model=FakeDetoxify(DETOX_OUT))

    raw = contestant.predict(["sos un pelotudo", "buena mano"], "es")

    assert raw[0]["toxicity"] == 0.99 and raw[1]["insult"] == 0.05
    assert effective_scores(contestant, raw[0]) == {
        "insult": 0.7, "threat": 0.1, "identity_hate": 0.2, "sexual_harassment": 0.3, "profanity": 0.9, FLAG: 0.7,
    }


def test_detoxify_handles_scalar_outputs_for_a_single_text():
    contestant = DetoxifyContestant(model=FakeDetoxify({k: v[0] for k, v in DETOX_OUT.items()}))

    assert contestant.predict(["x"], "es")[0]["insult"] == 0.7


class FakeTokenizer:
    def __init__(self):
        self.kwargs = None

    def __call__(self, texts, **kwargs):
        self.kwargs = kwargs
        n = len(texts)
        return {"input_ids": torch.zeros((n, 4), dtype=torch.long), "attention_mask": torch.ones((n, 4), dtype=torch.long)}


class FakeModel:
    config = SimpleNamespace(id2label={0: "toxicity", 1: "insult", 2: "identity_attack"})

    def to(self, device):
        return self

    def eval(self):
        return self

    def __call__(self, input_ids, attention_mask):
        return SimpleNamespace(logits=torch.tensor([[0.0, 2.0, -2.0]] * input_ids.shape[0]))


def test_hf_contestant_applies_sigmoid_names_outputs_and_truncates():
    tokenizer = FakeTokenizer()
    contestant = HFContestant("h", "org/model", "a" * 40, DETOXIFY_STYLE, "", "MIT", device="cpu",
                              loader=lambda: (tokenizer, FakeModel()))

    [raw] = contestant.predict(["hola"], "es")

    assert raw["toxicity"] == pytest.approx(0.5)
    assert raw["insult"] == pytest.approx(0.8808, abs=1e-4)
    assert tokenizer.kwargs["truncation"] is True and tokenizer.kwargs["max_length"] == 256
    assert contestant.to_scores(raw) == {"insult": raw["insult"], "identity_hate": raw["identity_attack"]}
    assert contestant.version == "a" * 12


def test_unload_releases_the_model_and_the_next_prediction_reloads_it():
    loads = []

    def loader():
        loads.append(1)
        return FakeTokenizer(), FakeModel()

    contestant = HFContestant("h", "org/model", "a" * 40, DETOXIFY_STYLE, "", "MIT", device="cpu", loader=loader)
    contestant.predict(["a"], "es")
    contestant.unload()
    contestant.predict(["b"], "es")

    assert len(loads) == 2
    detoxify = DetoxifyContestant(model=FakeDetoxify(DETOX_OUT))
    detoxify.unload()
    assert detoxify._model is None


def test_horizon_is_pinned_to_a_full_revision_sha():
    assert re.fullmatch(r"[0-9a-f]{40}", HORIZON_REVISION)


def test_registry_builds_every_contestant_without_loading_a_model():
    contestants = build_contestants(list(FACTORIES))

    assert [c.name for c in contestants] == list(FACTORIES)
    assert all(c.version and c.notes and c.languages for c in contestants)
    assert set(DEFAULT_CONTESTANTS) == set(FACTORIES)


def test_every_contestant_is_open_source():
    # The team only uses open, self-hostable models: no paid or hosted APIs.
    assert {c.name: c.license for c in build_contestants(list(FACTORIES)) if c.license not in OPEN_LICENSES} == {}


def test_unknown_contestant_is_an_error():
    with pytest.raises(ValueError, match="desconocido"):
        build_contestants(["gpt-9"])
