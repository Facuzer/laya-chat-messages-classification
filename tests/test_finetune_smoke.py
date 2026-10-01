"""End-to-end fine-tuning on the real Laya checkpoint with a tiny synthetic dataset.

Slow (loads the ~1.2 GB multilingual checkpoint and trains for one epoch), so it only runs with
`uv run pytest -m slow`.
"""

import json
import math

import pytest

from app.checkpoints import ActiveCheckpoint
from app.classifier import QUESTIONS, STATE_KEY
from app.examples import ExampleStore
from trainer.finetune import NotEnoughData, TrainConfig, run

pytestmark = pytest.mark.slow

NAMES = ["Juan", "Caro", "Pedro", "Lu", "Martín", "Sofi", "Nico", "Vale", "Tomi", "Agus", "Fer", "Majo"]
INSULTS = ["sos un pelotudo", "sos una forra", "callate gil", "sos un inútil de mierda", "te voy a romper la cara",
           "sos un tarado", "andate a la mierda"]
FRIENDLY = ["gracias por la ayuda", "nos vemos mañana", "qué lindo día", "¿venís al asado?",
            "buenísimo, dale", "mandame el archivo", "llego en cinco minutos"]


def dataset():
    rows = [{"text": f"{name}, {phrase}", "label": "insult"} for name in NAMES for phrase in INSULTS]
    rows += [{"text": f"{name}, {phrase}", "label": "clean"} for name in NAMES for phrase in FRIENDLY]
    return rows


@pytest.fixture(scope="module")
def trained(tmp_path_factory):
    root = tmp_path_factory.mktemp("ft")
    store = ExampleStore(root / "examples.json")
    store.add_candidates(dataset(), source="imported", accept_labels=True)
    models_dir = root / "models"
    result = run(store, models_dir, threshold=0.7, config=TrainConfig(epochs=1), activate=False, log=lambda *_: None)
    return result, models_dir


def test_checkpoint_folder_has_everything_laya_loads(trained):
    result, _ = trained
    ckpt = result["checkpoint"]

    for name in ("model.safetensors", "rl_agent_config.json", "training_meta.json", "tokenizer", "encoder"):
        assert (ckpt / name).exists(), name


def test_training_meta_records_questions_and_metrics(trained):
    result, _ = trained
    meta = json.loads((result["checkpoint"] / "training_meta.json").read_text(encoding="utf-8"))

    assert meta["questions"] == QUESTIONS
    assert all(math.isfinite(loss) for loss in meta["losses"])
    assert set(meta["metrics"]) >= {"base", "tuned", "sentiment_agreement", "test_counts"}
    assert 0.0 <= meta["metrics"]["tuned"]["f1"] <= 1.0


def test_saved_checkpoint_loads_and_answers(trained):
    import laya

    result, _ = trained
    agent = laya.load(str(result["checkpoint"]))

    answer = agent.predict({STATE_KEY: "Juan, sos un pelotudo"}, QUESTIONS)["answers"]["insult"]

    assert set(answer["probabilities"]) == {"insult", "clean"}
    assert sum(answer["probabilities"].values()) == pytest.approx(1.0, abs=1e-3)


def test_saved_weights_are_the_trained_ones_not_the_base(trained):
    import torch
    from safetensors.torch import load_file

    from trainer.finetune import _base_dir

    result, _ = trained
    tuned = load_file(str(result["checkpoint"] / "model.safetensors"))
    base = load_file(str(_base_dir() / "model.safetensors"))

    assert tuned.keys() == base.keys()
    assert any(not torch.equal(tuned[k], base[k]) for k in base)


def test_activate_false_leaves_the_base_model_in_use(trained):
    _, models_dir = trained

    assert ActiveCheckpoint(models_dir).get() is None


def test_too_few_examples_refuses_before_loading_anything(tmp_path):
    store = ExampleStore(tmp_path / "examples.json")
    store.add_candidates([{"text": "sos un gil", "label": "insult"}], source="imported", accept_labels=True)

    with pytest.raises(NotEnoughData):
        run(store, tmp_path / "models", threshold=0.7, log=lambda *_: None)
