import pytest

import bench.contestants.laya as laya_module
from bench.contestants.base import effective_scores
from bench.contestants.laya import (
    POSITIVE, TAXONOMY_QUESTIONS, LayaContestant, flatten, laya_app, laya_taxonomy, taxonomy_scores,
)
from bench.taxonomy import CATEGORIES, FLAG, TARGETS_PLAYER


def answer(options, first=0.8):
    rest = (1 - first) / (len(options) - 1)
    return {"type": "choice", "choice": options[0], "confidence": 0.4, "answer_confidence": first,
            "probabilities": {o: first if i == 0 else rest for i, o in enumerate(options)}, "action": {"act_probability": 0.5}}


def result(questions, model="multilingual"):
    return {"answers": {qid: answer(list(q["criteria"])) for qid, q in questions.items()},
            "routing": {"model": model, "repo": "r", "reason": "x"}, "usage": {}}


class FakeRouter:
    def __init__(self, model="multilingual"):
        self.batches, self.preloaded, self.model = [], None, model

    def preload(self, names):
        self.preloaded = names
        return self

    def predict_batch(self, requests, **kwargs):
        self.batches.append(requests)
        return [result(r["questions"], self.model) for r in requests]


def test_predict_sends_one_request_per_message_in_one_batch():
    router = FakeRouter()

    raw = laya_taxonomy(router=router).predict(["sos un manco", "gg"], "es")

    assert router.batches == [[
        {"state": {"message": "sos un manco"}, "questions": TAXONOMY_QUESTIONS},
        {"state": {"message": "gg"}, "questions": TAXONOMY_QUESTIONS},
    ]]
    assert raw[0]["insult.insult"] == 0.8 and raw[0]["routed_english"] == 0.0


def test_english_routing_is_recorded():
    assert laya_taxonomy(router=FakeRouter(model="english")).predict(["you suck"], "en")[0]["routed_english"] == 1.0


def test_taxonomy_scores_read_the_positive_option_of_each_question():
    raw = flatten(result(TAXONOMY_QUESTIONS))

    scores = taxonomy_scores(raw)

    assert set(scores) == set(CATEGORIES) | {TARGETS_PLAYER}
    assert scores["identity_hate"] == raw["identity_hate.hate"] == 0.8
    assert scores[TARGETS_PLAYER] == raw["target.player"]


def test_flag_is_the_max_of_the_core_categories():
    raw = {**flatten(result(TAXONOMY_QUESTIONS)), "threat.threat": 0.95}

    assert effective_scores(laya_taxonomy(router=FakeRouter()), raw)[FLAG] == 0.95


def test_app_contestant_asks_the_apps_questions_and_uses_insult_as_flag():
    from app.classifier import QUESTIONS

    router = FakeRouter()
    contestant = laya_app(router=router)
    [raw] = contestant.predict(["sos un idiota"], "es")

    assert router.batches[0][0]["questions"] == QUESTIONS
    assert effective_scores(contestant, raw) == {FLAG: raw["insult.insult"]}


def test_version_follows_the_questions():
    same = LayaContestant("x", TAXONOMY_QUESTIONS, taxonomy_scores, "", router=FakeRouter())
    changed = {**TAXONOMY_QUESTIONS, "threat": {**TAXONOMY_QUESTIONS["threat"], "instructions": "Is `message` a threat?"}}

    assert same.version == LayaContestant("x", TAXONOMY_QUESTIONS, taxonomy_scores, "", router=FakeRouter()).version
    assert same.version != LayaContestant("x", changed, taxonomy_scores, "", router=FakeRouter()).version


def test_router_is_built_lazily_once_with_the_device_and_both_checkpoints(monkeypatch):
    built = []

    def fake_build_router(device=None):
        built.append(device)
        return FakeRouter()

    monkeypatch.setattr(laya_module, "build_router", fake_build_router)
    contestant = laya_taxonomy(device="cpu")
    assert built == []

    contestant.predict(["hola"], "es")
    contestant.predict(["chau"], "es")

    assert built == ["cpu"]
    assert contestant._router.preloaded == ["english", "multilingual"]


def test_unload_drops_the_router_and_the_next_prediction_rebuilds_it(monkeypatch):
    built = []
    monkeypatch.setattr(laya_module, "build_router", lambda device=None: built.append(device) or FakeRouter())
    contestant = laya_taxonomy()

    contestant.predict(["hola"], "es")
    contestant.unload()
    contestant.predict(["chau"], "es")

    assert len(built) == 2


@pytest.mark.parametrize("qid", list(TAXONOMY_QUESTIONS))
def test_every_question_refers_to_the_message_and_has_two_or_three_options(qid):
    question = TAXONOMY_QUESTIONS[qid]

    assert "`message`" in question["instructions"]
    assert 2 <= len(question["criteria"]) <= 3


def test_every_positive_option_exists():
    for category, option in POSITIVE.items():
        assert option in TAXONOMY_QUESTIONS[category]["criteria"]
