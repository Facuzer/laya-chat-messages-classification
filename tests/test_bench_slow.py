"""Benchmark checks against real datasets and models. They download data, so they only run with
`uv run pytest -m slow`."""

from collections import Counter

import pytest

pytestmark = pytest.mark.slow


@pytest.mark.parametrize("lang,minimum", [("es", 3500), ("pt", 3400), ("en", 3700)])
def test_real_hatecheck_loads_and_maps_every_case(lang, minimum):
    from bench.sources.hatecheck import load_hatecheck

    examples = load_hatecheck(lang)

    assert len(examples) >= minimum
    assert len(Counter(e.functionality for e in examples)) >= 27


def test_real_toldbr_and_olidbr_load_with_both_flag_classes():
    from bench.sources.portuguese import load_olidbr, load_toldbr
    from bench.taxonomy import FLAG, gold

    told, olid = load_toldbr(), load_olidbr()

    assert len(told) == 21000 and len(olid) == 1738
    for examples in (told, olid):
        flags = Counter(gold(e, FLAG) for e in examples)
        assert flags[True] > 100 and flags[False] > 100, flags


def test_real_laya_answers_every_taxonomy_question():
    from bench.contestants.base import effective_scores
    from bench.contestants.laya import laya_taxonomy

    contestant = laya_taxonomy()
    scores = [effective_scores(contestant, r) for r in contestant.predict(["sos un pelotudo", "buena mano, gg"], "es")]

    assert all(0.0 <= s["flag"] <= 1.0 and 0.0 <= s["targets_player"] <= 1.0 for s in scores)


@pytest.mark.parametrize("name", ["detoxify", "horizon-mmbert"])
def test_real_classifiers_score_a_spanish_insult_above_a_greeting(name):
    from bench.contestants import build_contestants
    from bench.contestants.base import effective_scores

    [contestant] = build_contestants([name])
    insult, greeting = (effective_scores(contestant, r)["flag"]
                        for r in contestant.predict(["sos un pelotudo de mierda", "hola, suerte a todos"], "es"))

    assert insult > greeting
