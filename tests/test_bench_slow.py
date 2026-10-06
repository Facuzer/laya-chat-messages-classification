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
