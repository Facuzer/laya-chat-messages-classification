"""Routing runs on the real Laya Router. Deciding a route loads no checkpoint, so these stay fast."""

import pytest

from app.classifier import QUESTIONS, STATE_KEY, build_router


@pytest.fixture(scope="module")
def router():
    return build_router()


def route(router, text: str) -> str:
    return router.route({STATE_KEY: text}, QUESTIONS)["model"]


@pytest.mark.parametrize(
    "text",
    [
        # Every Spanish function word here (como, con, la, por, ser) is shared with other
        # Romance languages, and there are no accents, so Laya cannot identify the language.
        "Debido a la incapacidad de conectar neuronas por partes de ustedes considero oportuno "
        "no volver a jugar con gente involucionada como parecen ser ustedes",
        "callate gil",
        "sos un pelotudo",
    ],
)
def test_unidentified_spanish_goes_to_multilingual(router, text):
    assert route(router, text) == "multilingual"


def test_accented_spanish_goes_to_multilingual(router):
    assert route(router, "¡Gracias! Me encantó cómo quedó.") == "multilingual"


@pytest.mark.parametrize("text", ["You're a complete moron.", "The meeting is at 10am."])
def test_identified_english_goes_to_english(router, text):
    assert route(router, text) == "english"


def test_fine_tuned_checkpoint_replaces_only_the_multilingual_model(router, tmp_path):
    tuned = build_router(checkpoint=tmp_path)

    assert tuned.models["multilingual"] == str(tmp_path)
    assert tuned.models["english"] == router.models["english"]


def test_without_checkpoint_multilingual_comes_from_the_hub(router):
    assert "convaiinnovations" in str(router.models["multilingual"])


def test_explicit_device_overrides_the_environment(monkeypatch):
    import laya

    seen = {}
    monkeypatch.setattr(laya, "Router", lambda **kwargs: seen.update(kwargs))
    monkeypatch.setenv("LAYA_DEVICE", "cuda")

    build_router(device="cpu")

    assert seen["device"] == "cpu"


def test_revision_is_passed_to_the_router_and_defaults_to_none(monkeypatch):
    import laya

    seen = []
    monkeypatch.setattr(laya, "Router", lambda **kwargs: seen.append(kwargs))

    build_router(revision="abc")
    build_router()

    assert [kwargs["revision"] for kwargs in seen] == ["abc", None]
