import json

import pytest

from bench.cli import main
from bench.latency import measure, percentile
from bench.sources import Source
from bench.taxonomy import Example


class Fake:
    name = "fake"
    version = "1"
    notes = "Modelo de prueba."
    license = "MIT"
    languages = frozenset({"es"})

    def __init__(self, device=None):
        self.calls = 0

    def predict(self, texts, lang):
        self.calls += 1
        return [{"p": 0.9 if "idiota" in t else 0.1} for t in texts]

    def to_scores(self, raw):
        return {"insult": raw["p"], "threat": 0.0, "identity_hate": 0.0}


def fake_sources():
    examples = [Example(id=f"f:{i}", source="fake-es", lang="es", text=f"sos un idiota {i}" if i % 2 else f"hola {i}",
                        labels={"insult": bool(i % 2), "threat": False, "identity_hate": False}) for i in range(400)]
    return {"fake-es": Source("fake-es", "es", "test", lambda _: examples)}


def run_main(*args, factories=None):
    return main(list(args), sources=fake_sources(), factories=factories or {"fake": Fake})


def test_run_writes_markdown_and_json_scoreboards(tmp_path):
    assert run_main("run", "--contestants", "fake", "--out", str(tmp_path), "--bootstrap", "20") == 0

    board = json.loads((tmp_path / "scoreboard.json").read_text(encoding="utf-8"))
    [flag] = [r for r in board["slices"] if r["label"] == "flag" and r["source"] == "fake-es"]
    assert flag["auroc"] == 1.0 and flag["enough"] is True
    md = (tmp_path / "scoreboard.md").read_text(encoding="utf-8")
    assert "Modelo de prueba." in md and "PRELIMINAR" not in md


def test_include_unreviewed_marks_the_scoreboard_preliminary(tmp_path):
    run_main("run", "--contestants", "fake", "--out", str(tmp_path), "--bootstrap", "5", "--include-unreviewed")

    assert "PRELIMINAR" in (tmp_path / "scoreboard.md").read_text(encoding="utf-8")


def test_unknown_names_are_reported(tmp_path, capsys):
    assert run_main("run", "--sources", "nope", "--contestants", "fake", "--out", str(tmp_path)) == 1
    assert "nope" in capsys.readouterr().out


def test_internal_errors_keep_their_traceback(tmp_path):
    class Broken(Fake):
        def predict(self, texts, lang):
            raise ValueError("bug interno")

    with pytest.raises(ValueError, match="bug interno"):
        run_main("run", "--contestants", "fake", "--out", str(tmp_path), factories={"fake": Broken})


def test_latency_keeps_one_entry_per_contestant_device_and_language(tmp_path):
    args = ("latency", "--contestants", "fake", "--device", "cpu", "--lang", "es", "--n", "5", "--out", str(tmp_path))

    assert run_main(*args) == 0 and run_main(*args) == 0

    entries = json.loads((tmp_path / "latency.json").read_text(encoding="utf-8"))
    assert [(e["contestant"], e["device"], e["lang"], e["n"]) for e in entries] == [("fake", "cpu", "es", 5)]


def test_latency_keeps_earlier_measurements_when_a_later_contestant_fails(tmp_path):
    class Exploding(Fake):
        name = "exploding"

        def predict(self, texts, lang):
            raise RuntimeError("CUDA out of memory")

    with pytest.raises(RuntimeError):
        run_main("latency", "--contestants", "fake,exploding", "--n", "3", "--out", str(tmp_path),
                 factories={"fake": Fake, "exploding": Exploding})

    entries = json.loads((tmp_path / "latency.json").read_text(encoding="utf-8"))
    assert [e["contestant"] for e in entries] == ["fake"]


def test_latency_unloads_each_contestant_after_measuring_it(tmp_path):
    unloaded = []

    class Unloading(Fake):
        def unload(self):
            unloaded.append(self.name)

    assert run_main("latency", "--contestants", "fake", "--n", "3", "--out", str(tmp_path), factories={"fake": Unloading}) == 0
    assert unloaded == ["fake"]


def test_percentile_uses_nearest_rank():
    assert percentile([5, 1, 3, 2, 4], 50) == 3
    assert percentile(list(range(1, 11)), 95) == 10


def test_measure_times_single_messages_and_batches_after_a_warm_up():
    class Clock:
        t = 0.0

        def __call__(self):
            self.t += 0.01
            return self.t

    contestant = Fake()

    result = measure(contestant, [f"m{i}" for i in range(10)], "es", batch_size=4, clock=Clock())

    assert contestant.calls == 1 + 10 + 3
    assert result["p50_ms"] == pytest.approx(10.0) and result["p95_ms"] == pytest.approx(10.0)
    assert result["throughput_per_s"] == pytest.approx(10 / 0.01)
    assert (result["n"], result["batch_size"]) == (10, 4)
