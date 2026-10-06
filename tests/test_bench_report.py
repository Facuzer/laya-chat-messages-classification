from bench.report import render

FLAG_ROW = {"contestant": "laya", "source": "suite-es", "lang": "es", "label": "flag", "n": 90, "positives": 40,
            "prevalence": 0.44, "auroc": 0.91, "auroc_ci": [0.85, 0.95], "ap": 0.9, "threshold": 0.42,
            "recall_at_op": 0.7, "fpr_at_op": 0.06, "precision_at_op": 0.9, "enough": True}


def board(**overrides):
    b = {
        "meta": {"created_at": "2026-10-06T12:00:00Z", "preliminary": False, "n_boot": 100,
                 "contestants": [{"name": "laya", "version": "q1", "license": "Apache-2.0", "notes": "Nota de laya."},
                                 {"name": "detoxify", "version": "m", "license": "Apache-2.0", "notes": "Nota de detoxify."}],
                 "sources": [{"name": "suite-es", "lang": "es", "license": "propia", "n": 100}]},
        "coverage": [{"source": "suite-es", "lang": "es", "label": "flag", "positives": 40, "negatives": 50},
                     {"source": "suite-es", "lang": "es", "label": "profanity", "positives": 10, "negatives": 5}],
        "thresholds": {"laya": {"flag": 0.42, "insult": None}, "detoxify": {"flag": None}},
        "slices": [FLAG_ROW, {**FLAG_ROW, "label": "profanity", "n": 15, "positives": 10, "enough": False}],
        "comparisons": [{"source": "suite-es", "best": "laya", "contestant": "detoxify", "diff": -0.1, "lo": -0.2, "hi": -0.01}],
        "functionalities": [{"contestant": "laya", "source": "suite-es", "functionality": "negation", "n": 30, "accuracy": 0.9}],
        "routing": [{"contestant": "laya", "source": "suite-es", "english_share": 0.02}],
        "latency": [],
    }
    return b | overrides


def test_flag_cell_shows_auroc_interval_and_operating_point():
    assert "0.91 [0.85–0.95]<br>R 70% · FPR 6%" in render(board())


def test_missing_contestant_rows_say_na_and_uncovered_labels_say_dash():
    md = render(board())
    flag_line = next(line for line in md.splitlines() if line.startswith("| detoxify |"))

    assert "n/a" in flag_line
    assert "| laya | — |" in md  # per-category table: insult has no coverage, so it is a dash


def test_small_slices_say_pocos_datos_instead_of_numbers():
    assert "pocos datos (10+/5−)" in render(board())


def test_preliminary_banner_only_when_preliminary():
    assert "PRELIMINAR" not in render(board())
    assert "PRELIMINAR" in render(board(meta={**board()["meta"], "preliminary": True}))


def test_comparisons_functionalities_routing_notes_and_thresholds_are_rendered():
    md = render(board())

    assert "| suite-es | laya | detoxify | -0.10 [-0.20 – -0.01] |" in md
    assert "| negation | 30 | 90% | n/a |" in md
    assert "| laya | suite-es | 2% |" in md
    assert "Nota de laya. Umbrales de operación: flag=0.42." in md
    assert "Nota de detoxify. Umbrales de operación: n/a." in md


def test_latency_section_only_when_measured():
    assert "## Latencia" not in render(board())
    latency = [{"contestant": "laya", "device": "cpu", "lang": "es", "n": 200, "p50_ms": 120.4, "p95_ms": 300.2,
                "throughput_per_s": 15.2, "batch_size": 32}]
    assert "| laya | cpu | es | 120 | 300 | 15 |" in render(board(latency=latency))
