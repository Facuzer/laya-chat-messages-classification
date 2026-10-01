import pytest

from app.examples import ExampleStore, assign_split, normalize_text, parse_import_lines


@pytest.fixture
def store(tmp_path):
    return ExampleStore(tmp_path / "training" / "examples.json")


# --- normalization and splits ---------------------------------------------------------------


def test_normalize_collapses_case_and_whitespace_but_keeps_accents():
    assert normalize_text("  Sos   un\tPELOTUDO \n") == "sos un pelotudo"
    assert normalize_text("Qué Día") == "qué día"


def test_split_is_the_same_for_texts_that_normalize_the_same():
    assert assign_split("Sos un pelotudo") == assign_split("  sos   UN pelotudo ")


def test_splits_are_roughly_70_10_20():
    splits = [assign_split(f"mensaje de prueba número {i}") for i in range(3000)]

    assert 0.66 < splits.count("train") / 3000 < 0.74
    assert 0.07 < splits.count("calib") / 3000 < 0.13
    assert 0.17 < splits.count("test") / 3000 < 0.23


# --- candidates (generated / imported) --------------------------------------------------------


def test_candidates_arrive_pending_with_their_suggestion(store):
    result = store.add_candidates(
        [{"text": "callate gil", "label": "insult", "category": "insulto directo"}], source="generated"
    )

    assert result == {"added": 1, "duplicates": 0}
    [example] = store.all()
    assert example["text"] == "callate gil"
    assert example["source"] == "generated"
    assert example["status"] == "pending"
    assert example["label"] is None
    assert example["suggested"] == "insult"
    assert example["category"] == "insulto directo"
    assert example["split"] == assign_split("callate gil")


def test_accepted_labels_arrive_already_labeled(store):
    store.add_candidates([{"text": "gracias genio", "label": "clean"}], source="imported", accept_labels=True)

    [example] = store.all()
    assert example["status"] == "labeled"
    assert example["label"] == "clean"


def test_duplicate_texts_are_skipped_across_and_within_batches(store):
    store.add_candidates([{"text": "Sos un gil"}], source="generated")

    result = store.add_candidates([{"text": "sos un  GIL"}, {"text": "otra"}, {"text": "Otra"}], source="imported")

    assert result == {"added": 1, "duplicates": 2}
    assert sorted(e["text"] for e in store.all()) == ["Sos un gil", "otra"]


def test_unknown_suggested_label_is_rejected(store):
    with pytest.raises(ValueError, match="maybe"):
        store.add_candidates([{"text": "hola", "label": "maybe"}], source="imported")


# --- review queue -------------------------------------------------------------------------------


def test_next_pending_skips_reviewed_examples(store):
    store.add_candidates([{"text": "uno"}, {"text": "dos"}], source="generated")
    first = store.next_pending()

    store.set_label(first["id"], "clean")

    second = store.next_pending()
    assert second["text"] != first["text"]
    store.set_label(second["id"], "discard")
    assert store.next_pending() is None


def test_set_label_records_label_and_status(store):
    store.add_candidates([{"text": "te voy a romper la cara", "label": "insult"}], source="generated")
    example_id = store.all()[0]["id"]

    labeled = store.set_label(example_id, "insult")

    assert labeled["status"] == "labeled"
    assert labeled["label"] == "insult"
    assert labeled["labeled_at"]


def test_discard_clears_label(store):
    store.add_candidates([{"text": "asdf"}], source="generated")
    example_id = store.all()[0]["id"]
    store.set_label(example_id, "insult")

    discarded = store.set_label(example_id, "discard")

    assert discarded["status"] == "discarded"
    assert discarded["label"] is None


def test_set_label_on_unknown_id_raises_key_error(store):
    with pytest.raises(KeyError):
        store.set_label("nope", "clean")


def test_set_label_rejects_unknown_label(store):
    store.add_candidates([{"text": "hola"}], source="generated")
    with pytest.raises(ValueError):
        store.set_label(store.all()[0]["id"], "maybe")


# --- labels from the chat -------------------------------------------------------------------


def test_labeling_a_message_creates_a_chat_example(store):
    example = store.label_message("msg-1", "sos un pelotudo", "insult")

    assert example["source"] == "chat"
    assert example["message_id"] == "msg-1"
    assert example["status"] == "labeled"
    assert example["label"] == "insult"
    assert store.labels_by_message() == {"msg-1": "insult"}


def test_relabeling_a_message_updates_the_same_example(store):
    store.label_message("msg-1", "boludo vení", "insult")

    store.label_message("msg-1", "boludo vení", "clean")

    assert len(store.all()) == 1
    assert store.labels_by_message() == {"msg-1": "clean"}


def test_labeling_a_message_whose_text_already_exists_reuses_that_example(store):
    store.add_candidates([{"text": "Callate gil", "label": "insult"}], source="generated")

    store.label_message("msg-9", "callate gil", "insult")

    [example] = store.all()
    assert example["source"] == "generated"
    assert example["message_id"] == "msg-9"
    assert example["status"] == "labeled"


def test_labeling_a_message_rejects_discard(store):
    with pytest.raises(ValueError):
        store.label_message("msg-1", "hola", "discard")


# --- summary and training data ----------------------------------------------------------------


def test_summary_counts_statuses_labels_and_splits(store):
    store.add_candidates(
        [
            {"text": "a1", "label": "insult"},
            {"text": "a2", "label": "clean"},
            {"text": "a3", "label": "clean"},
        ],
        source="imported",
        accept_labels=True,
    )
    store.add_candidates([{"text": "pendiente"}, {"text": "basura"}], source="generated")
    store.set_label(next(e["id"] for e in store.all() if e["text"] == "basura"), "discard")

    summary = store.summary()

    assert summary["pending"] == 1
    assert summary["discarded"] == 1
    assert summary["labeled"] == {"insult": 1, "clean": 2}
    split_totals = {s: sum(c.values()) for s, c in summary["splits"].items()}
    assert sum(split_totals.values()) == 3
    assert set(summary["splits"]) == {"train", "calib", "test"}


def test_labeled_returns_only_labeled_examples_of_a_split(store):
    store.add_candidates([{"text": f"frase {i}", "label": "clean"} for i in range(30)], source="imported",
                         accept_labels=True)
    store.add_candidates([{"text": "sin revisar"}], source="generated")

    by_split = {s: store.labeled(s) for s in ("train", "calib", "test")}

    assert sum(len(v) for v in by_split.values()) == 30
    assert all(e["split"] == s and e["status"] == "labeled" for s, v in by_split.items() for e in v)


# --- import files ---------------------------------------------------------------------------


def test_parse_jsonl_lines():
    lines = ['{"text": "callate gil", "label": "insult", "category": "directo"}', "", '{"text": "hola"}']

    assert parse_import_lines(lines, fmt="jsonl") == [
        {"text": "callate gil", "label": "insult", "category": "directo"},
        {"text": "hola", "label": None, "category": None},
    ]


def test_parse_csv_lines_with_optional_label_column():
    lines = ["text,label", '"sos un gil, en serio",insult', "buen día,"]

    assert parse_import_lines(lines, fmt="csv") == [
        {"text": "sos un gil, en serio", "label": "insult", "category": None},
        {"text": "buen día", "label": None, "category": None},
    ]


def test_parse_rejects_rows_without_text():
    with pytest.raises(ValueError, match="line 2"):
        parse_import_lines(['{"text": "ok"}', '{"label": "insult"}'], fmt="jsonl")
