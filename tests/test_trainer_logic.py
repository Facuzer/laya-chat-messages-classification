import pytest

from trainer.metrics import activation_decision, agreement, binary_metrics, calibration_error
from trainer.targets import distribution_target, label_target

CRITERIA = {"insult": "…", "clean": "…"}
SENTIMENT = {"positive": "…", "negative": "…", "neutral": "…"}


# --- targets --------------------------------------------------------------------------------


def test_label_target_is_one_hot_in_criteria_order():
    assert label_target(CRITERIA, "insult") == [1.0, 0.0]
    assert label_target(CRITERIA, "clean") == [0.0, 1.0]


def test_label_target_rejects_labels_outside_the_criteria():
    with pytest.raises(ValueError):
        label_target(CRITERIA, "maybe")


def test_distribution_target_follows_criteria_order_and_normalizes():
    probs = {"neutral": 0.1, "positive": 0.6, "negative": 0.1}

    assert distribution_target(SENTIMENT, probs) == pytest.approx([0.75, 0.125, 0.125])


def test_distribution_target_without_mass_is_uniform():
    assert distribution_target(SENTIMENT, {}) == pytest.approx([1 / 3, 1 / 3, 1 / 3])


# --- metrics --------------------------------------------------------------------------------


def test_binary_metrics_at_threshold():
    labels = ["insult", "insult", "insult", "clean", "clean"]
    p_insult = [0.9, 0.75, 0.2, 0.8, 0.1]  # TP, TP, FN, FP, TN at 0.7

    m = binary_metrics(labels, p_insult, threshold=0.7)

    assert m["n"] == 5
    assert m["positives"] == 3
    assert m["false_positives"] == 1
    assert m["false_negatives"] == 1
    assert m["accuracy"] == pytest.approx(3 / 5)
    assert m["precision"] == pytest.approx(2 / 3)
    assert m["recall"] == pytest.approx(2 / 3)
    assert m["f1"] == pytest.approx(2 / 3)


def test_binary_metrics_with_no_predicted_positives_has_zero_precision_and_f1():
    m = binary_metrics(["insult", "clean"], [0.1, 0.2], threshold=0.7)

    assert m["precision"] == 0.0
    assert m["recall"] == 0.0
    assert m["f1"] == 0.0


def test_calibration_error_is_zero_when_confidence_matches_accuracy():
    # Four answers at confidence 0.75, three of them right: perfectly calibrated.
    labels = ["insult", "insult", "insult", "clean"]
    p_insult = [0.75, 0.75, 0.75, 0.75]

    assert calibration_error(labels, p_insult) == pytest.approx(0.0)


def test_calibration_error_measures_overconfidence():
    # Confidence 0.95 on every example, but only half are right: gap 0.45.
    labels = ["insult", "clean"]
    p_insult = [0.95, 0.95]

    assert calibration_error(labels, p_insult) == pytest.approx(0.45)


def test_agreement_is_share_of_equal_labels():
    assert agreement(["positive", "neutral", "negative", "neutral"], ["positive", "neutral", "neutral", "neutral"]) == 0.75


# --- activation rule ------------------------------------------------------------------------

ENOUGH = {"insult": 20, "clean": 30}


def test_activates_when_f1_improves_and_sentiment_is_preserved():
    ok, reason = activation_decision({"f1": 0.60}, {"f1": 0.80}, sentiment_agreement=0.95, test_counts=ENOUGH)

    assert ok is True
    assert reason


def test_does_not_activate_when_f1_does_not_improve():
    ok, _ = activation_decision({"f1": 0.80}, {"f1": 0.80}, sentiment_agreement=0.99, test_counts=ENOUGH)

    assert ok is False


def test_does_not_activate_when_sentiment_drifts():
    ok, _ = activation_decision({"f1": 0.60}, {"f1": 0.90}, sentiment_agreement=0.80, test_counts=ENOUGH)

    assert ok is False


@pytest.mark.parametrize("counts", [{"insult": 9, "clean": 30}, {"insult": 30, "clean": 9}])
def test_does_not_activate_with_too_few_test_examples_of_a_class(counts):
    ok, _ = activation_decision({"f1": 0.10}, {"f1": 0.95}, sentiment_agreement=1.0, test_counts=counts)

    assert ok is False
