import pytest

from app.classifier import QUESTIONS, Classification, LayaClassifier, Verdict, to_classification


def laya_result(insult_choice="insult", insult_answer_conf=0.93, insult_entropy_conf=0.63, model="multilingual"):
    """A Router.predict result with every field Laya returns for two `choice` questions."""
    other = round(1 - insult_answer_conf, 4)
    insult_probs = (
        {"insult": insult_answer_conf, "clean": other}
        if insult_choice == "insult"
        else {"insult": other, "clean": insult_answer_conf}
    )
    return {
        "answers": {
            "insult": {
                "type": "choice",
                "choice": insult_choice,
                "probabilities": insult_probs,
                "confidence": insult_entropy_conf,
                "answer_confidence": insult_answer_conf,
                "action": {"act_probability": 0.51},
            },
            "sentiment": {
                "type": "choice",
                "choice": "negative",
                "probabilities": {"positive": 0.02, "negative": 0.88, "neutral": 0.1},
                "confidence": 0.51,
                "answer_confidence": 0.88,
                "action": {"act_probability": 0.49},
            },
        },
        "routing": {"model": model, "repo": "convaiinnovations/laya", "reason": "Latin script, Spanish function words"},
        "usage": {"input_tokens": 57, "output_tokens": 0},
    }


def test_insult_above_threshold_is_flagged():
    assert to_classification(laya_result(insult_answer_conf=0.93), threshold=0.7, latency_ms=40).flagged is True


def test_insult_below_threshold_is_not_flagged():
    assert to_classification(laya_result(insult_answer_conf=0.65), threshold=0.7, latency_ms=40).flagged is False


def test_insult_exactly_at_threshold_is_flagged():
    assert to_classification(laya_result(insult_answer_conf=0.7), threshold=0.7, latency_ms=40).flagged is True


def test_threshold_reads_answer_confidence_not_entropy_confidence():
    result = laya_result(insult_answer_conf=0.75, insult_entropy_conf=0.19)

    assert to_classification(result, threshold=0.7, latency_ms=40).flagged is True


def test_confident_clean_answer_is_not_flagged():
    result = laya_result(insult_choice="clean", insult_answer_conf=0.95)

    assert to_classification(result, threshold=0.7, latency_ms=40).flagged is False


def test_classification_carries_verdicts_model_and_latency():
    classification = to_classification(laya_result(), threshold=0.7, latency_ms=42)

    assert classification == Classification(
        flagged=True,
        insult=Verdict(label="insult", confidence=0.93, probabilities={"insult": 0.93, "clean": 0.07}),
        sentiment=Verdict(
            label="negative", confidence=0.88, probabilities={"positive": 0.02, "negative": 0.88, "neutral": 0.1}
        ),
        model="multilingual",
        latency_ms=42,
    )


class RecordingRouter:
    def __init__(self, result):
        self.result = result
        self.calls = []

    def predict(self, state, questions):
        self.calls.append((state, questions))
        return self.result


def test_classify_sends_message_state_and_both_questions_in_one_call():
    router = RecordingRouter(laya_result())

    LayaClassifier(router, threshold=0.7).classify("sos un idiota")

    assert router.calls == [({"message": "sos un idiota"}, QUESTIONS)]


def test_classify_applies_the_configured_threshold():
    router = RecordingRouter(laya_result(insult_answer_conf=0.8))

    assert LayaClassifier(router, threshold=0.9).classify("hola").flagged is False
    assert LayaClassifier(router, threshold=0.75).classify("hola").flagged is True


def test_classify_reports_non_negative_latency():
    classification = LayaClassifier(RecordingRouter(laya_result()), threshold=0.7).classify("hola")

    assert classification.latency_ms >= 0


def test_classify_records_each_latency_for_metrics():
    classifier = LayaClassifier(RecordingRouter(laya_result()), threshold=0.7)

    classifier.classify("hola")
    classifier.classify("chau")

    assert classifier.latency.snapshot()["total"] == 2


@pytest.mark.parametrize("key", ["insult", "sentiment"])
def test_every_question_refers_to_the_state_key_classify_sends(key):
    assert "`message`" in QUESTIONS[key]["instructions"]
