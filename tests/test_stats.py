from app.stats import compute_stats


def message(sentiment: str, flagged: bool) -> dict:
    return {"sentiment": {"label": sentiment}, "flagged": flagged}


def test_empty_list_has_all_counts_at_zero():
    assert compute_stats([]) == {"total": 0, "positive": 0, "negative": 0, "neutral": 0, "flagged": 0}


def test_counts_each_sentiment_and_flagged_messages():
    messages = [
        message("positive", False),
        message("negative", True),
        message("negative", True),
        message("negative", False),
        message("neutral", False),
        message("neutral", True),
    ]

    assert compute_stats(messages) == {"total": 6, "positive": 1, "negative": 3, "neutral": 2, "flagged": 3}
