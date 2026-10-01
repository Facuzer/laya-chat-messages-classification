SENTIMENTS = ("positive", "negative", "neutral")


def compute_stats(messages: list[dict]) -> dict[str, int]:
    stats = {"total": len(messages), **{s: 0 for s in SENTIMENTS}, "flagged": 0}
    for message in messages:
        stats[message["sentiment"]["label"]] += 1
        stats["flagged"] += message["flagged"]
    return stats
