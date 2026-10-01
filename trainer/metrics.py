"""How well an insult classifier does on labeled examples, and whether a new one should replace the old."""

MIN_TEST_PER_CLASS = 10
# Agreement with the base model is only a proxy: sentiment has no labels, and many of the answers
# fine-tuning changes were wrong before (an unaccented "qué rico estuvo el asado" read as negative).
# Runs on the generated set keep 87-90% with most changes being fixes; a sentiment collapse falls far
# below this.
MIN_SENTIMENT_AGREEMENT = 0.85


def binary_metrics(labels: list[str], p_insult: list[float], threshold: float) -> dict:
    predicted = [p >= threshold for p in p_insult]
    actual = [label == "insult" for label in labels]
    tp = sum(p and a for p, a in zip(predicted, actual))
    fp = sum(p and not a for p, a in zip(predicted, actual))
    fn = sum(not p and a for p, a in zip(predicted, actual))
    n = len(labels)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "n": n,
        "positives": sum(actual),
        "false_positives": fp,
        "false_negatives": fn,
        "accuracy": (n - fp - fn) / n if n else 0.0,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def calibration_error(labels: list[str], p_insult: list[float], bins: int = 10) -> float:
    """Expected calibration error: how far the model's confidence is from how often it is right."""
    n = len(labels)
    if not n:
        return 0.0
    buckets: list[list[tuple[float, bool]]] = [[] for _ in range(bins)]
    for label, p in zip(labels, p_insult):
        confidence = max(p, 1 - p)
        correct = (p >= 0.5) == (label == "insult")
        buckets[min(int(confidence * bins), bins - 1)].append((confidence, correct))
    error = 0.0
    for bucket in buckets:
        if bucket:
            mean_conf = sum(c for c, _ in bucket) / len(bucket)
            accuracy = sum(ok for _, ok in bucket) / len(bucket)
            error += len(bucket) / n * abs(accuracy - mean_conf)
    return error


def agreement(a: list[str], b: list[str]) -> float:
    return sum(x == y for x, y in zip(a, b)) / len(a) if a else 1.0


def activation_decision(base: dict, tuned: dict, sentiment_agreement: float, test_counts: dict) -> tuple[bool, str]:
    """Whether the fine-tuned checkpoint should replace the one in use, with the reason in Spanish."""
    few = [label for label in ("insult", "clean") if test_counts.get(label, 0) < MIN_TEST_PER_CLASS]
    if few:
        return False, (
            f"Hay menos de {MIN_TEST_PER_CLASS} ejemplos de prueba de tipo {' y '.join(few)}: "
            "no alcanza para saber si el modelo nuevo es mejor. Etiquetá más ejemplos."
        )
    if tuned["f1"] <= base["f1"]:
        return False, f"El modelo nuevo no detecta mejor los insultos (F1 {base['f1']:.2f} → {tuned['f1']:.2f})."
    if sentiment_agreement < MIN_SENTIMENT_AGREEMENT:
        return False, (
            f"El modelo nuevo cambió demasiado el sentimiento: coincide con el original en el "
            f"{sentiment_agreement:.0%} de los mensajes (mínimo {MIN_SENTIMENT_AGREEMENT:.0%})."
        )
    return True, (
        f"Detecta mejor los insultos (F1 {base['f1']:.2f} → {tuned['f1']:.2f}) y mantiene el "
        f"sentimiento (coincide en el {sentiment_agreement:.0%})."
    )
