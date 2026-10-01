"""Training targets, one probability per option in the question's criteria order."""


def label_target(criteria: dict, label: str) -> list[float]:
    keys = list(criteria)
    if label not in keys:
        raise ValueError(f"label {label!r} is not an option of this question ({', '.join(keys)})")
    return [1.0 if key == label else 0.0 for key in keys]


def distribution_target(criteria: dict, probabilities: dict) -> list[float]:
    values = [float(probabilities.get(key, 0.0)) for key in criteria]
    total = sum(values)
    if total <= 0:
        return [1.0 / len(values)] * len(values)
    return [v / total for v in values]
