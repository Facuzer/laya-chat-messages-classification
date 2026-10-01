"""Run sample messages through the real Laya model and print each verdict.

Use it to tune FLAG_THRESHOLD and the wording in app/classifier.py:QUESTIONS.

    uv run python scripts/try_laya.py
    uv run python scripts/try_laya.py "otra frase" "another one"
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.classifier import load_classifier  # noqa: E402

SAMPLES = [
    "Sos un idiota, no servís para nada.",
    "No sos un idiota, tranquilo.",
    "sos un pelotudo",
    "boludo",
    "Qué día de mierda.",
    "Te voy a romper la cara.",
    "La app anda lentísima y estoy re decepcionado.",
    "¡Gracias! Me encantó cómo quedó.",
    "Mañana a las 10 es la reunión.",
    "You're a complete moron.",
    "This is great, thanks a lot!",
    "The meeting is at 10am.",
]


def main() -> None:
    texts = sys.argv[1:] or SAMPLES
    classifier = load_classifier()
    print(f"threshold={classifier.threshold}\n")
    print(f"{'flag':<5} {'P(insult)':>9} {'sentiment':<9} {'conf':>5} {'model':<12} {'ms':>5}  text")
    for text in texts:
        c = classifier.classify(text)
        print(
            f"{'YES' if c.flagged else '-':<5} {c.insult.probabilities['insult']:>9.2f} "
            f"{c.sentiment.label:<9} {c.sentiment.confidence:>5.2f} {c.model:<12} {c.latency_ms:>5}  {text}"
        )


if __name__ == "__main__":
    main()
