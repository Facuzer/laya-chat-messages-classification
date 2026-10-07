"""Fine-tune Laya's multilingual checkpoint on the labeled examples, evaluate it against the base
model, and activate it only if it detects insults better without changing sentiment.

    uv run python scripts/finetune.py
    uv run python scripts/finetune.py --epochs 6
    uv run python scripts/finetune.py --no-activate

Restart the server afterwards so it loads the active checkpoint.
"""

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.checkpoints import MODELS_DIR  # noqa: E402
from app.examples import DEFAULT_PATH, ExampleStore  # noqa: E402
from trainer.finetune import NotEnoughData, TrainConfig, run  # noqa: E402


SENTIMENT = {"positive": "positivo", "negative": "negativo", "neutral": "neutro"}


def pct(x: float) -> str:
    return f"{x:.0%}"


def print_report(result: dict) -> None:
    m = result["metrics"]
    base, tuned = m["base"], m["tuned"]
    print(f"\nResultados con {base['n']} ejemplos de prueba (umbral {m['threshold']}):")
    print(f"  {'':<34}{'original':>10}{'afinado':>10}")
    rows = [
        ("Aciertos", "accuracy"),
        ("Insultos detectados (recall)", "recall"),
        ("Alarmas correctas (precision)", "precision"),
        ("F1 (balance de las dos)", "f1"),
    ]
    for name, key in rows:
        print(f"  {name:<34}{pct(base[key]):>10}{pct(tuned[key]):>10}")
    print(f"  {'Insultos que se escaparon':<34}{base['false_negatives']:>10}{tuned['false_negatives']:>10}")
    print(f"  {'Mensajes tapados sin motivo':<34}{base['false_positives']:>10}{tuned['false_positives']:>10}")
    print(f"  {'Error de calibración (menor=mejor)':<34}{base['calibration_error']:>10.3f}{tuned['calibration_error']:>10.3f}")
    print(f"  Sentimiento igual al original en el {pct(m['sentiment_agreement'])} de los mensajes.")
    if m["sentiment_changes"]:
        print("\n  Mensajes que cambiaron de sentimiento (revisá si el cambio tiene sentido):")
        for c in m["sentiment_changes"][:15]:
            print(f"    {SENTIMENT[c['base']]:>9} → {SENTIMENT[c['tuned']]:<9}{c['text']}")
    if m["by_category"]:
        print("\n  Aciertos por categoría:")
        for cat, c in m["by_category"].items():
            print(f"    {cat:<32}{pct(c['base']):>8} → {pct(c['tuned']):<6}({c['n']} ejemplos)")
    print(f"\n{result['reason']}")
    if result["activated"]:
        print(f"Activado: {result['checkpoint'].name}. Reiniciá el server para usarlo.")
    else:
        print(f"No se activó. El checkpoint quedó guardado en {result['checkpoint']}.")


def main() -> int:
    # The report contains "→"; a redirected stdout on Windows defaults to cp1252, which can't encode it.
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--epochs", type=int, default=TrainConfig.epochs)
    parser.add_argument("--no-activate", action="store_true", help="save the checkpoint without putting it in use")
    args = parser.parse_args()

    try:
        result = run(
            ExampleStore(DEFAULT_PATH),
            MODELS_DIR,
            threshold=float(os.environ.get("FLAG_THRESHOLD", "0.7")),
            config=TrainConfig(epochs=args.epochs),
            activate=not args.no_activate,
            device=os.environ.get("LAYA_DEVICE") or None,
        )
    except NotEnoughData as e:
        print(e)
        return 1
    print_report(result)
    return 0


if __name__ == "__main__":
    sys.exit(main())
