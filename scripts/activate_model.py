"""Choose which multilingual checkpoint the app uses.

    uv run python scripts/activate_model.py --list
    uv run python scripts/activate_model.py models/ft-20261001-120000
    uv run python scripts/activate_model.py --base

Restart the server afterwards.
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.checkpoints import MODELS_DIR, ActiveCheckpoint  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("checkpoint", nargs="?", type=Path)
    group.add_argument("--base", action="store_true", help="go back to Laya's original multilingual checkpoint")
    group.add_argument("--list", action="store_true", help="show the fine-tuned checkpoints")
    args = parser.parse_args()
    active = ActiveCheckpoint(MODELS_DIR)

    if args.list:
        current = active.get()
        for meta_path in sorted(MODELS_DIR.glob("*/training_meta.json")):
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            m = meta["metrics"]
            mark = "*" if current and current["path"] == meta_path.parent else " "
            print(f"{mark} {meta_path.parent.name}  F1 {m['base']['f1']:.2f} → {m['tuned']['f1']:.2f}  "
                  f"({meta['counts']['train']} ejemplos de entrenamiento)")
        print("  (* = activo)" if current else "  Ninguno activo: se usa el modelo original.")
        return 0

    if args.base:
        active.deactivate()
        print("Listo: la app va a usar el modelo multilingüe original. Reiniciá el server.")
        return 0

    path = args.checkpoint.resolve()
    meta_path = path / "training_meta.json"
    if MODELS_DIR.resolve() not in path.parents:
        print(f"El checkpoint tiene que estar dentro de {MODELS_DIR}.")
        return 1
    if not meta_path.exists():
        print(f"{path} no es un checkpoint de este proyecto (falta training_meta.json).")
        return 1
    active.activate(path, json.loads(meta_path.read_text(encoding="utf-8"))["metrics"])
    print(f"Listo: la app va a usar {path.name}. Reiniciá el server.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
