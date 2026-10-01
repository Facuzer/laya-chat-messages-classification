"""Add candidate examples for fine-tuning from a JSONL or CSV file.

Rows need a `text`, and may carry a `label` (insult / clean) and a `category`. Without
--accept-labels every row arrives pending, with its label as a suggestion, for review in
/training.html.

    uv run python scripts/import_examples.py datasets/generated_rioplatense_v1.jsonl --source generated
    uv run python scripts/import_examples.py mis_mensajes.csv
    uv run python scripts/import_examples.py ya_etiquetados.csv --accept-labels
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.examples import DEFAULT_PATH, ExampleStore, parse_import_lines  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("file", type=Path)
    parser.add_argument("--source", choices=["imported", "generated"], default="imported")
    parser.add_argument("--accept-labels", action="store_true", help="trust the file's labels and skip review")
    args = parser.parse_args()

    fmt = {".jsonl": "jsonl", ".csv": "csv"}.get(args.file.suffix.lower())
    if fmt is None:
        print(f"Formato no soportado: {args.file.suffix}. Usá .jsonl o .csv.")
        return 1
    try:
        rows = parse_import_lines(args.file.read_text(encoding="utf-8-sig").splitlines(), fmt)
        result = ExampleStore(DEFAULT_PATH).add_candidates(rows, source=args.source, accept_labels=args.accept_labels)
    except ValueError as e:
        print(f"No se importó nada: {e}")
        return 1

    state = "etiquetados" if args.accept_labels else "pendientes de revisión"
    print(f"Importados {result['added']} ejemplos ({state}). Duplicados salteados: {result['duplicates']}.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
