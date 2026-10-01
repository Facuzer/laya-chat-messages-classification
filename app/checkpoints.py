"""Which fine-tuned checkpoint, if any, the app loads in place of Laya's multilingual one."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from app.jsonfile import read_json, write_json_atomic

ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = ROOT / "models"


class ActiveCheckpoint:
    """`<models_dir>/active.json` names the checkpoint in use. Without it the app uses the base model."""

    def __init__(self, models_dir: Path = MODELS_DIR):
        self.models_dir = Path(models_dir)
        self.file = self.models_dir / "active.json"

    def get(self) -> dict | None:
        data = read_json(self.file, None)
        if data is None:
            return None
        path = self.models_dir / data["path"]
        if not path.is_dir():
            return None
        return {**data, "path": path}

    def activate(self, path: Path, metrics: dict) -> None:
        relative = Path(path).resolve().relative_to(self.models_dir.resolve())
        activated_at = datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")
        write_json_atomic(self.file, {"path": relative.as_posix(), "activated_at": activated_at, "metrics": metrics})

    def deactivate(self) -> None:
        self.file.unlink(missing_ok=True)


def questions_match(checkpoint: Path, questions: dict) -> bool:
    """A checkpoint learned its answers against one wording of the questions."""
    meta = read_json(Path(checkpoint) / "training_meta.json", {})
    return meta.get("questions") == questions
