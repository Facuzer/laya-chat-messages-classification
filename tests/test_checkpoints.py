import json

from app.checkpoints import ActiveCheckpoint, questions_match


def make_checkpoint(models_dir, name="ft-1", questions=None):
    path = models_dir / name
    path.mkdir(parents=True)
    (path / "training_meta.json").write_text(json.dumps({"questions": questions or {"q": 1}}), encoding="utf-8")
    return path


def test_no_active_checkpoint_by_default(tmp_path):
    assert ActiveCheckpoint(tmp_path).get() is None


def test_activate_then_get_returns_path_and_metrics(tmp_path):
    path = make_checkpoint(tmp_path)
    active = ActiveCheckpoint(tmp_path)

    active.activate(path, {"f1": 0.8})

    got = ActiveCheckpoint(tmp_path).get()
    assert got["path"] == path
    assert got["metrics"] == {"f1": 0.8}
    assert got["activated_at"]


def test_deactivate_returns_to_base_model(tmp_path):
    active = ActiveCheckpoint(tmp_path)
    active.activate(make_checkpoint(tmp_path), {})

    active.deactivate()

    assert active.get() is None


def test_deactivate_without_active_checkpoint_is_harmless(tmp_path):
    ActiveCheckpoint(tmp_path).deactivate()


def test_active_checkpoint_whose_folder_was_deleted_counts_as_none(tmp_path):
    path = make_checkpoint(tmp_path)
    active = ActiveCheckpoint(tmp_path)
    active.activate(path, {})

    (path / "training_meta.json").unlink()
    path.rmdir()

    assert active.get() is None


def test_questions_match_compares_with_training_meta(tmp_path):
    path = make_checkpoint(tmp_path, questions={"insult": {"type": "choice"}})

    assert questions_match(path, {"insult": {"type": "choice"}}) is True
    assert questions_match(path, {"insult": {"type": "noul"}}) is False
