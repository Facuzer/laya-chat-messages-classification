import json

from app.store import MessageStore


def test_missing_file_lists_empty(tmp_path):
    store = MessageStore(tmp_path / "messages.json")

    assert store.list() == []


def test_add_creates_parent_directory_and_file(tmp_path):
    path = tmp_path / "data" / "messages.json"

    MessageStore(path).add({"id": "a", "text": "hola"})

    assert json.loads(path.read_text(encoding="utf-8")) == [{"id": "a", "text": "hola"}]


def test_list_returns_messages_in_insertion_order(tmp_path):
    store = MessageStore(tmp_path / "messages.json")

    store.add({"id": "a", "text": "primero"})
    store.add({"id": "b", "text": "segundo"})

    assert [m["id"] for m in store.list()] == ["a", "b"]


def test_messages_persist_across_instances(tmp_path):
    path = tmp_path / "messages.json"
    MessageStore(path).add({"id": "a", "text": "¿seguís ahí? ñandú"})

    assert MessageStore(path).list() == [{"id": "a", "text": "¿seguís ahí? ñandú"}]


def test_file_keeps_non_ascii_text_readable(tmp_path):
    path = tmp_path / "messages.json"

    MessageStore(path).add({"id": "a", "text": "ñandú"})

    assert "ñandú" in path.read_text(encoding="utf-8")
