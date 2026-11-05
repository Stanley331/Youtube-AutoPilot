"""history.json: every video the pipeline has produced, so topics never repeat."""
from datetime import date

from .config import HISTORY_FILE
from .util import read_json, write_json


def load() -> list[dict]:
    if HISTORY_FILE.exists():
        return read_json(HISTORY_FILE)
    return []


def add(slug: str, title: str, topic: str, kind: str = "", fmt: str = "short") -> None:
    entries = [e for e in load() if e["slug"] != slug]
    entries.append({"slug": slug, "title": title, "topic": topic, "kind": kind, "format": fmt,
                    "created": date.today().isoformat(), "youtube_id": None})
    write_json(HISTORY_FILE, entries)


def recent_kinds(count: int = 6) -> list[str]:
    """Content types of the latest videos, newest last, for alternating story and fact."""
    return [e.get("kind", "") for e in load()[-count:]]


def unpublished() -> list[dict]:
    return [e for e in load() if not e.get("youtube_id")]


def set_voice(slug: str, engine: str, name: str) -> None:
    """Record the narrator so the voice pool can rotate away from recent ones."""
    entries = load()
    for e in entries:
        if e["slug"] == slug:
            e["voice"] = {"engine": engine, "name": name}
    write_json(HISTORY_FILE, entries)


def mark_published(slug: str, youtube_id: str, publish_at: str | None = None) -> None:
    entries = load()
    for e in entries:
        if e["slug"] == slug:
            e["youtube_id"] = youtube_id
            if publish_at:
                e["publish_at"] = publish_at
    write_json(HISTORY_FILE, entries)


def past_titles() -> list[str]:
    return [e["title"] for e in load()]
