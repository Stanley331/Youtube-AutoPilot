"""Publishing slots: when the next finished video should go live.

YouTube accepts a future ``publishAt`` on a private upload and flips it to public
at that time, so a batch rendered tonight can trickle out over the week. Slots
come from ``youtube.schedule`` in config.yaml (daily times in a timezone); ones
already promised to earlier uploads are read from history.json and skipped.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

from . import history

DEFAULTS = {
    "times": ["18:00"],          # local times, one video per slot
    "timezone": "UTC",
    "lead_minutes": 60,          # never schedule closer than this to now
}


def settings(cfg: dict) -> dict:
    merged = dict(DEFAULTS)
    merged.update((cfg.get("youtube", {}) or {}).get("schedule") or {})
    times = merged["times"] if isinstance(merged["times"], list) else [merged["times"]]
    merged["times"] = sorted(str(t) for t in times) or DEFAULTS["times"]
    return merged


def tz(cfg: dict) -> ZoneInfo:
    return ZoneInfo(settings(cfg)["timezone"])


def to_rfc3339(moment: datetime) -> str:
    """UTC with a Z suffix, the form the YouTube API wants."""
    return moment.astimezone(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def parse_local(text: str, cfg: dict) -> datetime:
    """'2026-11-05 09:00' or ISO 8601, read in the configured timezone unless it carries one."""
    text = text.strip().replace("Z", "+00:00")
    moment = datetime.fromisoformat(text)
    if moment.tzinfo is None:
        moment = moment.replace(tzinfo=tz(cfg))
    return moment


def taken() -> set[str]:
    """Slots already promised to uploaded videos."""
    return {e["publish_at"] for e in history.load() if e.get("publish_at")}


def next_slots(cfg: dict, count: int = 1, now: datetime | None = None,
               reserved: set[str] | None = None) -> list[datetime]:
    """The next ``count`` free slots after now plus the lead time."""
    s = settings(cfg)
    zone = tz(cfg)
    now = (now or datetime.now(timezone.utc)).astimezone(zone)
    earliest = now + timedelta(minutes=int(s["lead_minutes"]))
    used = set(taken()) | set(reserved or ())
    slots: list[datetime] = []
    day = earliest.date()
    for _ in range(400):  # about a year of days; more than any queue
        for hhmm in s["times"]:
            hour, minute = (int(x) for x in hhmm.split(":")[:2])
            moment = datetime(day.year, day.month, day.day, hour, minute, tzinfo=zone)
            if moment < earliest or to_rfc3339(moment) in used:
                continue
            slots.append(moment)
            used.add(to_rfc3339(moment))
            if len(slots) == count:
                return slots
        day += timedelta(days=1)
    return slots


def describe(moment: datetime, cfg: dict) -> str:
    local = moment.astimezone(tz(cfg))
    return f"{local:%a %Y-%m-%d %H:%M} {settings(cfg)['timezone']}"
