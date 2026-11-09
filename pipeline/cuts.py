"""Shorts cut from a finished long video.

A long video's narration, media choices and camera plan are already on disk,
so a Short is a slice of consecutive scenes re-rendered vertically: the same
audio files, the same chosen clip or stills per scene (centre-cropped to 9:16),
Shorts-sized captions and its own title and thumbnail. The model picks the
slices (the hook, and a self-contained chapter or two); a heuristic takes over
when it cannot. Each cut becomes its own project folder, ``<slug>-short-<n>``,
that the UI lists and ``run.py publish`` uploads like any other.
"""
from __future__ import annotations

import copy
import shutil
from pathlib import Path

from . import doodle, formats, history
from .config import OUTPUT_DIR
from .llm import generate_json
from .util import log, read_json, write_json

DEFAULTS = {"count": 2, "min_seconds": 25, "max_seconds": 58}

CUTS_SCHEMA = {
    "type": "object",
    "properties": {
        "cuts": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "first_scene": {"type": "integer"},
                    "last_scene": {"type": "integer"},
                    "title": {"type": "string"},
                    "thumbnail_text": {"type": "string"},
                    "description": {"type": "string"},
                },
                "required": ["first_scene", "last_scene", "title", "thumbnail_text", "description"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["cuts"],
    "additionalProperties": False,
}


def settings(cfg: dict) -> dict:
    merged = dict(DEFAULTS)
    merged.update(cfg.get("cuts", {}) or {})
    return merged


def _span(durations: list[float], first: int, last: int) -> float:
    return float(sum(durations[first:last + 1]))


def _heuristic_cuts(script: dict, durations: list[float], cfg: dict, count: int,
                    used: set | None = None) -> list[dict]:
    """The hook from scene 0, then from each chapter start, each stretched to the length window."""
    s = settings(cfg)
    starts = [0] + [int(c.get("first_scene", 0)) for c in script.get("chapters", []) or []]
    # Any scene start works as a fallback once the chapter starts are used up.
    starts += [i for i in range(len(durations)) if i not in starts]
    cuts, used = [], set(used or ())
    for first in starts:
        if first in used or first >= len(durations):
            continue
        last = first
        while last + 1 < len(durations) and _span(durations, first, last + 1) <= s["max_seconds"]:
            last += 1
        if _span(durations, first, last) < s["min_seconds"]:
            continue
        title = script.get("title", "")
        for chapter in script.get("chapters", []) or []:
            if int(chapter.get("first_scene", -1)) == first:
                title = f"{chapter.get('title', '')}: {script.get('title', '')}"
        cuts.append({"first_scene": first, "last_scene": last, "title": title[:60],
                     "thumbnail_text": script.get("thumbnail_text", ""),
                     "description": script.get("description", "")})
        used.update(range(first, last + 1))
        if len(cuts) >= count:
            break
    return cuts


def plan_cuts(script: dict, durations: list[float], cfg: dict, count: int | None = None) -> list[dict]:
    """Up to ``count`` scene ranges that stand alone as Shorts, with their own titles."""
    s = settings(cfg)
    count = max(1, int(count or s["count"]))
    lines = []
    for i, (scene, d) in enumerate(zip(script.get("scenes", []), durations)):
        lines.append(f"{i} ({d:.0f}s): {scene.get('narration', '')}")
    system = (
        "You pick excerpts of a long YouTube video to publish as Shorts. A good excerpt is a run of "
        "consecutive scenes that makes sense without the rest of the video: it opens on a hook or a "
        "question and ends on a payoff. The first excerpt should start at scene 0 (the hook). "
        f"Each excerpt must last between {s['min_seconds']} and {s['max_seconds']} seconds in total; "
        "add up the scene durations given. Excerpts must not overlap."
    )
    prompt = (
        f"Video: {script.get('title', '')}\n\nScenes (index, spoken length, narration):\n" + "\n".join(lines)
        + f"\n\nReturn up to {count} excerpts, best first. For each: first_scene and last_scene (inclusive "
        "indexes), a Shorts title under 60 characters that works on its own, thumbnail_text of at most "
        "4 words in capitals, and a one-sentence description."
    )
    cuts: list[dict] = []
    try:
        result = generate_json(system, prompt, CUTS_SCHEMA, effort="medium")
        raw = result.get("cuts") or []
    except Exception as exc:
        log(f"cuts: the model could not pick excerpts ({type(exc).__name__}: {exc}); using chapter starts")
        raw = []
    used: set[int] = set()
    for item in raw:
        try:
            first, last = int(item.get("first_scene")), int(item.get("last_scene"))
        except (TypeError, ValueError):
            continue
        if not (0 <= first <= last < len(durations)):
            continue
        # Trim or extend to the length window where the model misjudged it.
        while last > first and _span(durations, first, last) > s["max_seconds"]:
            last -= 1
        while last + 1 < len(durations) and _span(durations, first, last) < s["min_seconds"] \
                and _span(durations, first, last + 1) <= s["max_seconds"]:
            last += 1
        span = _span(durations, first, last)
        if span < s["min_seconds"] or span > s["max_seconds"] or used & set(range(first, last + 1)):
            continue
        cuts.append({"first_scene": first, "last_scene": last,
                     "title": str(item.get("title") or script.get("title", ""))[:60],
                     "thumbnail_text": str(item.get("thumbnail_text") or script.get("thumbnail_text", "")),
                     "description": str(item.get("description") or "")})
        used.update(range(first, last + 1))
        if len(cuts) >= count:
            break
    if len(cuts) < count:
        cuts += _heuristic_cuts(script, durations, cfg, count - len(cuts), used)
    return sorted(cuts, key=lambda c: c["first_scene"])


def _scene_durations(out_dir: Path, count: int) -> list[float]:
    from .voice import load_scene_audio
    durations = []
    for i in range(count):
        audio = load_scene_audio(out_dir / "audio" / f"scene_{i:02d}.json")
        if audio is None:
            raise RuntimeError(f"scene {i + 1} has no narration in {out_dir}; finish the long video first")
        durations.append(audio.duration)
    return durations


def cut_shorts(parent_dir: Path, cfg: dict, count: int | None = None,
               cuts: list[dict] | None = None, visuals: str | None = None) -> list[Path]:
    """Render the planned excerpts of ``parent_dir`` as vertical Shorts; returns their folders."""
    from .assemble import assemble
    from .captions import build_ass
    from .motion import plan_to_json
    from .thumbnail import make_thumbnail
    from .visuals import clips_for_scenes
    from .voice import load_scene_audio

    parent_dir = Path(parent_dir)
    script = read_json(parent_dir / "script.json")
    plan_path = parent_dir / "clip_plan.json"
    if not plan_path.exists():
        raise RuntimeError(f"no clip_plan.json in {parent_dir}; render the long video first")
    parent_plan = read_json(plan_path)
    durations = _scene_durations(parent_dir, len(script.get("scenes", [])))
    cuts = cuts or plan_cuts(script, durations, cfg, count)
    if not cuts:
        raise RuntimeError("no excerpt of the right length was found")
    short_cfg = formats.apply(cfg, "short")
    source = visuals or _project_visuals(parent_dir, cfg)
    candidates = {}
    if (parent_dir / "media_candidates.json").exists():
        candidates = read_json(parent_dir / "media_candidates.json")
    parent_entry = next((e for e in history.load() if e.get("slug") == parent_dir.name), {})
    link = f"https://youtu.be/{parent_entry['youtube_id']}" if parent_entry.get("youtube_id") else ""

    made: list[Path] = []
    for n, cut in enumerate(cuts, 1):
        first, last = int(cut["first_scene"]), int(cut["last_scene"])
        child = OUTPUT_DIR / f"{parent_dir.name}-short-{n}"
        child.mkdir(parents=True, exist_ok=True)
        (child / "audio").mkdir(exist_ok=True)
        scenes = copy.deepcopy(script["scenes"][first:last + 1])
        description = str(cut.get("description") or "").strip()
        full = f"Watch the full video: {link}" if link else f"From the full video \"{script.get('title', '')}\"."
        child_script = {
            "format": "short", "parent": parent_dir.name, "cut": {"first_scene": first, "last_scene": last},
            "content_type": script.get("content_type", "fact"),
            "title": str(cut.get("title") or script.get("title", ""))[:100],
            "alternative_titles": [], "thumbnail_text": str(cut.get("thumbnail_text") or script.get("thumbnail_text", "")),
            "description": f"{description}\n\n{full}".strip(),
            "tags": list(dict.fromkeys(list(script.get("tags", [])) + ["shorts"]))[:30],
            "main_character": script.get("main_character", ""),
            "art_style": script.get("art_style", ""), "voice": script.get("voice"),
            "scenes": scenes, "claims_to_verify": [],
        }
        if doodle.is_doodle(script):
            # Doodle cuts keep a drawing per sentence: the parent's images, re-framed vertically.
            src = script.get("doodle_source") or {}
            child_script["doodle_source"] = {"dir": src.get("dir") or str(parent_dir),
                                             "first_scene": int(src.get("first_scene", 0)) + first}
        # Narration: the parent's files, renumbered from scene 0.
        audios = []
        for k, i in enumerate(range(first, last + 1)):
            src_meta = parent_dir / "audio" / f"scene_{i:02d}.json"
            saved = read_json(src_meta)
            src_mp3 = parent_dir / "audio" / saved.get("path", f"scene_{i:02d}.mp3")
            dst_mp3 = child / "audio" / f"scene_{k:02d}.mp3"
            shutil.copyfile(src_mp3, dst_mp3)
            saved["path"] = dst_mp3.name
            write_json(child / "audio" / f"scene_{k:02d}.json", saved)
            audios.append(load_scene_audio(child / "audio" / f"scene_{k:02d}.json"))
        # Media: the parent's choice per scene, plus its ranked candidates for the still cuts.
        selections, child_candidates = {}, {}
        for k, i in enumerate(range(first, last + 1)):
            entries = parent_plan[i] if i < len(parent_plan) else []
            if entries:
                selections[k] = str(entries[0][0])
            if str(i) in candidates:
                child_candidates[str(k)] = candidates[str(i)]
        if child_candidates:
            write_json(child / "media_candidates.json", child_candidates)
        write_json(child / "media_plan.json", {"selections": {str(k): v for k, v in selections.items()},
                                              "custom_media": {}})
        write_json(child / "ui_state.json", {"visuals": source})
        from .script import save_script
        save_script(child_script, child)
        log(f"cuts: short {n}: scenes {first + 1}-{last + 1} ({_span(durations, first, last):.0f}s) '{child_script['title']}'")
        if doodle.is_doodle(child_script):
            frames = doodle.compose_frames(child, child_script, short_cfg)
            plan = doodle.clip_plan(child_script, audios, frames, short_cfg)
        else:
            plan = clips_for_scenes(scenes, [a.duration for a in audios], child, short_cfg, source,
                                    script_context=child_script, selections=selections)
        write_json(child / "clip_plan.json", plan_to_json(plan))
        build_ass(audios, child / "captions.ass", short_cfg)
        final = assemble(child, plan, audios, short_cfg)
        make_thumbnail(final, child_script["thumbnail_text"], child / "thumbnail.jpg", short_cfg)
        history.add(child.name, child_script["title"], script.get("title", ""), kind="short")
        if script.get("voice"):
            history.set_voice(child.name, script["voice"].get("engine", ""), script["voice"].get("name", ""))
        made.append(child)
    return made


def _project_visuals(out_dir: Path, cfg: dict) -> str:
    path = out_dir / "ui_state.json"
    if path.exists():
        try:
            source = read_json(path).get("visuals")
            if source:
                return str(source)
        except (OSError, ValueError):
            pass
    return str(cfg["visuals"]["source"])
