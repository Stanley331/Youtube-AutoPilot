"""Everything the studio UI does to disk and to the pipeline, with no Streamlit in it.

The page (``ui/app.py``) only draws; each action here is a plain function that
reads and writes ``output/<slug>/`` and returns plain values, so a step's state
is always what is on disk. That is what lets the wizard resume a project where
it was left, from any browser, after any crash.
"""
from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
import time
import traceback
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pipeline import diffusion, doodle, formats, history, kaggle_images, lofi, lofi_library, lofi_music  # noqa: E402
from pipeline.config import load_config  # noqa: E402
from pipeline.motion import plan_to_json  # noqa: E402
from pipeline.ranking import LOW_MATCH_SCORE, ensure_search_terms  # noqa: E402
from pipeline.util import duration_of, log, read_json, slugify, write_json  # noqa: E402
from pipeline.visuals import (SOURCES, clips_for_scenes, deserialize_candidates,  # noqa: E402
                              extract_thumbnail, fetch_candidates_for_scenes, serialize_candidates)
from pipeline.voice import apply_scene_speed, load_scene_audio, preview_voice  # noqa: E402
from pipeline.voices import ENGINES, all_voices, describe, pool_for  # noqa: E402

OUTPUT_DIR = ROOT / "output"
THUMB_CACHE = OUTPUT_DIR / "_media_thumbs"
_VENV_PYTHON = ROOT / ".venv" / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
PYTHON_EXE = str(_VENV_PYTHON) if _VENV_PYTHON.exists() else sys.executable
RUN_PY = ROOT / "run.py"

__all__ = [
    "ROOT", "OUTPUT_DIR", "SOURCES", "ENGINES", "LOW_MATCH_SCORE", "cfg", "project_cfg", "describe",
    "all_voices", "pool_for", "preview_voice", "diffusion", "lofi", "lofi_music", "formats", "doodle",
]

KIND_LABELS = {"short": "Short", "long": "Long video", "doodle": "Doodle explainer", "lofi": "Lofi music"}
KIND_ICONS = {"short": "🎬", "long": "📺", "doodle": "✏️", "lofi": "🎧"}


class ActionError(RuntimeError):
    """An action failed; ``detail`` holds the log or traceback for the Details panel."""

    def __init__(self, message: str, detail: str = ""):
        super().__init__(message)
        self.detail = detail


def detail_of(exc: BaseException) -> str:
    if isinstance(exc, ActionError) and exc.detail:
        return exc.detail
    return "".join(traceback.format_exception(exc))[-4000:]


# ---------------------------------------------------------------------------
# Config, processes
# ---------------------------------------------------------------------------

def cfg() -> dict:
    return load_config()


def project_cfg(slug: str) -> dict:
    return formats.for_project(load_config(), OUTPUT_DIR / slug)


def run_cli(cmd: list[str]) -> tuple[int, str, str]:
    env = dict(os.environ, PYTHONUTF8="1", PYTHONIOENCODING="utf-8")
    result = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True,
                            encoding="utf-8", errors="replace", env=env)
    return result.returncode, result.stdout or "", result.stderr or ""


def run_cli_or_raise(cmd: list[str], what: str) -> str:
    code, out, err = run_cli(cmd)
    if code != 0:
        tail = (out + "\n" + err).strip()[-5000:]
        # The last log line usually names the cause; surface it in the headline.
        lines = [ln for ln in tail.splitlines() if ln.strip()]
        headline = lines[-1][:220] if lines else f"exit code {code}"
        raise ActionError(f"{what} failed: {headline}", tail)
    return out


def start_background(cmd: list[str], log_path: Path) -> None:
    """Run a command detached; its output streams to ``log_path`` and ends with ``exit <code>``."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, PYTHONUTF8="1", PYTHONUNBUFFERED="1")
    log_path.write_text(f"$ {' '.join(cmd)}\n", encoding="utf-8")
    wrapper = ("import subprocess, sys; log, cmd = sys.argv[1], sys.argv[2:]; "
               "h = open(log, 'a', encoding='utf-8'); "
               "r = subprocess.run(cmd, stdout=h, stderr=subprocess.STDOUT); "
               "h.write('\\nexit %d\\n' % r.returncode); h.close()")
    flags = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
    proc = subprocess.Popen([PYTHON_EXE, "-c", wrapper, str(log_path), *cmd], cwd=str(ROOT),
                            stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                            env=env, creationflags=flags)
    log_path.with_suffix(".pid").write_text(str(proc.pid), encoding="utf-8")


def _pid_alive(pid: int) -> bool:
    """Whether a process is still running (never signals it; os.kill would end it on Windows)."""
    if os.name == "nt":
        import ctypes
        kernel32 = ctypes.windll.kernel32
        handle = kernel32.OpenProcess(0x1000, False, int(pid))   # PROCESS_QUERY_LIMITED_INFORMATION
        if not handle:
            return False
        code = ctypes.c_ulong()
        ok = kernel32.GetExitCodeProcess(handle, ctypes.byref(code))
        kernel32.CloseHandle(handle)
        return bool(ok) and code.value == 259                      # STILL_ACTIVE
    try:
        os.kill(int(pid), 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True


def log_state(log_path: Path, stale_minutes: int = 15) -> str:
    """none | running | done | failed, from the log's exit line, its process, and its age."""
    if not log_path.exists():
        return "none"
    text = log_path.read_text(encoding="utf-8", errors="replace")
    finished = re.search(r"^exit (\d+)\s*$", text, flags=re.MULTILINE)
    if not finished:
        # A job whose process is gone (the studio restarted, the PC slept) failed, however fresh its log.
        pid_file = log_path.with_suffix(".pid")
        if pid_file.exists():
            try:
                if not _pid_alive(int(pid_file.read_text(encoding="utf-8").strip())):
                    return "failed"
            except (OSError, ValueError):
                pass
        return "running" if time.time() - log_path.stat().st_mtime < stale_minutes * 60 else "failed"
    return "done" if finished.group(1) == "0" else "failed"


def log_tail(log_path: Path, chars: int = 3000) -> str:
    if not log_path.exists():
        return ""
    return log_path.read_text(encoding="utf-8", errors="replace")[-chars:]


# ---------------------------------------------------------------------------
# Projects on disk
# ---------------------------------------------------------------------------

def read_script(slug: str) -> dict | None:
    path = OUTPUT_DIR / slug / "script.json"
    if not path.exists():
        return None
    try:
        return read_json(path)
    except (OSError, ValueError):
        return None


def write_script(slug: str, script: dict) -> None:
    from pipeline.script import save_script
    save_script(script, OUTPUT_DIR / slug)


def ui_state(slug: str) -> dict:
    path = OUTPUT_DIR / slug / "ui_state.json"
    if path.exists():
        try:
            return read_json(path)
        except (OSError, ValueError):
            return {}
    return {}


def set_ui_state(slug: str, **values) -> None:
    state = ui_state(slug)
    state.update(values)
    write_json(OUTPUT_DIR / slug / "ui_state.json", state)


def history_entry(slug: str) -> dict:
    return next((e for e in history.load() if e.get("slug") == slug), {})


def unique_slug(base: str) -> str:
    slug, n = base, 2
    while (OUTPUT_DIR / slug).exists():
        slug = f"{base}-{n}"
        n += 1
    return slug


def scene_audios(slug: str, count: int) -> list:
    audio_dir = OUTPUT_DIR / slug / "audio"
    audios = []
    for i in range(count):
        audio = load_scene_audio(audio_dir / f"scene_{i:02d}.json")
        if audio is None or not (audio_dir / audio.path).exists():
            break
        audios.append(audio)
    return audios


def load_candidates(slug: str) -> dict[int, list[dict]]:
    path = OUTPUT_DIR / slug / "media_candidates.json"
    if not path.exists():
        return {}
    try:
        return deserialize_candidates(read_json(path))
    except (OSError, ValueError):
        return {}


def save_candidates(slug: str, candidates: dict[int, list[dict]]) -> None:
    write_json(OUTPUT_DIR / slug / "media_candidates.json", serialize_candidates(candidates))


def load_media_plan(slug: str) -> tuple[dict[str, str], dict[str, str]]:
    path = OUTPUT_DIR / slug / "media_plan.json"
    if not path.exists():
        return {}, {}
    try:
        data = read_json(path)
    except (OSError, ValueError):
        return {}, {}
    selections = {str(k): str(v) for k, v in (data.get("selections") or {}).items() if v}
    custom = {str(k): str(v) for k, v in (data.get("custom_media") or {}).items() if v}
    return selections, custom


def save_media_plan(slug: str, selections: dict, custom_media: dict) -> None:
    write_json(OUTPUT_DIR / slug / "media_plan.json", {
        "selections": {str(k): str(v) for k, v in selections.items() if v},
        "custom_media": {str(k): str(v) for k, v in custom_media.items() if v},
    })


def chosen_media(slug: str, scene_count: int) -> dict[int, str]:
    """Scene index -> chosen file, uploads winning over picks; missing files are dropped."""
    selections, custom = load_media_plan(slug)
    chosen = {}
    for i in range(scene_count):
        path = custom.get(str(i)) or selections.get(str(i))
        if path and Path(path).exists():
            chosen[i] = path
    return chosen


def progress(slug: str) -> dict:
    """What exists on disk for a project; every step reads its state from this."""
    out_dir = OUTPUT_DIR / slug
    script = read_script(slug) or {}
    fmt = formats.of_script(script)
    scenes = script.get("scenes", [])
    state = ui_state(slug)
    entry = history_entry(slug)
    info = {
        "slug": slug, "format": fmt, "script": bool(script), "title": script.get("title", slug),
        "scene_count": len(scenes), "video": (out_dir / "final.mp4").exists(),
        "thumbnail": (out_dir / "thumbnail.jpg").exists(),
        "youtube_id": entry.get("youtube_id"), "publish_at": entry.get("publish_at"),
        "render": log_state(out_dir / "render.log"), "state": state,
    }
    if fmt == "lofi":
        image = script.get("scene_image")
        info["scene_image"] = image if image and Path(image).exists() else None
        info["tracks"] = script.get("tracks", [])
        info["settings"] = lofi.project_settings(script, project_cfg(slug)) if script else {}
        # Projects made before the wizard carry no step flags (and no "kind"); infer them from what exists.
        if "kind" not in state:
            if info["scene_image"]:
                state.setdefault("scene_ok", True)
                state.setdefault("look_ok", True)
            if info["tracks"] or info["render"] != "none":
                state.setdefault("music_ok", True)
    elif fmt == "doodle":
        audios = scene_audios(slug, len(scenes)) if scenes else []
        info["voice"] = bool(scenes) and len(audios) == len(scenes)
        info["durations"] = [a.duration for a in audios]
        info["images"] = doodle.image_status(out_dir, script)
        info["drawing"] = log_state(out_dir / "drawing.log", stale_minutes=20)
        info["selected"] = bool(scenes) and not info["images"]["missing"]
        info["candidates"] = info["images"]["done"] > 0
        info["chosen"] = {}
        backend = state.get("visuals")
        info["visuals"] = backend if backend in doodle.BACKENDS else doodle.settings(cfg())["images"]
    else:
        audios = scene_audios(slug, len(scenes)) if scenes else []
        info["voice"] = bool(scenes) and len(audios) == len(scenes)
        info["durations"] = [a.duration for a in audios]
        info["candidates"] = bool(load_candidates(slug))
        chosen = chosen_media(slug, len(scenes))
        info["chosen"] = chosen
        info["selected"] = bool(scenes) and len(chosen) == len(scenes)
        info["visuals"] = state.get("visuals") or cfg()["visuals"]["source"]
        if "kind" not in state:
            if info["voice"]:
                state.setdefault("script_ok", True)
            if info["video"]:
                state.setdefault("visuals_ok", True)
    return info


def status_line(p: dict) -> str:
    """One phrase for a library card."""
    if p["youtube_id"]:
        when = f", public at {p['publish_at'][:16].replace('T', ' ')}" if p.get("publish_at") else ""
        return f"Published{when}"
    if p["format"] == "lofi":
        if p["render"] == "running":
            return "Rendering in the background"
        if p["video"]:
            return "Ready to publish"
        if p["render"] == "failed":
            return "Render failed; open to see why"
        if p.get("scene_image"):
            return "Scene drawn; pick music and render"
        return "Planned; draw the scene next"
    if p["format"] == "doodle":
        if p["render"] == "running":
            return "Rendering in the background"
        if p["video"]:
            return "Ready to publish"
        if p["drawing"] == "running":
            return f"Drawing on {p['visuals']}"
        if p["drawing"] == "failed":
            return "Drawing failed; open to see why"
        if p["selected"] and p.get("voice"):
            return "Drawings ready; render next"
        if p["images"]["done"]:
            return f"{p['images']['done']}/{p['images']['needed']} drawings"
        return "Narrated; draw the pictures next" if p.get("voice") else "Script written; narrate next"
    if p["video"]:
        return "Ready to publish"
    if p.get("selected"):
        return "Visuals chosen; ready to render"
    if p.get("candidates"):
        return "Choosing visuals"
    if p.get("voice"):
        return "Narrated; pick the visuals next"
    return "Script written; narrate next"


def list_projects() -> list[dict]:
    if not OUTPUT_DIR.exists():
        return []
    dirs = [p for p in OUTPUT_DIR.iterdir()
            if p.is_dir() and not p.name.startswith("_") and (p / "script.json").exists()]
    dirs.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    projects = []
    for out_dir in dirs:
        try:
            p = progress(out_dir.name)
        except Exception as exc:  # one broken folder must not hide the library
            log(f"ui: skipping {out_dir.name}: {type(exc).__name__}: {exc}")
            continue
        script = read_script(out_dir.name) or {}
        p["created"] = out_dir.name[:10] if re.match(r"\d{4}-\d{2}-\d{2}", out_dir.name) else ""
        p["idea"] = history_entry(out_dir.name).get("topic") or script.get("idea", "")
        p["parent"] = script.get("parent")
        p["card_image"] = str(out_dir / "thumbnail.jpg") if p["thumbnail"] else (p.get("scene_image") or "")
        projects.append(p)
    return projects


def delete_project(slug: str) -> None:
    shutil.rmtree(OUTPUT_DIR / slug)
    entries = [e for e in history.load() if e.get("slug") != slug]
    write_json(ROOT / "history.json", entries)


# ---------------------------------------------------------------------------
# Narrated videos: script, narration, visuals, render, publish
# ---------------------------------------------------------------------------

def _slug_from_output(stdout: str) -> str | None:
    m = re.search(r"output[\\/]([^\\/\r\n]+)[\\/]script\.md", stdout)
    return m.group(1).strip() if m else None


def create_script(idea: str, fmt: str, length: str, visuals: str) -> str:
    """Write the script for a new project (a minute for a Short, a few for a long video)."""
    out = run_cli_or_raise([PYTHON_EXE, str(RUN_PY), "make", "--idea", idea, "--format", fmt,
                            "--length", length, "--visuals", visuals, "--stop-after", "script"],
                           "Writing the script")
    slug = _slug_from_output(out)
    if not slug or not (OUTPUT_DIR / slug / "script.json").exists():
        raise ActionError("The script was written but its folder could not be found", out[-3000:])
    set_ui_state(slug, visuals=visuals, kind=length)
    return slug


JOBS_DIR = OUTPUT_DIR / "_jobs"


def start_script_job(idea: str, fmt: str, length: str, visuals: str) -> str:
    """Write a new project's script in the background; returns the log path to follow.

    A doodle or long script is an outline plus one request per chapter (several minutes,
    more when the model is busy and retries), so the page shows the log instead of a spinner.
    """
    JOBS_DIR.mkdir(parents=True, exist_ok=True)
    log_path = JOBS_DIR / f"script_{time.strftime('%Y%m%d_%H%M%S')}_{slugify(idea, 30)}.log"
    start_background([PYTHON_EXE, str(RUN_PY), "make", "--idea", idea, "--format", fmt, "--length", length,
                      "--visuals", visuals, "--stop-after", "script"], log_path)
    return str(log_path)


def script_job(log_path: str) -> dict:
    """state (running/done/failed), the new slug when done, elapsed seconds, and the log text."""
    path = Path(log_path)
    text = log_tail(path, 20000)
    state = log_state(path, stale_minutes=10)
    slug = _slug_from_output(text) if state == "done" else None
    if state == "done" and not (slug and (OUTPUT_DIR / slug / "script.json").exists()):
        state = "failed"
    started = path.with_suffix(".pid").stat().st_mtime if path.with_suffix(".pid").exists() else path.stat().st_mtime
    return {"state": state, "slug": slug, "elapsed": time.time() - started, "log": text}


def narrate(slug: str, engine: str, name: str = "") -> list[float]:
    cmd = [PYTHON_EXE, str(RUN_PY), "make", "--slug", slug, "--voice", engine, "--stop-after", "voice"]
    if name:
        cmd += ["--voice-name", name]
    run_cli_or_raise(cmd, "Narration")
    out_dir = OUTPUT_DIR / slug
    for stale in ("final.mp4", "final_1x.mp4"):
        (out_dir / stale).unlink(missing_ok=True)
    script = read_script(slug) or {}
    return [a.duration for a in scene_audios(slug, len(script.get("scenes", [])))]


def fetch_candidates(slug: str, visuals: str) -> dict[int, list[dict]]:
    script = read_script(slug)
    if not script:
        raise ActionError("No script.json in this project")
    out_dir = OUTPUT_DIR / slug
    cfg_ = project_cfg(slug)
    if visuals == "pixabay":
        script = ensure_search_terms(script, cfg_, out_dir)
    candidates = fetch_candidates_for_scenes(
        script["scenes"], out_dir, cfg_,
        candidates_per_scene=int(cfg_["visuals"].get("candidates_per_scene", 8)),
        source=visuals, script_context=script)
    save_candidates(slug, candidates)
    set_ui_state(slug, visuals=visuals)
    return candidates


def refetch_candidates(slug: str, visuals: str) -> dict[int, list[dict]]:
    """Replace every scene's candidates and clear picks; uploads are kept."""
    (OUTPUT_DIR / slug / "media_candidates.json").unlink(missing_ok=True)
    _, custom = load_media_plan(slug)
    save_media_plan(slug, {}, custom)
    return fetch_candidates(slug, visuals)


def regenerate_scene(slug: str, scene_index: int, visuals: str, page: int) -> list[dict]:
    script = read_script(slug)
    if not script or scene_index >= len(script.get("scenes", [])):
        return []
    out_dir = OUTPUT_DIR / slug
    cfg_ = project_cfg(slug)
    if visuals == "pixabay":
        script = ensure_search_terms(script, cfg_, out_dir)
    fetched = fetch_candidates_for_scenes(
        [script["scenes"][scene_index]], out_dir, cfg_,
        candidates_per_scene=int(cfg_["visuals"].get("candidates_per_scene", 8)),
        source=visuals, script_context=script, start_index=scene_index, page=page)
    new_cands = fetched.get(scene_index, [])
    if new_cands:
        candidates = load_candidates(slug)
        if visuals == "diffusion":
            seen = {str(c["path"]) for c in new_cands}
            new_cands = new_cands + [c for c in candidates.get(scene_index, []) if str(c["path"]) not in seen]
        candidates[scene_index] = new_cands
        save_candidates(slug, candidates)
    return new_cands


def select_media(slug: str, scene_index: int, path: str) -> None:
    selections, custom = load_media_plan(slug)
    selections[str(scene_index)] = path
    custom.pop(str(scene_index), None)
    save_media_plan(slug, selections, custom)


def save_upload(slug: str, scene_index: int, name: str, data: bytes) -> Path:
    upload_dir = OUTPUT_DIR / slug / "custom_uploads"
    upload_dir.mkdir(exist_ok=True)
    safe = re.sub(r"[^\w.\-]+", "_", name)
    path = upload_dir / f"scene_{scene_index:02d}_{safe}"
    path.write_bytes(data)
    selections, custom = load_media_plan(slug)
    custom[str(scene_index)] = str(path)
    selections.pop(str(scene_index), None)
    save_media_plan(slug, selections, custom)
    return path


def get_media_thumb(media_path: str | Path) -> Path | None:
    import hashlib
    THUMB_CACHE.mkdir(parents=True, exist_ok=True)
    key = hashlib.sha1(str(Path(media_path).resolve()).encode("utf-8")).hexdigest()[:20]
    thumb = THUMB_CACHE / f"{key}.jpg"
    if thumb.exists():
        return thumb
    if Path(media_path).exists() and extract_thumbnail(media_path, thumb, at_seconds=0.3):
        return thumb
    return None


def render_video(slug: str) -> Path:
    """Captions, clip plan, ffmpeg assembly and the thumbnail, from the chosen media."""
    from pipeline.assemble import assemble
    from pipeline.captions import build_ass
    from pipeline.thumbnail import make_thumbnail

    script = read_script(slug)
    if not script:
        raise ActionError("No script.json in this project")
    cfg_ = project_cfg(slug)
    out_dir = OUTPUT_DIR / slug
    audios = scene_audios(slug, len(script["scenes"]))
    if len(audios) != len(script["scenes"]):
        raise ActionError("Narration is missing for some scenes; go back to the Narrator step")
    selections, custom = load_media_plan(slug)
    durations = [a.duration for a in audios]
    build_ass(audios, out_dir / "captions.ass", cfg_)
    if doodle.is_doodle(script):   # a doodle video, or a Short cut from one: a drawing per sentence
        plan = doodle.clip_plan(script, audios, doodle.compose_frames(out_dir, script, cfg_), cfg_)
    else:
        plan = clips_for_scenes(script["scenes"], durations, out_dir, cfg_,
                                ui_state(slug).get("visuals") or cfg_["visuals"]["source"],
                                script_context=script, selections=selections, custom_media=custom)
    write_json(out_dir / "clip_plan.json", plan_to_json(plan))
    final = assemble(out_dir, plan, audios, cfg_)
    (out_dir / "final_1x.mp4").unlink(missing_ok=True)
    make_thumbnail(final, script.get("thumbnail_text", ""), out_dir / "thumbnail.jpg", cfg_)
    return final


def apply_speed(slug: str, speed: float) -> None:
    from pipeline.assemble import change_speed
    from pipeline.thumbnail import make_thumbnail
    out_dir = OUTPUT_DIR / slug
    final, base = out_dir / "final.mp4", out_dir / "final_1x.mp4"
    if not final.exists():
        raise ActionError("Render the video first")
    if not base.exists():
        shutil.copyfile(final, base)
    if abs(speed - 1.0) < 1e-6:
        shutil.copyfile(base, final)
    else:
        change_speed(final, speed, source=base)
    make_thumbnail(final, (read_script(slug) or {}).get("thumbnail_text", ""), out_dir / "thumbnail.jpg",
                   project_cfg(slug))


def retime_scene(slug: str, scene_index: int, speed: float) -> Path:
    apply_scene_speed(OUTPUT_DIR / slug, scene_index, speed)
    return render_video(slug)


def cut_shorts(slug: str, count: int) -> list[str]:
    out = run_cli_or_raise([PYTHON_EXE, str(RUN_PY), "cut", slug, "--count", str(count)], "Cutting Shorts")
    return re.findall(rf"^\s+({re.escape(slug)}-short-\d+)\s", out, flags=re.MULTILINE)


def publish(slug: str, privacy: str, when: str, at: str = "") -> str:
    cmd = [PYTHON_EXE, str(RUN_PY), "publish", slug, "--privacy", privacy]
    if when == "slot":
        cmd.append("--next-slot")
    elif when == "at":
        if not at.strip():
            raise ActionError("Enter the time to go public first")
        cmd += ["--at", at.strip()]
    out = run_cli_or_raise(cmd, "Publishing")
    m = re.search(r"https://youtu\.be/\S+", out)
    return m.group(0) if m else "uploaded"


def next_slot_label() -> str:
    from pipeline import scheduler
    try:
        return scheduler.describe(scheduler.next_slots(cfg(), 1)[0], cfg())
    except Exception:
        return "the next free slot"


def voice_sample(engine: str, name: str) -> Path:
    return preview_voice(cfg(), engine, name)


# ---------------------------------------------------------------------------
# Doodle explainers
# ---------------------------------------------------------------------------

def kaggle_ready() -> dict:
    """Whether Kaggle drawing can start, and what to do if not (no network call)."""
    return {"installed": kaggle_images.installed(), "configured": kaggle_images.configured(),
            "hint": kaggle_images.credentials_hint(),
            "ok": kaggle_images.installed() and kaggle_images.configured()}


def doodle_start_drawing(slug: str, backend: str) -> bool:
    """Draw every missing picture in the background (Kaggle runs take 20-60 minutes)."""
    if backend not in doodle.BACKENDS:
        raise ActionError(f"Unknown drawing backend '{backend}'")
    if backend == "kaggle" and not kaggle_ready()["ok"]:
        raise ActionError(kaggle_images.credentials_hint())
    if backend == "diffusion":
        release_gpu()
    if log_state(OUTPUT_DIR / slug / "drawing.log", stale_minutes=20) == "running":
        raise ActionError("A drawing run is already going for this video; wait for it to finish")
    set_ui_state(slug, visuals=backend)
    start_background([PYTHON_EXE, str(RUN_PY), "make", "--slug", slug, "--visuals", backend,
                      "--stop-after", "media_selection"], OUTPUT_DIR / slug / "drawing.log")
    return True


def doodle_redo(slug: str, ids: list[str]) -> int:
    """New takes for these beats: their drawings are deleted and their seeds change."""
    script = read_script(slug) or {}
    count = doodle.mark_for_redo(OUTPUT_DIR / slug, script, ids)
    if count:
        write_script(slug, script)
        for stale in ("final.mp4", "final_1x.mp4"):
            (OUTPUT_DIR / slug / stale).unlink(missing_ok=True)
    return count


def doodle_beats(slug: str) -> list[dict]:
    """Every beat with what the gallery shows: id, scene, chapter, text, picture, image (or None)."""
    script = read_script(slug) or {}
    out_dir = OUTPUT_DIR / slug
    chapters = sorted(((int(c.get("first_scene", 0)), c.get("title", "")) for c in script.get("chapters", []) or []))
    rows = []
    for i, scene in enumerate(script.get("scenes", [])):
        chapter = next((t for first, t in reversed(chapters) if first <= i), "")
        for k, beat in enumerate(scene.get("beats") or []):
            raw = doodle.raw_path(out_dir, i, k)
            rows.append({"id": doodle.beat_id(i, k), "scene": i, "chapter": chapter, **beat,
                         "image": str(raw) if raw.exists() else None})
    return rows


def doodle_thumb(path: str | Path, width: int = 360) -> Path | None:
    """A small JPEG of a drawing for the gallery, cached by file and modification time."""
    import hashlib
    from PIL import Image
    path = Path(path)
    if not path.exists():
        return None
    THUMB_CACHE.mkdir(parents=True, exist_ok=True)
    key = hashlib.sha1(f"{path.resolve()}|{path.stat().st_mtime_ns}|{width}".encode("utf-8")).hexdigest()[:20]
    thumb = THUMB_CACHE / f"d_{key}.jpg"
    if not thumb.exists():
        with Image.open(path) as im:
            im = im.convert("RGB")
            im.thumbnail((width, width))
            im.save(thumb, format="JPEG", quality=82)
    return thumb


def doodle_start_render(slug: str) -> bool:
    """Frames, timing, ffmpeg and thumbnail in the background (several minutes for ten minutes of video)."""
    for stale in ("final.mp4", "final_1x.mp4"):
        (OUTPUT_DIR / slug / stale).unlink(missing_ok=True)
    start_background([PYTHON_EXE, str(RUN_PY), "make", "--slug", slug], OUTPUT_DIR / slug / "render.log")
    return True


def doodle_fit_note(slug: str) -> str:
    """How the narration length compares with the doodle window (9-11 minutes by default)."""
    script = read_script(slug) or {}
    total = sum(a.duration for a in scene_audios(slug, len(script.get("scenes", []))))
    low, high = doodle.settings(project_cfg(slug))["fit_minutes"]
    minutes = total / 60
    if not total:
        return ""
    if low - 0.05 <= minutes <= high + 0.05:
        return f"✅ {minutes:.1f} minutes, inside the {low}-{high} minute window."
    return (f"⚠️ {minutes:.1f} minutes even after re-timing as far as allowed; the script is too "
            f"{'short' if minutes < low else 'long'} for {low}-{high} minutes. Rewrite or edit script.json.")


# ---------------------------------------------------------------------------
# Lofi
# ---------------------------------------------------------------------------

def lofi_plan(idea: str, minutes: int, visuals: str, music: str, rain: str) -> str:
    lofi_cfg = formats.apply(cfg(), "lofi")
    slug = unique_slug(f"{date.today().isoformat()}-lofi-{slugify(idea, 36)}")
    script = lofi.plan_project(idea, OUTPUT_DIR / slug, lofi_cfg, minutes=minutes, visuals=visuals,
                               music_engine=music, rain=rain)
    history.add(slug, script["title"], idea, kind="lofi", fmt="lofi")
    set_ui_state(slug, visuals=visuals, kind="lofi")
    return slug


def lofi_draw(slug: str, visuals: str, variant: int | None = None) -> Path:
    return lofi.draw_scene(OUTPUT_DIR / slug, project_cfg(slug), variant=variant, visuals=visuals)


def release_gpu() -> None:
    """The studio keeps Stable Diffusion loaded after drawing; ACE-Step needs the whole card."""
    diffusion.unload()


def lofi_preview_track(slug: str, engine: str) -> dict:
    if engine == "acestep":
        release_gpu()
    return lofi.preview_track(OUTPUT_DIR / slug, project_cfg(slug), music_engine=engine).to_dict()


def lofi_save_settings(slug: str, **values) -> None:
    script = read_script(slug) or {}
    script["settings"] = dict(script.get("settings") or {}, **{k: v for k, v in values.items() if v is not None})
    write_json(OUTPUT_DIR / slug / "script.json", script)


def lofi_save_details(slug: str, title: str, description: str, thumbnail_text: str, tags: list[str]) -> None:
    script = read_script(slug) or {}
    script.update({"title": title.strip()[:100], "description": description.strip(),
                   "thumbnail_text": thumbnail_text.strip(), "tags": [t for t in tags if t]})
    write_json(OUTPUT_DIR / slug / "script.json", script)


def lofi_start_render(slug: str) -> bool:
    release_gpu()
    (OUTPUT_DIR / slug / "final.mp4").unlink(missing_ok=True)
    start_background([PYTHON_EXE, str(RUN_PY), "lofi", "--slug", slug], OUTPUT_DIR / slug / "render.log")
    return True   # act() treats None as a failure, which kept the page from switching to the progress view


def acestep_running() -> bool:
    return lofi_music.acestep_available(cfg())


def kaggle_ready() -> bool:
    from pipeline import kaggle_images
    return kaggle_images.installed() and kaggle_images.configured()


def music_library() -> dict:
    """How much downloaded Pixabay music and rain is on hand, and where to put more."""
    return lofi_library.library_status(cfg())


def lofi_clip(slug: str) -> dict | None:
    """The chosen Pixabay loop for a lofi project (path and credit), or None."""
    script = read_script(slug) or {}
    clip = script.get("scene_clip")
    if clip and Path(clip).exists():
        return {"path": clip, **(script.get("clip_credit") or {})}
    return None


def tracks_needed(minutes: float) -> int:
    per = max(0.5, float(lofi_music.settings(cfg())["track_minutes"]))
    return max(1, int(-(-minutes // per)))


def video_duration(slug: str) -> float:
    try:
        return duration_of(OUTPUT_DIR / slug / "final.mp4")
    except Exception:
        return 0.0
