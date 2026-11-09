"""Generate a lofi tracklist with ACE-Step on Kaggle's free GPU and bring it back.

Same mechanics as the doodle images (``kaggle_images``): one private script kernel
per purpose, reused as a new version each run, a job id in the output to tell a
stale result apart, and ``music/kaggle/job.json`` in the project so a closed
studio can pick the run up again. One job generates the whole tracklist, so a
60-minute video is a single push, roughly a minute of generation per track on a
T4 plus the one-time install and weight download.
"""
from __future__ import annotations

import base64
import hashlib
import json
import os
import shutil
import time
import uuid
import zipfile
from pathlib import Path

from .kaggle_images import ACTIVE, STALE_GRACE_SECONDS, _api, _logs_tail, _status, _username, credentials_hint
from .kaggle_images import settings as kaggle_settings
from .util import log, read_json, write_json

KERNEL_TEMPLATE = Path(__file__).resolve().parent / "kaggle_kernel" / "acestep_music.py"

DEFAULTS = {
    "kaggle_kernel_slug": "yt-autopilot-lofi-music",
    "kaggle_repo": "git+https://github.com/ACE-Step/ACE-Step-1.5.git",
    "kaggle_config_path": "acestep-v15-turbo",
    "kaggle_max_minutes": 600,
}


def settings(cfg: dict) -> dict:
    merged = dict(DEFAULTS)
    merged.update({k: v for k, v in (cfg.get("music", {}) or {}).items() if k in DEFAULTS})
    return merged


def write_kernel(folder: Path, kid: str, job: dict, cfg: dict) -> Path:
    folder.mkdir(parents=True, exist_ok=True)
    payload = base64.b64encode(json.dumps(job, ensure_ascii=False).encode("utf-8")).decode("ascii")
    code = KERNEL_TEMPLATE.read_text(encoding="utf-8").replace("__JOB_B64__", payload, 1)
    (folder / "acestep_music.py").write_text(code, encoding="utf-8")
    write_json(folder / "kernel-metadata.json", {
        "id": kid, "title": kid.split("/", 1)[1].replace("-", " "),
        "code_file": "acestep_music.py", "language": "python", "kernel_type": "script",
        "is_private": True, "enable_gpu": True, "enable_internet": True,
        "machine_shape": kaggle_settings(cfg)["machine_shape"],
        "dataset_sources": [], "competition_sources": [], "kernel_sources": [], "model_sources": [],
    })
    return folder


def build_job(tracks: list[dict], seconds: int, prompt: str, steps: int, cfg: dict) -> dict:
    s = settings(cfg)
    return {
        "job_id": uuid.uuid4().hex[:12],
        "tracks": [{k: t[k] for k in ("id", "title", "seed")} for t in tracks],
        "settings": {"seconds": int(seconds), "prompt": prompt, "steps": int(steps),
                     "repo": s["kaggle_repo"], "config_path": s["kaggle_config_path"],
                     "max_minutes": s["kaggle_max_minutes"]},
    }


def _digest(tracks: list[dict], seconds: int, prompt: str) -> str:
    key = json.dumps([[t["id"], t["seed"]] for t in tracks] + [seconds, prompt])
    return hashlib.sha256(key.encode()).hexdigest()[:16]


def run_job(music_dir: Path, tracks: list[dict], seconds: int, prompt: str, steps: int, cfg: dict) -> dict:
    """Push, wait, download, unpack mp3s into ``music_dir``. Resumes a run this project already pushed."""
    ks = kaggle_settings(cfg)
    api = _api()
    kid = f"{_username(api)}/{settings(cfg)['kaggle_kernel_slug']}"
    work = Path(music_dir) / "kaggle"
    state_path = work / "job.json"
    digest = _digest(tracks, seconds, prompt)

    state = read_json(state_path) if state_path.exists() else {}
    resumable = (state.get("digest") == digest and state.get("kernel") == kid and state.get("status") == "pushed"
                 and time.time() - float(state.get("pushed_at", 0)) < float(ks["timeout_minutes"]) * 60)
    if resumable:
        log(f"kaggle: resuming music job {state['job_id']} on {kid}")
    else:
        job = build_job(tracks, seconds, prompt, steps, cfg)
        folder = write_kernel(work / "kernel", kid, job, cfg)
        log(f"kaggle: pushing {len(tracks)} tracks to {kid} (GPU {ks['machine_shape']})")
        result = api.kernels_push(str(folder))
        error = getattr(result, "error", None) if result is not None else "no response"
        if error:
            raise RuntimeError(f"Kaggle refused the notebook: {error}")
        version = getattr(result, "version_number", None) or getattr(result, "versionNumber", None)
        url = getattr(result, "url", "") or f"https://www.kaggle.com/code/{kid}"
        state = {"job_id": job["job_id"], "kernel": kid, "digest": digest, "status": "pushed",
                 "pushed_at": time.time(), "version": version, "url": url, "count": len(tracks)}
        write_json(state_path, state)
        log(f"kaggle: running as version {version}; watch it at {url}")

    deadline = float(state["pushed_at"]) + float(ks["timeout_minutes"]) * 60
    poll = max(10.0, float(ks["poll_seconds"]))
    last, last_note = "", 0.0
    time.sleep(min(poll, 20.0))
    while True:
        try:
            status, message = _status(api, kid)
        except Exception as exc:
            status, message = "UNKNOWN", f"{type(exc).__name__}"
        if status != last or time.time() - last_note > 300:
            minutes = (time.time() - float(state["pushed_at"])) / 60
            log(f"kaggle: {status.lower()} ({minutes:.0f} min since the push)" + (f": {message}" if message else ""))
            last, last_note = status, time.time()
        if status == "COMPLETE":
            result = _download(api, kid, work / "output", state)
            if result is None:
                if time.time() - float(state["pushed_at"]) < STALE_GRACE_SECONDS:
                    time.sleep(poll)
                    continue
                raise RuntimeError("Kaggle finished a different job (another project used the notebook "
                                   "meanwhile). Render again.")
            summary = _unpack(work / "output", Path(music_dir), tracks, result)
            state.update(status="done", finished_at=time.time(), done=summary["done"], failed=summary["failed"])
            write_json(state_path, state)
            return summary
        if status in ("ERROR", "CANCEL_REQUESTED", "CANCEL_ACKNOWLEDGED"):
            if time.time() - float(state["pushed_at"]) < STALE_GRACE_SECONDS and status == "ERROR" and not message:
                time.sleep(poll)
                if _status(api, kid)[0] in ACTIVE:
                    continue
            state.update(status="failed")
            write_json(state_path, state)
            raise RuntimeError(f"The Kaggle run ended with {status.lower()}"
                               + (f": {message}" if message else "") + "\n" + _logs_tail(api, kid))
        if time.time() > deadline:
            raise RuntimeError(f"Stopped waiting after {ks['timeout_minutes']} minutes; the run may still finish. "
                               f"Render again later to pick up its result: {state.get('url', kid)}")
        time.sleep(poll)


def _download(api, kid: str, dest: Path, state: dict) -> dict | None:
    if dest.exists():
        shutil.rmtree(dest)
    dest.mkdir(parents=True)
    api.kernels_output(kid, str(dest), force=True, quiet=True)
    result_path = next(iter(dest.rglob("result.json")), None)
    if not result_path:
        raise RuntimeError("The Kaggle run finished without result.json; see its log on kaggle.com")
    result = read_json(result_path)
    if result.get("job_id") != state["job_id"]:
        return None
    if result.get("error"):
        raise RuntimeError("The Kaggle script failed:\n" + str(result["error"])[-2500:])
    return result


def _unpack(dest: Path, music_dir: Path, tracks: list[dict], result: dict) -> dict:
    zip_path = next(iter(dest.rglob("lofi_music.zip")), None)
    if not zip_path:
        raise RuntimeError("The Kaggle run produced no lofi_music.zip")
    music_dir.mkdir(parents=True, exist_ok=True)
    wanted = {t["id"] for t in tracks}
    done = []
    with zipfile.ZipFile(zip_path) as zf:
        for name in zf.namelist():
            stem, ext = os.path.splitext(os.path.basename(name))
            if ext.lower() != ".mp3" or stem not in wanted:
                continue
            with zf.open(name) as src, open(music_dir / f"{stem}.mp3", "wb") as dst:
                shutil.copyfileobj(src, dst)
            done.append(stem)
    failed = result.get("failed") or {}
    log(f"kaggle: {len(done)} tracks unpacked, {len(failed)} failed"
        + (f" in {result.get('minutes')} min on {result.get('gpu')}" if result.get("gpu") else ""))
    for tid, why in list(failed.items())[:5]:
        log(f"kaggle:   {tid}: {why}")
    return {"done": done, "failed": failed}
