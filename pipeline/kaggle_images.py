"""Draw a project's images on Kaggle's free GPUs and bring them back.

Kaggle gives every phone-verified account about 30 GPU hours a week (two T4s per
session). One run here:
  1. writes a small Python script with the whole job embedded (prompts, seeds,
     settings; see kaggle_kernel/doodle_images.py) plus its kernel-metadata.json,
  2. pushes it as a private script kernel with a GPU and internet (for the model
     download) and lets it run,
  3. polls until it finishes, downloads doodle_images.zip and result.json, checks
     the job id, and unpacks the images into the project.

The same kernel (kaggle.kernel_slug) is reused, each run a new version, so the
Kaggle account does not fill up with notebooks. That also means one job at a
time per account: a job id written into the output tells a stale or foreign
result apart from ours.

Credentials are the Kaggle client's own: KAGGLE_API_TOKEN (or ~/.kaggle/access_token)
from kaggle.com/settings, or the legacy KAGGLE_USERNAME + KAGGLE_KEY / kaggle.json.
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

from .util import log, read_json, write_json

KERNEL_TEMPLATE = Path(__file__).resolve().parent / "kaggle_kernel" / "doodle_images.py"

DEFAULTS = {
    "kernel_slug": "yt-autopilot-doodle-images",
    "machine_shape": "NvidiaTeslaT4",     # P100 is retired on Kaggle; T4 (x2) is the free GPU
    "model": "stabilityai/stable-diffusion-xl-base-1.0",
    "vae": "madebyollin/sdxl-vae-fp16-fix",
    "lightning": True,                    # SDXL-Lightning: 8 steps instead of ~28
    "lightning_repo": "ByteDance/SDXL-Lightning",
    "lightning_file": "sdxl_lightning_8step_lora.safetensors",
    "steps": 8,
    "guidance": 0.0,
    "slow_steps": 28,
    "slow_guidance": 6.0,
    "width": 1344,                        # SDXL's native 16:9 size; upscaled to 1920x1080 here
    "height": 768,
    "ip_adapter": True,                   # keep the recurring character recognisable
    "ip_adapter_scale": 0.45,
    "character_seed": 7,
    "max_gpus": 2,
    "max_minutes": 660,                   # stop drawing before Kaggle's 12-hour session limit
    "poll_seconds": 30,
    "timeout_minutes": 300,               # give up waiting locally after this long
}
ACTIVE = {"QUEUED", "RUNNING", "NEW_SCRIPT", "UNKNOWN"}
STALE_GRACE_SECONDS = 15 * 60


def settings(cfg: dict) -> dict:
    merged = dict(DEFAULTS)
    merged.update(cfg.get("kaggle", {}) or {})
    return merged


def credentials_hint() -> str:
    return ("Kaggle is not set up. Create an API token at https://www.kaggle.com/settings (API, "
            "'Generate New Token') and put it in .env as KAGGLE_API_TOKEN=..., then verify your phone "
            "number on Kaggle so notebooks may use a GPU and the internet.")


def configured() -> bool:
    """Whether credentials are present (does not contact Kaggle)."""
    home = Path(os.environ.get("KAGGLE_CONFIG_DIR") or Path.home() / ".kaggle")
    return bool(os.environ.get("KAGGLE_API_TOKEN") or (home / "access_token").exists()
                or (os.environ.get("KAGGLE_USERNAME") and os.environ.get("KAGGLE_KEY"))
                or (home / "kaggle.json").exists())


def installed() -> bool:
    import importlib.util
    return importlib.util.find_spec("kaggle") is not None


def _api():
    if not installed():
        raise RuntimeError("The Kaggle client is not installed: .venv\\Scripts\\pip install -r requirements.txt")
    if not configured():
        raise RuntimeError(credentials_hint())
    from kaggle.api.kaggle_api_extended import KaggleApi
    api = KaggleApi()
    try:
        api.authenticate()
    except (Exception, SystemExit) as exc:
        raise RuntimeError(f"Kaggle rejected the credentials ({type(exc).__name__}). " + credentials_hint()) from exc
    return api


def _username(api) -> str:
    name = (api.config_values or {}).get("username") or os.environ.get("KAGGLE_USERNAME")
    if not name:
        raise RuntimeError("Could not tell the Kaggle username from the credentials; add KAGGLE_USERNAME to .env")
    return str(name)


def kernel_id(api, cfg: dict) -> str:
    return f"{_username(api)}/{settings(cfg)['kernel_slug']}"


def _title(slug: str) -> str:
    # Kaggle derives the slug from the title; keep them in step.
    return slug.replace("-", " ")


def write_kernel(folder: Path, kid: str, job: dict, cfg: dict) -> Path:
    """The pushable folder: the script with the job embedded, and its metadata."""
    s = settings(cfg)
    folder.mkdir(parents=True, exist_ok=True)
    payload = base64.b64encode(json.dumps(job, ensure_ascii=False).encode("utf-8")).decode("ascii")
    code = KERNEL_TEMPLATE.read_text(encoding="utf-8").replace("__JOB_B64__", payload, 1)
    (folder / "doodle_images.py").write_text(code, encoding="utf-8")
    write_json(folder / "kernel-metadata.json", {
        "id": kid,
        "title": _title(kid.split("/", 1)[1]),
        "code_file": "doodle_images.py",
        "language": "python",
        "kernel_type": "script",
        "is_private": True,
        "enable_gpu": True,
        "enable_internet": True,
        "machine_shape": s["machine_shape"],
        "dataset_sources": [],
        "competition_sources": [],
        "kernel_sources": [],
        "model_sources": [],
    })
    return folder


def build_job(jobs: list[dict], character_prompt: str, cfg: dict) -> dict:
    from .doodle import settings as doodle_settings
    s = settings(cfg)
    keys = ("model", "vae", "lightning", "lightning_repo", "lightning_file", "steps", "guidance", "slow_steps",
            "slow_guidance", "width", "height", "ip_adapter", "ip_adapter_scale", "character_seed", "max_gpus",
            "max_minutes")
    return {
        "job_id": uuid.uuid4().hex[:12],
        "jobs": [{k: j[k] for k in ("id", "prompt", "seed", "character")} for j in jobs],
        "character_prompt": character_prompt,
        "negative": doodle_settings(cfg)["negative"],
        "settings": {k: s[k] for k in keys},
    }


def _digest(jobs: list[dict]) -> str:
    return hashlib.sha256(json.dumps([[j["id"], j["prompt"], j["seed"]] for j in jobs]).encode()).hexdigest()[:16]


def _status(api, kid: str) -> tuple[str, str]:
    resp = api.kernels_status(kid)
    status = getattr(resp, "status", "")
    name = getattr(status, "name", str(status)).split(".")[-1].upper()
    return name or "UNKNOWN", str(getattr(resp, "failure_message", "") or "")


def _logs_tail(api, kid: str, chars: int = 3000) -> str:
    try:
        text = api.kernels_logs(kid) or ""
    except Exception as exc:
        return f"(could not read the Kaggle log: {type(exc).__name__})"
    # The log is JSON lines of {"data": "..."} on Kaggle; keep it readable either way.
    try:
        lines = [e.get("data", "") for e in json.loads(text)]
        text = "".join(lines)
    except (ValueError, TypeError, AttributeError):
        pass
    return text[-chars:]


def run_job(out_dir: Path, jobs: list[dict], character_prompt: str, cfg: dict, api=None) -> dict:
    """Push, wait, download, unpack. Resumes waiting on a job this project already pushed."""
    from .doodle import raw_dir
    s = settings(cfg)
    api = api or _api()
    kid = kernel_id(api, cfg)
    work = Path(out_dir) / "doodle" / "kaggle"
    state_path = work / "job.json"
    digest = _digest(jobs)

    state = read_json(state_path) if state_path.exists() else {}
    resumable = (state.get("digest") == digest and state.get("kernel") == kid and state.get("status") == "pushed"
                 and time.time() - float(state.get("pushed_at", 0)) < float(s["timeout_minutes"]) * 60)
    if resumable:
        log(f"kaggle: resuming job {state['job_id']} on {kid}")
    else:
        job = build_job(jobs, character_prompt, cfg)
        folder = write_kernel(work / "kernel", kid, job, cfg)
        log(f"kaggle: pushing {len(jobs)} prompts to {kid} (GPU {s['machine_shape']})")
        result = api.kernels_push(str(folder))
        error = getattr(result, "error", None) if result is not None else "no response"
        if error:
            raise RuntimeError(f"Kaggle refused the notebook: {error}")
        version = getattr(result, "version_number", None) or getattr(result, "versionNumber", None)
        url = getattr(result, "url", "") or f"https://www.kaggle.com/code/{kid}"
        state = {"job_id": job["job_id"], "kernel": kid, "digest": digest, "status": "pushed",
                 "pushed_at": time.time(), "version": version, "url": url, "count": len(jobs)}
        write_json(state_path, state)
        log(f"kaggle: running as version {version}; watch it at {url}")

    deadline = float(state["pushed_at"]) + float(s["timeout_minutes"]) * 60
    poll = max(10.0, float(s["poll_seconds"]))
    last, last_note = "", 0.0
    time.sleep(min(poll, 20.0))
    while True:
        try:
            status, message = _status(api, kid)
        except Exception as exc:  # a network blip must not lose a long job
            status, message = "UNKNOWN", f"{type(exc).__name__}"
        if status != last or time.time() - last_note > 300:
            # A line every few minutes also tells the UI the job is alive.
            minutes = (time.time() - float(state["pushed_at"])) / 60
            log(f"kaggle: {status.lower()} ({minutes:.0f} min since the push)" + (f": {message}" if message else ""))
            last, last_note = status, time.time()
        if status == "COMPLETE":
            got = _download(api, kid, work / "output", state)
            if got is None:
                if time.time() - float(state["pushed_at"]) < STALE_GRACE_SECONDS:
                    time.sleep(poll)   # Kaggle still reports the previous version; ours has not started
                    continue
                raise RuntimeError("Kaggle finished a different job (another project used the notebook "
                                   "meanwhile). Draw again.")
            summary = _unpack(work / "output", raw_dir(out_dir), jobs, state["job_id"])
            state.update(status="done", finished_at=time.time(), **{k: summary[k] for k in ("done", "failed")})
            write_json(state_path, state)
            return summary
        if status in ("ERROR", "CANCEL_REQUESTED", "CANCEL_ACKNOWLEDGED"):
            if time.time() - float(state["pushed_at"]) < STALE_GRACE_SECONDS and status == "ERROR" and not message:
                # Right after a push the status can still be the previous run's.
                time.sleep(poll)
                if _status(api, kid)[0] in ACTIVE:
                    continue
            state.update(status="failed")
            write_json(state_path, state)
            raise RuntimeError(f"The Kaggle run ended with {status.lower()}"
                               + (f": {message}" if message else "") + "\n" + _logs_tail(api, kid))
        if time.time() > deadline:
            raise RuntimeError(f"Stopped waiting after {s['timeout_minutes']} minutes; the run may still finish. "
                               f"Draw again later to pick up its result: {state.get('url', kid)}")
        time.sleep(poll)


def _download(api, kid: str, dest: Path, state: dict) -> dict | None:
    """Fetch the output; None when it belongs to another job."""
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


def _unpack(dest: Path, raw: Path, jobs: list[dict], job_id: str) -> dict:
    zip_path = next(iter(dest.rglob("doodle_images.zip")), None)
    result = read_json(next(iter(dest.rglob("result.json"))))
    if not zip_path:
        raise RuntimeError("The Kaggle run produced no doodle_images.zip")
    raw.mkdir(parents=True, exist_ok=True)
    wanted = {j["id"] for j in jobs} | {"character"}
    done = []
    with zipfile.ZipFile(zip_path) as zf:
        for name in zf.namelist():
            stem, ext = os.path.splitext(os.path.basename(name))
            if ext.lower() != ".png" or stem not in wanted:
                continue   # never write a path the job did not ask for
            target = raw / f"{stem}.png"
            with zf.open(name) as src, open(target, "wb") as dst:
                shutil.copyfileobj(src, dst)
            if stem != "character":
                done.append(stem)
    failed = result.get("failed") or {}
    log(f"kaggle: {len(done)} drawings unpacked, {len(failed)} failed"
        + (f" in {result.get('minutes')} min on {', '.join(result.get('gpus', []))}" if result.get("gpus") else ""))
    for bid, why in list(failed.items())[:5]:
        log(f"kaggle:   {bid}: {why}")
    return {"done": len(done), "failed": len(failed), "job_id": job_id}


def job_state(out_dir: Path) -> dict:
    path = Path(out_dir) / "doodle" / "kaggle" / "job.json"
    try:
        return read_json(path) if path.exists() else {}
    except (OSError, ValueError):
        return {}
