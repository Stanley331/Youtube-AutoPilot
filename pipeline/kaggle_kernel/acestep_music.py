"""Runs on Kaggle, pushed there by yt-autopilot: generates lofi tracks with ACE-Step 1.5 on a free GPU.

Everything the job needs is embedded below (track titles, seeds, prompt, length),
so the notebook needs no dataset. ACE-Step 1.5 (MIT; its generated audio may be
used commercially) is installed from GitHub, its weights download on first use,
and every track is written to /kaggle/working as mp3, zipped with result.json,
which yt-autopilot downloads and unpacks into the project's music folder.

Local test without a GPU: MUSIC_FAKE=1 MUSIC_WORK_DIR=some/dir python acestep_music.py
(writes short synthesized tones with ffmpeg instead of running the model).
"""
import base64
import json
import os
import subprocess
import sys
import time
import traceback
import zipfile

JOB = json.loads(base64.b64decode("__JOB_B64__").decode("utf-8"))
WORK = os.environ.get("MUSIC_WORK_DIR", "/kaggle/working")
FAKE = os.environ.get("MUSIC_FAKE") == "1"
S = JOB["settings"]
STARTED = time.time()


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# Only the zip and result.json may live in /kaggle/working: everything there becomes notebook
# output that is downloaded afterwards. The environment and the 10 GB of weights go to /tmp.
TMP = "/tmp/yt-autopilot" if os.path.isdir("/tmp") else os.path.join(WORK, "tmp")
ENV = os.path.join(TMP, "ace_env")

WORKER = r"""
import json, os, shutil, sys, time
job_path, music, scratch = sys.argv[1], sys.argv[2], sys.argv[3]
JOB = json.load(open(job_path, encoding="utf-8"))
S = JOB["settings"]
def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)
import torch
log(f"worker: python {sys.version.split()[0]}, torch {torch.__version__}, GPU {torch.cuda.get_device_name(0)}")
from acestep.handler import AceStepHandler
from acestep.llm_inference import LLMHandler
from acestep.inference import GenerationConfig, GenerationParams, generate_music
dit = AceStepHandler()
dit.initialize_service(project_root=os.path.join(os.path.dirname(scratch), "acestep"), config_path=S["config_path"], device="cuda")
llm = LLMHandler()
log("ACE-Step ready")
deadline = float(JOB["started"]) + float(S.get("max_minutes", 600)) * 60
done, failed = [], {}
for track in JOB["tracks"]:
    if time.time() > deadline:
        failed[track["id"]] = "out of session time"
        continue
    try:
        t0 = time.time()
        params = GenerationParams(task_type="text2music", caption=S["prompt"][:512], lyrics="[Instrumental]",
                                  instrumental=True, duration=int(S["seconds"]), inference_steps=int(S["steps"]),
                                  seed=int(track["seed"]), shift=3.0, thinking=False)
        config = GenerationConfig(batch_size=1, use_random_seed=False, audio_format="mp3")
        result = generate_music(dit, llm, params, config, save_dir=scratch)
        if not getattr(result, "success", False):
            raise RuntimeError(str(getattr(result, "error", "generation failed"))[:400])
        audios = getattr(result, "audios", None) or []
        first = audios[0] if audios else None
        path = first.get("path") if isinstance(first, dict) else getattr(first, "path", None)
        if not path or not os.path.exists(path):
            raise RuntimeError("ACE-Step reported success but wrote no audio file")
        shutil.move(path, os.path.join(music, track["id"] + ".mp3"))   # /tmp and /kaggle/working differ
        done.append(track["id"])
        log(f"{track['id']} '{track['title']}' in {time.time() - t0:.0f} s")
    except Exception as exc:
        failed[track["id"]] = f"{type(exc).__name__}: {exc}"[:300]
        log(f"{track['id']} failed: {failed[track['id']]}")
json.dump({"done": done, "failed": failed}, open(os.path.join(scratch, "worker.json"), "w"))
"""


def sh(cmd, what):
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"{what} failed (is internet enabled for this notebook?):\n" + result.stderr[-2000:])
    return result.stdout


def install():
    """Kaggle's image runs Python 3.13; ACE-Step wants 3.11 or 3.12, so it gets its own uv environment."""
    sh([sys.executable, "-m", "pip", "install", "-q", "uv"], "pip install uv")
    uv = [sys.executable, "-m", "uv"]
    sh(uv + ["venv", ENV, "--python", "3.12", "--seed"], "creating the Python 3.12 environment")
    python = os.path.join(ENV, "bin", "python")
    log("installing ACE-Step and its dependencies into the 3.12 environment (a few minutes)")
    sh(uv + ["pip", "install", "--python", python, S["repo"]], "pip install of ACE-Step")
    return python


def generate_all(python, music, scratch):
    job_path = os.path.join(scratch, "job.json")
    with open(job_path, "w", encoding="utf-8") as f:
        json.dump(dict(JOB, started=STARTED), f)
    worker = os.path.join(scratch, "worker.py")
    with open(worker, "w", encoding="utf-8") as f:
        f.write(WORKER)
    proc = subprocess.run([python, worker, job_path, music, scratch], text=True)
    out = os.path.join(scratch, "worker.json")
    if not os.path.exists(out):
        raise RuntimeError(f"the generation worker exited with code {proc.returncode} before writing results; "
                           "see the lines above")
    with open(out, encoding="utf-8") as f:
        return json.load(f)


def fake_track(track, path):
    """A quiet tone whose pitch follows the seed, so the pipeline has something to cut."""
    freq = 110 + (int(track["seed"]) % 12) * 20
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-f", "lavfi",
                    "-i", f"sine=frequency={freq}:duration={min(int(S['seconds']), 20)}",
                    "-af", "volume=0.2", "-c:a", "libmp3lame", "-b:a", "128k", path], check=True)


def main():
    music = os.path.join(WORK, "music")
    os.makedirs(music, exist_ok=True)
    tracks = JOB["tracks"]
    done, failed = [], {}
    info = {"job_id": JOB["job_id"], "fake": FAKE}
    log(f"job {JOB['job_id']}: {len(tracks)} tracks of {S['seconds']} s")

    if FAKE:
        for track in tracks:
            fake_track(track, os.path.join(music, track["id"] + ".mp3"))
            done.append(track["id"])
    else:
        import torch
        if not torch.cuda.is_available():
            raise SystemExit("No GPU in this session. Turn on a GPU accelerator (phone-verified account).")
        info.update(gpu=torch.cuda.get_device_name(0))
        log(f"GPU: {info['gpu']}; notebook python {sys.version.split()[0]}")
        python = install()
        scratch = os.path.join(TMP, "scratch")
        os.makedirs(scratch, exist_ok=True)
        outcome = generate_all(python, music, scratch)
        done, failed = outcome["done"], outcome["failed"]

    with zipfile.ZipFile(os.path.join(WORK, "lofi_music.zip"), "w", zipfile.ZIP_STORED) as zf:
        for name in sorted(os.listdir(music)):
            zf.write(os.path.join(music, name), name)
    for name in os.listdir(music):
        os.remove(os.path.join(music, name))
    os.rmdir(music)
    info.update(done=sorted(done), failed=failed, minutes=round((time.time() - STARTED) / 60, 1))
    with open(os.path.join(WORK, "result.json"), "w", encoding="utf-8") as f:
        json.dump(info, f, indent=1)
    log(f"finished: {len(done)} tracks, {len(failed)} failed, {info['minutes']} min")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        with open(os.path.join(WORK, "result.json"), "w", encoding="utf-8") as f:
            json.dump({"job_id": JOB["job_id"], "error": traceback.format_exc()[-3000:]}, f)
        raise
