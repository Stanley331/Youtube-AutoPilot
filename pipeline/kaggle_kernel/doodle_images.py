"""Runs on Kaggle, pushed there by yt-autopilot: draws doodle images on a free GPU.

Everything it needs is embedded below (the job: prompts, seeds, model settings),
so the notebook needs no dataset. It draws one reference picture of the video's
recurring character, then every beat's picture with SDXL (SDXL-Lightning for
speed, IP-Adapter so the character stays recognisable), using every GPU the
session has. Results go to /kaggle/working as one zip plus result.json, which
yt-autopilot downloads and unpacks into the project.

Licences: SDXL base 1.0 and SDXL-Lightning (CreativeML OpenRAIL++-M), the
fp16-fix VAE (MIT), IP-Adapter (Apache 2.0): all allow commercial use of outputs.

Local test without a GPU: DOODLE_FAKE=1 DOODLE_WORK_DIR=some/dir python doodle_images.py
"""
import base64
import json
import os
import subprocess
import sys
import threading
import time
import traceback
import zipfile

JOB = json.loads(base64.b64decode("__JOB_B64__").decode("utf-8"))
WORK = os.environ.get("DOODLE_WORK_DIR", "/kaggle/working")
FAKE = os.environ.get("DOODLE_FAKE") == "1"
S = JOB["settings"]
STARTED = time.time()


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def install():
    """Kaggle's image ships torch; diffusers and peft may be missing or old."""
    cmd = [sys.executable, "-m", "pip", "install", "-q", "-U", "diffusers>=0.32", "peft>=0.14"]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        log("pip install failed (is internet enabled for this notebook?):\n" + result.stderr[-1500:])


def load_pipeline(device):
    import torch
    from diffusers import AutoencoderKL, EulerDiscreteScheduler, StableDiffusionXLPipeline

    vae = AutoencoderKL.from_pretrained(S["vae"], torch_dtype=torch.float16)
    pipe = StableDiffusionXLPipeline.from_pretrained(S["model"], vae=vae, torch_dtype=torch.float16,
                                                     variant="fp16", use_safetensors=True)
    fast = False
    if S.get("lightning"):
        try:
            from huggingface_hub import hf_hub_download
            pipe.load_lora_weights(hf_hub_download(S["lightning_repo"], S["lightning_file"]))
            pipe.fuse_lora()
            pipe.scheduler = EulerDiscreteScheduler.from_config(pipe.scheduler.config,
                                                                timestep_spacing="trailing")
            fast = True
        except Exception as exc:  # slower, but still draws
            log(f"SDXL-Lightning unavailable ({type(exc).__name__}: {exc}); using {S['slow_steps']} steps")
    pipe.to(device)
    pipe.set_progress_bar_config(disable=True)
    return pipe, fast


def add_ip_adapter(pipe):
    try:
        pipe.load_ip_adapter("h94/IP-Adapter", subfolder="sdxl_models",
                             weight_name="ip-adapter_sdxl_vit-h.safetensors",
                             image_encoder_folder="models/image_encoder")
        return True
    except Exception as exc:
        log(f"IP-Adapter unavailable ({type(exc).__name__}: {exc}); character kept by prompt only")
        return False


def draw(pipe, fast, prompt, seed, ip=False, ref=None, character=False):
    import torch
    kwargs = dict(prompt=prompt, width=int(S["width"]), height=int(S["height"]),
                  generator=torch.Generator(device="cpu").manual_seed(int(seed)))
    if fast:
        kwargs.update(num_inference_steps=int(S["steps"]), guidance_scale=float(S["guidance"]))
    else:
        kwargs.update(num_inference_steps=int(S["slow_steps"]), guidance_scale=float(S["slow_guidance"]),
                      negative_prompt=JOB["negative"])
    if ip:
        pipe.set_ip_adapter_scale(float(S["ip_adapter_scale"]) if character else 0.0)
        kwargs["ip_adapter_image"] = ref
    return pipe(**kwargs).images[0]


def fake_draw(prompt, path):
    from PIL import Image, ImageDraw
    img = Image.new("RGB", (int(S["width"]), int(S["height"])), "white")
    d = ImageDraw.Draw(img)
    d.rectangle((10, 10, img.width - 10, img.height - 10), outline="black", width=4)
    d.text((30, 30), prompt[:120], fill="black")
    img.save(path)


def main():
    images = os.path.join(WORK, "images")
    os.makedirs(images, exist_ok=True)
    jobs = JOB["jobs"]
    done, failed = [], {}
    info = {"job_id": JOB["job_id"], "fake": FAKE}
    log(f"job {JOB['job_id']}: {len(jobs)} images")

    if FAKE:
        for job in jobs:
            fake_draw(job["prompt"], os.path.join(images, job["id"] + ".png"))
            done.append(job["id"])
    else:
        install()
        import torch
        if not torch.cuda.is_available():
            raise SystemExit("No GPU in this session. Turn on a GPU accelerator (phone-verified account).")
        gpus = min(torch.cuda.device_count(), int(S.get("max_gpus", 2)))
        info.update(torch=torch.__version__, gpus=[torch.cuda.get_device_name(i) for i in range(gpus)])
        log(f"torch {torch.__version__}, GPUs: {', '.join(info['gpus'])}")

        # The first GPU downloads the weights; the others then load from the cache.
        pipes = [load_pipeline("cuda:0")]
        for g in range(1, gpus):
            try:
                pipes.append(load_pipeline(f"cuda:{g}"))
            except Exception as exc:
                log(f"GPU {g} unused ({type(exc).__name__}: {exc})")
        fast = pipes[0][1]
        log(f"pipelines ready on {len(pipes)} GPU(s), {'Lightning' if fast else 'standard'} sampling")

        ref, ip = None, [False] * len(pipes)
        if JOB.get("character_prompt") and S.get("ip_adapter") and any(j["character"] for j in jobs):
            ref = draw(pipes[0][0], fast, JOB["character_prompt"], int(S.get("character_seed", 7)))
            ref.save(os.path.join(images, "character.png"))
            ip = [add_ip_adapter(p) for p, _ in pipes]
            log("character reference drawn")

        deadline = STARTED + float(S.get("max_minutes", 660)) * 60
        lock = threading.Lock()
        queue = list(jobs)

        def worker(n):
            pipe, worker_fast = pipes[n]
            while True:
                with lock:
                    if not queue:
                        return
                    job = queue.pop(0)
                if time.time() > deadline:
                    with lock:
                        failed[job["id"]] = "out of session time"
                    continue
                try:
                    img = draw(pipe, worker_fast, job["prompt"], job["seed"], ip[n], ref, job["character"])
                    img.save(os.path.join(images, job["id"] + ".png"))
                    with lock:
                        done.append(job["id"])
                        if len(done) % 10 == 0:
                            rate = (time.time() - STARTED) / len(done)
                            log(f"{len(done)}/{len(jobs)} drawn, {rate:.1f} s each overall")
                except Exception as exc:
                    with lock:
                        failed[job["id"]] = f"{type(exc).__name__}: {exc}"[:300]
                    log(f"{job['id']} failed: {failed[job['id']]}")

        threads = [threading.Thread(target=worker, args=(n,)) for n in range(len(pipes))]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

    with zipfile.ZipFile(os.path.join(WORK, "doodle_images.zip"), "w", zipfile.ZIP_STORED) as zf:
        for name in sorted(os.listdir(images)):
            zf.write(os.path.join(images, name), name)
    for name in os.listdir(images):
        os.remove(os.path.join(images, name))
    os.rmdir(images)
    info.update(done=sorted(done), failed=failed, minutes=round((time.time() - STARTED) / 60, 1))
    with open(os.path.join(WORK, "result.json"), "w", encoding="utf-8") as f:
        json.dump(info, f, indent=1)
    log(f"finished: {len(done)} drawn, {len(failed)} failed, {info['minutes']} min")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        with open(os.path.join(WORK, "result.json"), "w", encoding="utf-8") as f:
            json.dump({"job_id": JOB["job_id"], "error": traceback.format_exc()[-3000:]}, f)
        raise
