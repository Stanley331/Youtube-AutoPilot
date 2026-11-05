"""Local AI illustrations: one Stable Diffusion 1.5 panel per scene, on your own GPU.

Nothing leaves the machine and nothing is billed. The default checkpoint is
DreamShaper 8 (CreativeML OpenRAIL-M), a general-purpose SD 1.5 fine-tune that
handles both painterly story panels and photographic fact visuals. With
``diffusion.fast`` on, the LCM LoRA brings generation down to a handful of
steps, which matters on small cards.

Install the optional dependencies first (see requirements-diffusion.txt).
Everything here is lazy: the pipeline module imports without torch installed,
and the model is loaded once per process on first use.
"""
from __future__ import annotations

import hashlib
import importlib.util
import re
import threading
from pathlib import Path

from PIL import Image

from .config import ROOT
from .util import log

DEFAULTS = {
    "model": "Lykon/dreamshaper-8",
    "fast": True,                         # LCM LoRA: ~8 steps instead of ~25
    "lcm_lora": "latent-consistency/lcm-lora-sdv1-5",
    "steps": 8,                           # used when fast; slow mode uses slow_steps
    "slow_steps": 25,
    "guidance": 1.5,                      # LCM wants 1-2; slow mode uses slow_guidance
    "slow_guidance": 7.0,
    "width": 512,                         # generation size; upscaled to the video resolution
    "height": 896,
    "style": "storybook illustration, flat vector art, bold clean shapes, soft cinematic lighting, "
             "rich colors, highly detailed, vertical composition",
    "negative": "text, letters, watermark, logo, signature, blurry, deformed, disfigured, extra limbs, "
                "extra fingers, low quality, cropped, frame, border, nsfw",
    "candidates": 1,                      # panels per scene per fetch; Regenerate adds more
    "device": "auto",                     # auto | cuda | cpu
    "cache_dir": "cache/diffusion",       # model weights (a few GB, downloaded once)
}

_PIPE = None
_PIPE_KEY = None
_DEVICE = None
# One load and one render at a time. diffusers, transformers and peft patch torch's
# register_parameter process-wide while they load, so two threads loading at once
# (easy in the Streamlit UI: click, wait, click again) corrupt each other.
_LOCK = threading.RLock()


def settings(cfg: dict) -> dict:
    merged = dict(DEFAULTS)
    merged.update(cfg.get("diffusion", {}) or {})
    return merged


def available() -> bool:
    """True when torch and diffusers are importable (model may still need downloading)."""
    return all(importlib.util.find_spec(m) is not None for m in ("torch", "diffusers", "transformers"))


def install_hint() -> str:
    return ("Local AI illustrations need the optional packages. Run:\n"
            "  .venv\\Scripts\\pip install -r requirements-diffusion.txt "
            "--extra-index-url https://download.pytorch.org/whl/cu126")


def _pick_device(requested: str) -> str:
    import torch
    if requested in ("cuda", "cpu"):
        if requested == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("diffusion.device is 'cuda' but torch cannot see a CUDA GPU")
        return requested
    return "cuda" if torch.cuda.is_available() else "cpu"


def _load(cfg: dict):
    """Load (once) the SD 1.5 pipeline sized for a 4 GB card."""
    with _LOCK:
        return _load_unlocked(cfg)


def _load_unlocked(cfg: dict):
    global _PIPE, _PIPE_KEY, _DEVICE
    if not available():
        raise RuntimeError(install_hint())
    s = settings(cfg)
    key = (s["model"], bool(s["fast"]), s["lcm_lora"], s["device"])
    if _PIPE is not None and _PIPE_KEY == key:
        return _PIPE

    import torch
    from diffusers import DPMSolverMultistepScheduler, LCMScheduler, StableDiffusionPipeline

    device = _pick_device(str(s["device"]))
    dtype = torch.float16 if device == "cuda" else torch.float32
    cache_dir = ROOT / s["cache_dir"]
    cache_dir.mkdir(parents=True, exist_ok=True)
    log(f"diffusion: loading {s['model']} on {device} (first run downloads a few GB)")
    load_kwargs = dict(torch_dtype=dtype, cache_dir=str(cache_dir),
                       safety_checker=None, requires_safety_checker=False)
    try:
        pipe = StableDiffusionPipeline.from_pretrained(s["model"], **load_kwargs)
    except TypeError as exc:
        if "_is_hf_initialized" not in str(exc):
            raise
        # Older accelerate releases trip over transformers 5 while loading on the meta
        # device (accelerate issue 3943). Loading straight into RAM sidesteps it.
        log("diffusion: low-memory load failed; loading straight into RAM instead "
            "(upgrading accelerate fixes this for good)")
        pipe = StableDiffusionPipeline.from_pretrained(s["model"], low_cpu_mem_usage=False, **load_kwargs)
    if s["fast"]:
        pipe.load_lora_weights(s["lcm_lora"], cache_dir=str(cache_dir))
        pipe.fuse_lora()
        pipe.scheduler = LCMScheduler.from_config(pipe.scheduler.config)
    else:
        pipe.scheduler = DPMSolverMultistepScheduler.from_config(
            pipe.scheduler.config, use_karras_sigmas=True)
    pipe.set_progress_bar_config(disable=True)
    if device == "cuda":
        pipe.to(device)
        # Memory savers; names moved between diffusers releases, so probe for them.
        if hasattr(pipe, "enable_attention_slicing"):
            pipe.enable_attention_slicing()
        if hasattr(pipe.vae, "enable_slicing"):
            pipe.vae.enable_slicing()
        elif hasattr(pipe, "enable_vae_slicing"):
            pipe.enable_vae_slicing()
        # Pascal-era cards decode fp16 latents to black frames now and then; keep the VAE in fp32.
        pipe.vae.to(dtype=torch.float32)
    else:
        log("diffusion: no CUDA GPU found; generating on the CPU (minutes per panel)")
    _PIPE, _PIPE_KEY, _DEVICE = pipe, key, device
    return pipe


def unload() -> None:
    """Drop the loaded pipeline and give the VRAM back (ACE-Step needs the whole card)."""
    global _PIPE, _PIPE_KEY, _DEVICE
    with _LOCK:
        if _PIPE is None:
            return
        _PIPE, _PIPE_KEY, _DEVICE = None, None, None
        import gc
        gc.collect()
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass
        log("diffusion: model unloaded, GPU memory released")


def _clip_words(text: str, limit: int) -> str:
    words = re.sub(r"\s+", " ", str(text or "")).strip().split()
    return " ".join(words[:limit])


PROMPT_WORD_BUDGET = 55  # CLIP reads about 77 tokens; anything past that is silently dropped


def build_prompt(scene: dict, script_context: dict | None, cfg: dict) -> tuple[str, str]:
    """Positive and negative prompts for a scene; the storyboard fields do the work.

    Parts are added in priority order (style, subject, character, setting, props)
    and the prompt stops before CLIP's token limit, so the style is never cut.
    """
    s = settings(cfg)
    context = script_context or {}
    parts = [_clip_words(context.get("art_style") or s["style"], 20),
             _clip_words(scene.get("visual_description") or scene.get("visual_query", ""), 28)]
    character = _clip_words(context.get("main_character", ""), 14)
    subject = str(scene.get("visual_subject", "")).lower()
    if character and context.get("content_type", "story") == "story" and (
            not subject or any(w in subject for w in character.lower().split()[:2]) or "character" in subject):
        parts.append(character)
    setting = _clip_words(scene.get("setting", ""), 8)
    if setting and setting.lower() not in parts[1].lower():
        parts.append(setting)
    props = [p for p in scene.get("visual_objects", []) if p][:3]
    if props:
        parts.append(", ".join(props))

    chosen, used = [], 0
    for part in parts:
        n = len(part.split())
        if chosen and used + n > PROMPT_WORD_BUDGET:
            break
        chosen.append(part)
        used += n
    return ", ".join(chosen), s["negative"]


def pick_style(cfg: dict, script: dict) -> str:
    """One of ``diffusion.styles`` for this video (by title), so a batch does not all look alike."""
    styles = [str(x) for x in (settings(cfg).get("styles") or []) if str(x).strip()]
    if not styles:
        return ""
    title = str(script.get("title", ""))
    return styles[int(hashlib.sha256(title.encode("utf-8")).hexdigest()[:8], 16) % len(styles)]


def _seed(script_context: dict | None, index: int, variant: int) -> int:
    title = (script_context or {}).get("title", "")
    base = int(hashlib.sha256(title.encode("utf-8")).hexdigest()[:8], 16)
    return (base + index * 101 + variant * 7919) % (2 ** 31)


def generate_panel(scene: dict, out_dir: Path, index: int, cfg: dict,
                   script_context: dict | None = None, variant: int = 0) -> Path:
    """Render (or reuse) the AI panel for a scene and return its PNG path.

    ``variant`` changes the seed so Regenerate produces a different take.
    """
    s = settings(cfg)
    prompt, negative = build_prompt(scene, script_context, cfg)
    width, height = cfg["video"]["resolution"]
    gen_w, gen_h = int(s["width"]), int(s["height"])
    seed = _seed(script_context, index, variant)
    steps = int(s["steps"] if s["fast"] else s["slow_steps"])
    guidance = float(s["guidance"] if s["fast"] else s["slow_guidance"])
    key = "|".join([s["model"], str(bool(s["fast"])), prompt, negative, str(seed),
                    f"{gen_w}x{gen_h}", str(steps), str(guidance), f"{width}x{height}"])
    digest = hashlib.sha256(key.encode("utf-8")).hexdigest()[:10]
    art_dir = out_dir / "illustrations"
    art_dir.mkdir(parents=True, exist_ok=True)
    path = art_dir / f"ai_scene_{index:02d}_v{variant}_{digest}.png"
    if path.exists():
        return path

    import torch
    pipe = _load(cfg)
    generator = torch.Generator(device="cpu").manual_seed(seed)
    log(f"diffusion: scene {index + 1} take {variant + 1} ({steps} steps, seed {seed})")
    with _LOCK, torch.inference_mode():
        latents = pipe(
            prompt=prompt, negative_prompt=negative, width=gen_w, height=gen_h,
            num_inference_steps=steps, guidance_scale=guidance, generator=generator,
            output_type="latent",
        ).images
        latents = latents.to(dtype=pipe.vae.dtype) / pipe.vae.config.scaling_factor
        decoded = pipe.vae.decode(latents).sample
        image = pipe.image_processor.postprocess(decoded, output_type="pil")[0]

    # Upscale to the video frame and crop to the exact aspect ratio.
    scale = max(width / image.width, height / image.height)
    big = image.resize((round(image.width * scale), round(image.height * scale)), Image.Resampling.LANCZOS)
    left = (big.width - width) // 2
    top = (big.height - height) // 2
    big.crop((left, top, left + width, top + height)).save(path, format="PNG", optimize=True)
    (art_dir / f"ai_scene_{index:02d}_v{variant}_{digest}.txt").write_text(
        f"prompt: {prompt}\nnegative: {negative}\nseed: {seed}\n", encoding="utf-8")
    return path
