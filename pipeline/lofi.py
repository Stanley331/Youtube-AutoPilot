"""Lofi music videos: one illustrated scene, gently animated, over an hour of generated music.

The picture is an AI panel (or a drawn one in a dry run) of the scene you
describe, "a student at a desk by a rainy window at night". It is turned into
a short seamless loop: the camera breathes in and out, rain streaks fall past
the window, the lamp light flickers, film grain moves. Because the loop joins
to itself exactly, the hour-long picture track is the loop file repeated with
stream copy, which takes seconds, not an hour of encoding. The music is the
tracklist from ``lofi_music`` joined with crossfades, with soft rain under it
when the scene is rainy, and the description carries the tracklist with
timestamps the way lofi channels do.
"""
from __future__ import annotations

import hashlib
import math
import random
from pathlib import Path

from PIL import Image, ImageDraw

from . import diffusion, lofi_library, lofi_music
from .llm import generate_json
from .util import duration_of, log, run, write_json

DEFAULTS = {
    "minutes": 60,                 # length of the finished video
    "loop_seconds": 24,            # the animated loop; longer loops hide the repeat better
    "rain": "auto",                # auto (when the scene mentions rain), always, never
    "ambience_volume": 0.10,       # rain under the music
    "crossfade_seconds": 3,        # between tracks
    "grain": True,
    "flicker": True,
    "style": "lofi anime illustration, cozy, warm lamp light, soft muted colors, gentle shading, "
             "highly detailed, wide cinematic composition",
    "title_suffix": "lofi hip hop beats to study and relax to",
    "tags": ["lofi", "lofi hip hop", "study music", "chill beats", "relax", "focus music",
             "lo-fi", "beats to study to", "ambient", "background music"],
}

SCENE_SCHEMA = {
    "type": "object",
    "properties": {
        "title": {"type": "string"},
        "thumbnail_text": {"type": "string"},
        "description": {"type": "string"},
        "tags": {"type": "array", "items": {"type": "string"}},
        "visual_description": {"type": "string"},
        "visual_query": {"type": "string"},
        "setting": {"type": "string"},
        "visual_subject": {"type": "string"},
        "action": {"type": "string"},
        "visual_objects": {"type": "array", "items": {"type": "string"}},
        "rainy": {"type": "boolean"},
        "track_titles": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["title", "thumbnail_text", "description", "tags", "visual_description", "visual_query",
                 "setting", "visual_subject", "action", "visual_objects", "rainy", "track_titles"],
    "additionalProperties": False,
}


def settings(cfg: dict) -> dict:
    merged = dict(DEFAULTS)
    merged.update(cfg.get("lofi", {}) or {})
    return merged


def _seed(text: str) -> int:
    return int(hashlib.sha256(text.encode("utf-8")).hexdigest()[:8], 16)


# ---------------------------------------------------------------------------
# Scene and metadata
# ---------------------------------------------------------------------------

def plan_scene(idea: str, cfg: dict, minutes: float, use_llm: bool = True) -> dict:
    """Title, description, tags and a storyboard scene from a one-line idea."""
    s = settings(cfg)
    idea = " ".join(str(idea or "").split()) or "a student studying at a desk by a window on a rainy night"
    fallback = {
        "title": f"{(idea[:1].upper() + idea[1:])[:48].rstrip(' ,')} | {s['title_suffix']} [{int(minutes)} min]"[:100],
        "thumbnail_text": "STUDY BEATS",
        "description": f"{int(minutes)} minutes of original lofi beats for studying, working and winding down. "
                       f"Scene: {idea}.",
        "tags": list(s["tags"]),
        "visual_description": idea,
        "visual_query": " ".join(idea.split()[:4]),
        "setting": "", "visual_subject": "", "action": "", "visual_objects": [],
        "rainy": "rain" in idea.lower(),
        "track_titles": [],
    }
    if not use_llm:
        return fallback
    system = (
        "You name and storyboard lofi music videos: an illustrated scene that loops under an hour of "
        "chill instrumental beats. Titles follow the genre's style, e.g. 'Rainy Night Study Session ☔ "
        "lofi hip hop beats to focus and relax', under 90 characters, honest, no clickbait. "
        "The scene must be one calm, specific picture (a person or a place, time of day, light, weather, "
        "a few props) that an illustrator could draw in a cozy anime style. Track titles are short, "
        "evocative, two or three words, all different."
    )
    prompt = (f"Scene idea: {idea}\nVideo length: about {int(minutes)} minutes.\n\n"
              "Return the title, thumbnail_text (at most 3 words in capitals), a two-paragraph description "
              "(what the music is for and the scene; no timestamps), 8 to 12 tags, the storyboard fields "
              "(visual_description, visual_query of 2-4 nouns, setting, visual_subject, action, 1-3 "
              "visual_objects), whether the scene is rainy, and 12 track titles.")
    try:
        result = generate_json(system, prompt, SCENE_SCHEMA, effort="medium")
    except Exception as exc:
        log(f"lofi: the model could not plan the scene ({type(exc).__name__}: {exc}); using the idea as is")
        return fallback
    planned = dict(fallback)
    for key in SCENE_SCHEMA["properties"]:
        value = result.get(key)
        if value in (None, "", []):
            continue
        planned[key] = value
    planned["title"] = str(planned["title"])[:100]
    planned["tags"] = [str(t) for t in planned["tags"]][:20] or list(s["tags"])
    planned["track_titles"] = [str(t) for t in planned["track_titles"]]
    planned["visual_objects"] = [str(t) for t in planned["visual_objects"]][:3]
    planned["rainy"] = bool(planned["rainy"])
    return planned


def scene_dict(planned: dict) -> dict:
    """The storyboard scene in the shape the illustrators expect."""
    return {
        "narration": "", "visual_query": planned.get("visual_query", ""),
        "visual_description": planned.get("visual_description", ""),
        "setting": planned.get("setting", ""), "visual_subject": planned.get("visual_subject", ""),
        "action": planned.get("action", ""), "visual_objects": list(planned.get("visual_objects", [])),
        "on_screen_text": "",
    }


def render_scene_image(scene: dict, out_dir: Path, cfg: dict, source: str, context: dict) -> Path:
    """The still the loop is built from: an AI panel, a drawn panel, or a plain slide."""
    s = settings(cfg)
    width, height = cfg["video"]["resolution"]
    if source == "diffusion":
        if not diffusion.available():
            raise RuntimeError(diffusion.install_hint())
        ctx = dict(context, art_style=s["style"], content_type="story", main_character="")
        return diffusion.generate_panel(scene, out_dir, 0, cfg, ctx)
    if source == "illustration":
        from .illustrations import render_panel
        return render_panel(scene, out_dir, 0, cfg, dict(context, content_type="story", no_caption=True))
    art = out_dir / "illustrations"
    art.mkdir(parents=True, exist_ok=True)
    path = art / "scene_00_placeholder.png"
    if not path.exists():
        img = Image.new("RGB", (width, height), (24, 28, 44))
        draw = ImageDraw.Draw(img)
        draw.rectangle((width * 0.1, height * 0.55, width * 0.9, height * 0.95), fill=(44, 38, 58))
        draw.ellipse((width * 0.68, height * 0.18, width * 0.78, height * 0.36), fill=(255, 210, 120))
        img.save(path)
    return path


# ---------------------------------------------------------------------------
# The loop
# ---------------------------------------------------------------------------

def rain_texture(out_dir: Path, width: int, height: int, seed: int) -> Path:
    """A transparent PNG twice the frame height with thin streaks; it scrolls and tiles."""
    path = out_dir / "work" / "rain.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        return path
    rng = random.Random(seed)
    img = Image.new("RGBA", (width, height * 2), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    for _ in range(int(width * height / 4200)):
        x = rng.randrange(width)
        y = rng.randrange(height)
        length = rng.randint(int(height * 0.03), int(height * 0.08))
        alpha = rng.randint(40, 110)
        w = 1 if rng.random() < 0.7 else 2
        for yy in (y, y + height):   # draw twice so the tile joins seamlessly
            draw.line((x, yy, x - length * 0.08, yy + length), fill=(220, 230, 255, alpha), width=w)
    img.save(path)
    return path


def make_loop(image: Path, out_dir: Path, cfg: dict, rainy: bool, seed: int) -> Path:
    """A seamless animated loop of the still: breathing zoom, rain, lamp flicker, grain."""
    s = settings(cfg)
    vid = cfg["video"]
    w, h = vid["resolution"]
    fps = int(vid["fps"])
    seconds = max(6, int(s["loop_seconds"]))
    frames = seconds * fps
    work = out_dir / "work"
    work.mkdir(parents=True, exist_ok=True)
    key = hashlib.sha256(f"{image.resolve()}|{w}x{h}|{fps}|{seconds}|{rainy}|{s['grain']}|{s['flicker']}|{seed}"
                         .encode("utf-8")).hexdigest()[:10]
    loop = work / f"loop_{key}.mp4"
    if loop.exists():
        return loop
    big_w, big_h = int(round(w * 1.5 / 2)) * 2, int(round(h * 1.5 / 2)) * 2
    # Every motion is periodic in the loop length, so the last frame leads back into the first.
    zoom = f"(1.05+0.025*sin(2*PI*on/{frames}))"
    pan_x = f"(iw-iw/zoom)*(0.5+0.06*sin(2*PI*on/{frames}+1.0))"
    pan_y = f"(ih-ih/zoom)*(0.5+0.04*cos(2*PI*on/{frames}))"
    chain = [
        f"scale={big_w}:{big_h}:force_original_aspect_ratio=increase,crop={big_w}:{big_h}",
        f"zoompan=z='{zoom}':x='{pan_x}':y='{pan_y}':d=1:s={big_w}x{big_h}:fps={fps}",
        f"scale={w}:{h}:flags=lanczos,setsar=1",
    ]
    if s["flicker"]:
        cycles = max(1, round(seconds / 2.7))
        chain.append(f"eq=brightness='0.012*sin(2*PI*t*{cycles}/{seconds})':eval=frame")
    base = ",".join(chain)
    inputs = ["-loop", "1", "-framerate", str(fps), "-i", str(image)]
    if rainy:
        texture = rain_texture(out_dir, w, h, seed)
        cycles = max(1, round(seconds * 700 / h))       # whole scrolls per loop keep the seam invisible
        speed = cycles * h / seconds
        inputs += ["-loop", "1", "-framerate", str(fps), "-i", str(texture)]
        graph = (f"[0:v]{base}[bg];"
                 f"[bg][1:v]overlay=x='-{int(h * 0.02)}+mod(t*{speed * 0.08:.3f},{h})':"
                 f"y='-{h}+mod(t*{speed:.3f},{h})':shortest=1:format=auto[v0]")
    else:
        graph = f"[0:v]{base}[v0]"
    post = ["vignette=PI/4.5"]
    if s["grain"]:
        post.append("noise=alls=7:allf=t+u")
    graph += f";[v0]{','.join(post)},format=yuv420p[v]"
    log(f"lofi: rendering a {seconds}s loop at {w}x{h} ({'rain, ' if rainy else ''}breathing camera)")
    run(["ffmpeg", "-y", "-loglevel", "error", *inputs, "-filter_complex", graph, "-map", "[v]",
         "-t", str(seconds), "-r", str(fps), "-c:v", "libx264", "-preset", "medium", "-crf", "18",
         "-g", str(frames), "-keyint_min", str(frames), "-movflags", "+faststart", str(loop)])
    return loop


# ---------------------------------------------------------------------------
# Assembly
# ---------------------------------------------------------------------------

def track_starts(tracks: list, crossfade: float) -> list[float]:
    starts, t = [], 0.0
    for i, track in enumerate(tracks):
        starts.append(round(t, 2))
        t += float(track.seconds) - (crossfade if i < len(tracks) - 1 else 0.0)
    return starts


def tracklist_lines(tracks: list, cfg: dict) -> list[str]:
    from .longform import format_timestamp
    s = settings(cfg)
    return [f"{format_timestamp(start)} {track.title}"
            for track, start in zip(tracks, track_starts(tracks, float(s["crossfade_seconds"])))]


def credit_lines(script: dict, tracks: list, ambience: Path | None) -> list[str]:
    """Attribution for the description: Pixabay asks for none, but naming creators is fair."""
    lines = []
    credit = script.get("clip_credit") or {}
    if credit.get("user"):
        page = f" ({credit['pageURL']})" if credit.get("pageURL") else ""
        lines.append(f"Animation: {credit['user']} via Pixabay{page}")
    if any(getattr(t, "engine", "") == "pixabay" for t in tracks):
        lines.append("Music: royalty-free tracks from Pixabay (pixabay.com/music), Pixabay Content License")
    if ambience is not None:
        lines.append("Rain ambience: Pixabay sound effects")
    return lines


def build_video(out_dir: Path, loop: Path, tracks: list, cfg: dict, rainy: bool, seed_text: str,
                ambience: Path | None = None) -> Path:
    s = settings(cfg)
    work = out_dir / "work"
    work.mkdir(parents=True, exist_ok=True)
    music_dir = out_dir / "music"
    xf = float(s["crossfade_seconds"])

    # 1. Music: the tracks joined with crossfades.
    inputs = []
    for track in tracks:
        inputs += ["-i", str(music_dir / track.path)]   # library tracks carry absolute paths
    if len(tracks) == 1:
        graph = "[0:a]anull[m]"
    else:
        graph, prev = "", "[0:a]"
        for i in range(1, len(tracks)):
            out = "[m]" if i == len(tracks) - 1 else f"[x{i}]"
            graph += f"{prev}[{i}:a]acrossfade=d={xf}:c1=tri:c2=tri{out};"
            prev = out
        graph = graph.rstrip(";")
    run(["ffmpeg", "-y", "-loglevel", "error", *inputs, "-filter_complex", graph, "-map", "[m]",
         "-c:a", "aac", "-b:a", "192k", str(work / "music.m4a")])
    total = duration_of(work / "music.m4a")

    # 2. Rain under the music, when the scene is rainy.
    audio = work / "music.m4a"
    if rainy and float(s["ambience_volume"]) > 0:
        if ambience is not None and Path(ambience).exists():
            rain = Path(ambience)
        else:
            import soundfile as sf
            rain = work / "rain.wav"
            if not rain.exists():
                sf.write(str(rain), lofi_music.rain_ambience(90, seed_text), lofi_music.SR, subtype="PCM_16")
        run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(work / "music.m4a"),
             "-stream_loop", "-1", "-i", str(rain), "-filter_complex",
             f"[1:a]volume={float(s['ambience_volume'])}[r];"
             f"[0:a][r]amix=inputs=2:duration=first:dropout_transition=3:normalize=0[a]",
             "-map", "[a]", "-c:a", "aac", "-b:a", "192k", str(work / "mixed.m4a")])
        audio = work / "mixed.m4a"

    # 3. Picture: the loop repeated by stream copy, then muxed with the audio.
    loop_seconds = duration_of(loop)
    repeats = int(math.ceil(total / loop_seconds)) + 1
    (work / "loop_list.txt").write_text("".join(f"file '{loop.name}'\n" for _ in range(repeats)), encoding="utf-8")
    run(["ffmpeg", "-y", "-loglevel", "error", "-f", "concat", "-safe", "0", "-i", "loop_list.txt",
         "-c", "copy", "picture.mp4"], cwd=work)
    final = out_dir / "final.mp4"
    run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(work / "picture.mp4"), "-i", str(audio),
         "-map", "0:v", "-map", "1:a", "-c:v", "copy", "-c:a", "copy", "-shortest",
         "-movflags", "+faststart", str(final)])
    (work / "picture.mp4").unlink(missing_ok=True)
    return final


PRESETS = [
    "a student at a desk by a rainy window at night, warm lamp, a cup of tea",
    "a quiet cafe corner on a rainy afternoon, steam rising from a coffee cup",
    "a cat asleep on a bookshop counter in late afternoon sun",
    "a night train window, city lights sliding past, a sleepy passenger with headphones",
    "a rooftop at dusk over a glowing city, headphones and an open notebook",
    "a cabin fireplace on a snowy evening, a blanket and an open book",
]

STAGE_SETTINGS = ("minutes", "visuals", "music", "rain", "variant", "ambience")
PICTURE_SOURCES = ("pixabay", "diffusion", "illustration", "placeholder")


def project_settings(script: dict, cfg: dict) -> dict:
    """The choices saved with a project (minutes, art source, music engine, rain, scene take)."""
    s = settings(cfg)
    merged = {"minutes": float(s["minutes"]), "visuals": str(s.get("picture", "pixabay")),
              "music": str(lofi_music.settings(cfg)["engine"]), "rain": str(s["rain"]), "variant": 0,
              "ambience": str(s.get("ambience_source", "library"))}
    merged.update({k: v for k, v in (script.get("settings") or {}).items() if k in STAGE_SETTINGS and v is not None})
    return merged


def _load(out_dir: Path) -> dict:
    from .util import read_json
    path = Path(out_dir) / "script.json"
    if not path.exists():
        raise RuntimeError(f"no script.json in {out_dir}; plan the lofi video first")
    return read_json(path)


def _save(script: dict, out_dir: Path) -> None:
    from .script import save_script
    save_script(script, out_dir)


def plan_project(idea: str, out_dir: Path, cfg: dict, minutes: float | None = None,
                 visuals: str | None = None, music_engine: str | None = None, rain: str | None = None,
                 use_llm: bool = True) -> dict:
    """Stage 1: title, description, tags, storyboard and track names; writes script.json, renders nothing."""
    s = settings(cfg)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    minutes = float(minutes or s["minutes"])
    planned = plan_scene(idea, cfg, minutes, use_llm=use_llm)
    script = {
        "format": "lofi", "content_type": "music", "title": planned["title"],
        "alternative_titles": [], "thumbnail_text": planned["thumbnail_text"],
        "description": planned["description"], "tags": planned["tags"], "main_character": "",
        "category_id": "10", "lofi_plan": planned, "idea": idea,
        "settings": {"minutes": minutes, "visuals": visuals or str(s.get("picture", "pixabay")),
                     "music": music_engine or str(lofi_music.settings(cfg)["engine"]),
                     "rain": rain or str(s["rain"]), "variant": 0,
                     "ambience": str(s.get("ambience_source", "library"))},
        "scenes": [scene_dict(planned)], "claims_to_verify": [], "tracks": [],
    }
    _save(script, out_dir)
    log(f"lofi: planned '{script['title']}' ({int(minutes)} min)")
    return script


def is_rainy(script: dict, cfg: dict) -> bool:
    rain = str(project_settings(script, cfg)["rain"]).lower()
    return {"always": True, "never": False}.get(rain, bool((script.get("lofi_plan") or {}).get("rainy")))


def draw_scene(out_dir: Path, cfg: dict, variant: int | None = None, visuals: str | None = None) -> Path:
    """Stage 2: the still. A new ``variant`` is another take of the AI panel (same prompt, new seed)."""
    out_dir = Path(out_dir)
    script = _load(out_dir)
    st = project_settings(script, cfg)
    if variant is not None:
        st["variant"] = int(variant)
    if visuals:
        st["visuals"] = visuals
    scene = script["scenes"][0]
    log(f"lofi: scene via {st['visuals']} (take {st['variant'] + 1}): {scene.get('visual_description', '')[:80]}")
    if st["visuals"] == "pixabay":
        from .visuals import extract_thumbnail
        planned = script.get("lofi_plan") or {}
        candidates = lofi_library.search_clips(cfg, planned, count=12)
        if not candidates:
            raise RuntimeError("Pixabay returned no animated clips for this scene; try another picture source")
        chosen = candidates[int(st["variant"]) % len(candidates)]
        clip = lofi_library.fetch_clip(cfg, chosen)
        art = out_dir / "illustrations"
        art.mkdir(parents=True, exist_ok=True)
        frame = art / f"clip_{chosen['id']}.jpg"
        if not frame.exists():
            extract_thumbnail(clip, frame, at_seconds=1.0)
        script["scene_clip"] = str(clip)
        script["clip_credit"] = {k: chosen.get(k) for k in ("id", "user", "pageURL", "tags", "duration")}
        script["settings"] = dict(script.get("settings") or {}, **st)
        script["scene_image"] = str(frame)
        _save(script, out_dir)
        log(f"lofi: Pixabay clip {chosen['id']} by {chosen.get('user', '')} ({chosen['duration']:.0f}s, "
            f"used {chosen['uses']} times before)")
        return frame
    script.pop("scene_clip", None)
    script.pop("clip_credit", None)
    if st["visuals"] == "diffusion":
        if not diffusion.available():
            raise RuntimeError(diffusion.install_hint())
        ctx = dict(script, art_style=settings(cfg)["style"], content_type="story", main_character="")
        image = diffusion.generate_panel(scene, out_dir, 0, cfg, ctx, variant=int(st["variant"]))
    else:
        image = render_scene_image(scene, out_dir, cfg, st["visuals"], script)
    script["settings"] = dict(script.get("settings") or {}, **st)
    script["scene_image"] = str(image)
    _save(script, out_dir)
    return image


def preview_track(out_dir: Path, cfg: dict, music_engine: str | None = None):
    """Stage 3: the first track, which the full render reuses (same seed, same file)."""
    out_dir = Path(out_dir)
    script = _load(out_dir)
    st = project_settings(script, cfg)
    engine = music_engine or st["music"]
    seed_text = script.get("title") or script.get("idea", "")
    if engine == "pixabay":
        return lofi_library.pick_tracks(cfg, 0.1, seed_text)[0]
    titles = (script.get("lofi_plan") or {}).get("track_titles")
    track = lofi_music.generate_tracks(cfg, out_dir / "music", 0.1, seed_text, titles=titles, engine=engine)[0]
    return track


def render_project(out_dir: Path, cfg: dict, minutes: float | None = None,
                   music_engine: str | None = None, visuals: str | None = None) -> dict:
    """Stage 4: loop, every track, the full video, thumbnail. Reuses whatever earlier stages made."""
    from .thumbnail import make_thumbnail

    out_dir = Path(out_dir)
    script = _load(out_dir)
    st = project_settings(script, cfg)
    if minutes:
        st["minutes"] = float(minutes)
    if music_engine:
        st["music"] = music_engine
    if visuals:
        st["visuals"] = visuals
    script["settings"] = dict(script.get("settings") or {}, **st)
    _save(script, out_dir)
    planned = script.get("lofi_plan") or {}
    seed_text = script.get("title") or script.get("idea", "")
    rainy = is_rainy(script, cfg)

    slug = out_dir.name
    if st["visuals"] == "pixabay":
        if not (script.get("scene_clip") and Path(script["scene_clip"]).exists()):
            draw_scene(out_dir, cfg)
            script = _load(out_dir)
        loop = lofi_library.seamless_loop(Path(script["scene_clip"]), out_dir, cfg)
        if script.get("clip_credit", {}).get("id") is not None:
            lofi_library.record("clips", str(script["clip_credit"]["id"]), slug)
    else:
        image = Path(script["scene_image"]) if script.get("scene_image") and Path(script["scene_image"]).exists() \
            else draw_scene(out_dir, cfg)
        loop = make_loop(image, out_dir, cfg, rainy, _seed(seed_text))
    if st["music"] == "pixabay":
        tracks = lofi_library.pick_tracks(cfg, float(st["minutes"]), seed_text)
        lofi_library.record_tracks(tracks, slug)
    else:
        tracks = lofi_music.generate_tracks(cfg, out_dir / "music", float(st["minutes"]), seed_text,
                                            titles=planned.get("track_titles"), engine=st["music"])
    ambience = None
    if rainy and st.get("ambience", "library") == "library":
        ambience = lofi_library.pick_ambience(cfg, seed_text)
        if ambience:
            lofi_library.record("ambience", ambience.name, slug)
        else:
            log("lofi: the ambience library is empty; synthesizing rain instead")
    script["tracks"] = [tr.to_dict() for tr in tracks]
    script["tracklist"] = tracklist_lines(tracks, cfg)
    script["credits"] = credit_lines(script, tracks, ambience)
    _save(script, out_dir)
    log(f"lofi: {len(tracks)} tracks, {sum(tr.seconds for tr in tracks) / 60:.1f} minutes of music; assembling")
    final = build_video(out_dir, loop, tracks, cfg, rainy, seed_text, ambience=ambience)
    script["duration"] = round(duration_of(final), 2)
    _save(script, out_dir)
    write_json(out_dir / "clip_plan.json", [[[str(loop), script["duration"]]]])
    make_thumbnail(final, script["thumbnail_text"], out_dir / "thumbnail.jpg", cfg, at_seconds=2.0)
    log(f"lofi: {final} ({script['duration'] / 60:.1f} min)")
    return script


def make_lofi(idea: str, out_dir: Path, cfg: dict, minutes: float | None = None,
              music_engine: str | None = None, visuals: str | None = None, use_llm: bool = True,
              seed_text: str = "") -> dict:
    """All stages in one go (the CLI): plan unless the project exists, then render."""
    out_dir = Path(out_dir)
    if not (out_dir / "script.json").exists():
        plan_project(idea, out_dir, cfg, minutes=minutes, visuals=visuals, music_engine=music_engine,
                     use_llm=use_llm)
    else:
        log(f"lofi: resuming '{_load(out_dir).get('title', '')}'")
    return render_project(out_dir, cfg, minutes=minutes, music_engine=music_engine, visuals=visuals)
