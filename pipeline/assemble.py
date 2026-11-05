"""Stage 5b: cut clips to length, concatenate, mix voice and music, burn captions."""
import hashlib
import json
import random
from pathlib import Path

from . import motion
from .config import ROOT
from .util import log, run


def assemble(out_dir: Path, plan: list, scene_audios: list, cfg: dict) -> Path:
    vid = cfg["video"]
    w, h = vid["resolution"]
    work = out_dir / "work"
    work.mkdir(exist_ok=True)

    # 1. Every segment becomes a uniform silent mp4 of exactly the right length.
    segments = []
    n = 0
    strength = float(motion.settings(cfg)["strength"])
    for scene_clips in plan:
        for item in scene_clips:
            clip, seconds = Path(item[0]), float(item[1])
            shot = item[2] if len(item) > 2 and item[2] else None
            segment_key = f"{clip.resolve()}|{seconds:.3f}|{w}x{h}|{vid['fps']}|{json.dumps(shot, sort_keys=True)}|{strength}"
            segment_hash = hashlib.sha256(segment_key.encode()).hexdigest()[:10]
            seg = work / f"seg_{n:03d}_{segment_hash}.mp4"
            if not seg.exists():
                if clip.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp"}:
                    input_args = ["-loop", "1", "-framerate", str(vid["fps"]), "-i", str(clip)]
                    filters = motion.zoompan_filter(shot, w, h, vid["fps"], seconds, strength)
                else:
                    input_args = ["-stream_loop", "-1", "-i", str(clip)]
                    filters = f"scale={w}:{h}:force_original_aspect_ratio=increase,crop={w}:{h},fps={vid['fps']},setsar=1"
                run(["ffmpeg", "-y", "-loglevel", "error", *input_args,
                     "-t", f"{seconds:.3f}", "-an", "-vf", filters,
                     "-c:v", "libx264", "-preset", "veryfast", "-crf", "20",
                     "-pix_fmt", "yuv420p", str(seg)])
            segments.append(seg)
            n += 1
    log(f"assemble: {len(segments)} segments cut")

    # 2. Concatenate the picture track.
    (work / "video_list.txt").write_text(
        "".join(f"file '{_concat_name(s.name)}'\n" for s in segments), encoding="utf-8")
    run(["ffmpeg", "-y", "-loglevel", "error", "-f", "concat", "-safe", "0",
         "-i", "video_list.txt", "-c", "copy", "video.mp4"], cwd=work)

    # 3. Concatenate the narration, then mix in music if any is available.
    (work / "audio_list.txt").write_text(
        "".join(f"file '../audio/{_concat_name(a.path)}'\n" for a in scene_audios),
        encoding="utf-8")
    run(["ffmpeg", "-y", "-loglevel", "error", "-f", "concat", "-safe", "0",
         "-i", "audio_list.txt", "-c:a", "aac", "-b:a", "192k", "voice.m4a"], cwd=work)
    audio = "voice.m4a"
    music = _pick_music(cfg)
    if music:
        log(f"assemble: mixing music {music.name}")
        run(["ffmpeg", "-y", "-loglevel", "error", "-i", "voice.m4a",
             "-stream_loop", "-1", "-i", str(music),
             "-filter_complex",
             f"[1:a]volume={cfg['music']['volume']}[m];"
             f"[0:a][m]amix=inputs=2:duration=first:dropout_transition=2:normalize=0[a]",
             "-map", "[a]", "-c:a", "aac", "-b:a", "192k", "mixed.m4a"], cwd=work)
        audio = "mixed.m4a"

    # 4. Mux, burning captions if enabled. Relative paths keep ffmpeg's filter parser happy.
    final = out_dir / "final.mp4"
    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-i", "video.mp4", "-i", audio]
    if cfg["captions"]["enabled"] and (out_dir / "captions.ass").exists():
        cmd += ["-vf", "ass=../captions.ass"]
    cmd += ["-c:v", "libx264", "-preset", "medium", "-crf", "20", "-pix_fmt", "yuv420p",
            "-c:a", "aac", "-b:a", "192k", "-shortest", "-movflags", "+faststart", str(final)]
    run(cmd, cwd=work)
    return final


def _concat_name(name: str) -> str:
    """Quote a file name for ffmpeg's concat demuxer list (single quotes)."""
    return name.replace("'", r"'\''")


def change_speed(video: Path, speed: float, source: Path | None = None) -> Path:
    """Re-time a finished video (picture, burned captions, and audio together).

    ``source`` lets the caller keep an untouched 1x render and derive every speed
    from it, so repeated adjustments do not compound.
    """
    speed = float(speed)
    if not 0.5 <= speed <= 2.0:
        raise ValueError("speed must be between 0.5 and 2.0")
    tmp = video.with_name(video.stem + ".speed.mp4")
    run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(source or video),
         "-filter_complex",
         f"[0:v]setpts=PTS/{speed:.4f}[v];[0:a]atempo={speed:.4f}[a]",
         "-map", "[v]", "-map", "[a]",
         "-c:v", "libx264", "-preset", "medium", "-crf", "20", "-pix_fmt", "yuv420p",
         "-c:a", "aac", "-b:a", "192k", "-movflags", "+faststart", str(tmp)])
    tmp.replace(video)
    return video


def _pick_music(cfg: dict) -> Path | None:
    if not cfg["music"]["enabled"]:
        return None
    tracks = sorted((ROOT / cfg["music"]["dir"]).glob("*.mp3"))
    return random.choice(tracks) if tracks else None
