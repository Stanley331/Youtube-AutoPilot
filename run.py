#!/usr/bin/env python3
"""YouTube autopilot. One command takes an idea to a finished video for review.

  python run.py make                      # pick a topic, write, voice, edit, thumbnail (a Short)
  python run.py make --length long        # a ten-minute 16:9 video with chapters
  python run.py make --length doodle --idea "..."  # 9-11 min doodle explainer, drawings made on Kaggle
  python run.py cut <slug>                # Shorts cut from a finished long video
  python run.py lofi --idea "..."         # an hour of generated lofi music over an animated scene
  python run.py make --idea "..."         # expand a rough premise into a full script and video
  python run.py make --idea "..." --stop-after script  # generate and review only the script
  python run.py make --stop-after script  # write the script, then stop so you can edit it
  python run.py make --slug <slug>        # resume an existing output/<slug>
  python run.py make --slug <slug> --stop-after media_selection  # fetch media candidates and stop
  python run.py make --dry-run            # no API keys needed: sample story, local voice, drawn panels
  python run.py ideas                     # print 10 topic ideas
  python run.py publish <slug>            # upload output/<slug>/final.mp4 to YouTube
  python run.py list                      # everything produced so far
"""
import argparse
import shutil
import sys
from datetime import date
from pathlib import Path

from pipeline import formats, history
from pipeline.config import OUTPUT_DIR, ROOT, load_config
from pipeline.util import log, read_json, slugify, write_json

STAGES = ["topic", "script", "voice", "media_selection", "visuals", "video", "thumbnail"]


def _unique_out_dir(slug: str) -> Path:
    """Never silently reuse another run's folder (and its audio/media) for a new script."""
    out_dir = OUTPUT_DIR / slug
    n = 2
    while out_dir.exists():
        out_dir = OUTPUT_DIR / f"{slug}-{n}"
        n += 1
    return out_dir


def cmd_ideas(args, cfg):
    from pipeline.topics import generate_ideas
    for i, idea in enumerate(generate_ideas(cfg, count=10), 1):
        print(f"{i:2d}. [{idea['kind']}] {idea['title']}  [{idea['search_demand']} demand]")
        print(f"    {idea['angle']}")


def cmd_voices(args, cfg):
    from pipeline.voices import KOKORO_VOICES, all_voices, describe, pool_for, usage
    for engine in ("kokoro", "gemini"):
        last = usage(engine)
        story, fact = pool_for(cfg, engine, "story"), pool_for(cfg, engine, "fact")
        print(f"\n{engine}  (rotation: {'on' if cfg['voice'].get('rotate', True) else 'off'})")
        for name in all_voices(cfg, engine):
            tags = [t for t, pool in (("story", story), ("fact", fact)) if name in pool]
            grade = f" grade {KOKORO_VOICES[name][1]}" if engine == "kokoro" else ""
            used = f"  last used {last[name]}" if name in last else ""
            print(f"  {describe(engine, name):48s} {'/'.join(tags) or '-':10s}{grade}{used}")
    print("\nPick one with: python run.py make --voice kokoro --voice-name bm_fable")


class _Stopped(Exception):
    """Raised by --stop-after so a batch can carry on with the next video."""

    def __init__(self, out_dir: Path, hint: str):
        super().__init__(hint)
        self.out_dir = out_dir


def cmd_make(args, cfg):
    try:
        make_video(args, cfg)
    except _Stopped as stop:
        log(f"stopped as requested. Next: {stop}")


def cmd_batch(args, cfg):
    """Render several videos in one go: fresh topics, alternating kinds, rotating narrators."""
    from pipeline.topics import pick_topics
    count = max(1, int(args.count))
    length = getattr(args, "length", "short") or "short"
    if args.dry_run:
        topics = [None] * count
    else:
        log(f"batch: asking the configured model for {count} fresh {length} topics")
        topics = pick_topics(cfg, count, kind=None if args.format == "auto" else args.format, length=length)
        if len(topics) < count:
            log(f"batch: only {len(topics)} distinct topics found; making those")
    results = []
    for i, topic in enumerate(topics, 1):
        title = topic["title"] if topic else "dry-run sample"
        log(f"batch: video {i}/{len(topics)}: {title}")
        one = argparse.Namespace(
            idea=None, slug=None, stop_after=args.stop_after, dry_run=args.dry_run,
            voice=args.voice, voice_name=None, visuals=args.visuals, force=False, topic_obj=topic,
            format=args.format if args.format != "auto" else (topic or {}).get("kind", "auto"),
            length=length,
        )
        try:
            out_dir = make_video(one, cfg)
            results.append((title, out_dir.name, "ready"))
        except _Stopped as stop:
            results.append((title, stop.out_dir.name, f"stopped after {args.stop_after}"))
        except KeyboardInterrupt:
            log("batch: interrupted")
            break
        except Exception as exc:  # one bad video must not sink the batch
            log(f"batch: FAILED {title}: {type(exc).__name__}: {str(exc)[:300]}")
            results.append((title, "-", f"failed: {type(exc).__name__}"))
    print()
    print("BATCH SUMMARY")
    for title, slug, status in results:
        print(f"  {status:<28} {slug:<60} {title}")
    print("\nReview each in the browser UI, then: python run.py queue --publish")


def make_video(args, cfg) -> Path:
    dry = args.dry_run
    # The format (short or long) decides resolution, caption size, script length and more;
    # a resumed project keeps the one written into its script.json.
    length = getattr(args, "length", None) or "short"
    if args.slug and (OUTPUT_DIR / args.slug / "script.json").exists():
        length = formats.of_script(read_json(OUTPUT_DIR / args.slug / "script.json"))
    cfg = formats.apply(cfg, length)
    voice_engine = args.voice or ("local" if dry else cfg["voice"]["engine"])
    # The dry run must work with no API keys, so it draws its panels locally.
    visual_source = args.visuals or ("illustration" if dry else cfg["visuals"]["source"])
    if length == "doodle":
        from pipeline import doodle
        visual_source = _doodle_backend(args, cfg, dry)

    # ---- topic + script -------------------------------------------------
    if args.slug:
        out_dir = OUTPUT_DIR / args.slug
        if not (out_dir / "script.json").exists():
            sys.exit(f"no script.json in {out_dir}")
        script = read_json(out_dir / "script.json")
        log(f"resuming '{script['title']}'")
        # Resuming keeps the narration already made: its engine, unless --voice asks for another.
        saved_engine = (script.get("voice") or {}).get("engine")
        if not args.voice and saved_engine:
            voice_engine = saved_engine
    else:
        if dry:
            fixture = {"long": "sample_long_script.json", "doodle": "sample_doodle_script.json"}.get(
                length, "sample_script.json")
            script = read_json(ROOT / "fixtures" / fixture)
            topic = {"title": script["title"]}
        else:
            from pipeline.script import write_script
            from pipeline.topics import pick_topic
            if getattr(args, "topic_obj", None):
                topic = dict(args.topic_obj)
                topic["format"] = args.format if args.format != "auto" else topic.get("kind", "auto")
            elif args.idea:
                topic = {"title": args.idea, "angle": "", "why_it_will_perform": "",
                         "format": args.format}
            else:
                log("topic: asking the configured model for ideas")
                topic = pick_topic(cfg, kind=None if args.format == "auto" else args.format, length=length)
                topic["format"] = args.format if args.format != "auto" else topic.get("kind", "auto")
            log(f"topic: {topic['title']}")
            if length in ("long", "doodle"):
                from pipeline.longform import long_script_provider, write_long_script
                who = long_script_provider() or "the configured model"
                log(f"script: writing a {length} script with {who} (outline, then one pass per chapter; "
                    "a few minutes)")
                script = write_long_script(cfg, topic, doodle=length == "doodle")
            else:
                log("script: writing (this takes a minute)")
                script = write_script(cfg, topic)
        script["format"] = length
        slug = f"{date.today().isoformat()}-{slugify(script['title'])}"
        out_dir = OUTPUT_DIR / slug
        if out_dir.exists():
            if args.force:
                shutil.rmtree(out_dir)
            else:
                out_dir = _unique_out_dir(slug)
                slug = out_dir.name
        out_dir.mkdir(parents=True, exist_ok=True)
        from pipeline.diffusion import pick_style
        from pipeline.script import save_script
        style = pick_style(cfg, script) if length != "doodle" else ""
        if style:
            script["art_style"] = style   # AI panels: one look per video, rotated across a batch
        save_script(script, out_dir)
        if args.stop_after == "script" or length == "doodle":
            # The studio continues this project step by step (it starts scripts in the background).
            write_json(out_dir / "ui_state.json", {"visuals": visual_source, "kind": length})
        history.add(slug, script["title"], topic["title"], kind=script.get("content_type", ""), fmt=length)
        words = sum(len(s["narration"].split()) for s in script["scenes"])
        log(f"script: {len(script['scenes'])} scenes, {words} words -> {out_dir / 'script.md'}")
    if args.stop_after == "script":
        _stop(out_dir, "edit script.json or script.md, then rerun with --slug " + out_dir.name)

    # ---- voice ----------------------------------------------------------
    from pipeline.script import save_script
    if length == "doodle" and doodle.sync_beats(script):
        log("doodle: narration was edited; beats rebuilt from it")
        save_script(script, out_dir)
    from pipeline.voice import synthesize_all
    from pipeline.voices import describe, resolve_voice
    voice = resolve_voice(script, cfg, voice_engine, args.voice_name)
    if script.get("voice") != voice:
        script["voice"] = voice
        save_script(script, out_dir)
    history.set_voice(out_dir.name, voice["engine"], voice["name"])
    log(f"voice: narrator {describe(voice['engine'], voice['name'])} via {voice['engine']}")
    audios = synthesize_all(script["scenes"], out_dir, cfg, voice_engine, voice["name"])
    total = sum(a.duration for a in audios)
    if length == "doodle" and not dry:   # the sample is a minute long; nothing to fit
        audios = doodle.fit_duration(out_dir, audios, cfg)
        total = sum(a.duration for a in audios)
    log(f"voice: {total / 60:.1f} minutes of narration")
    if script.get("chapters"):
        from pipeline.longform import chapter_times, format_timestamp
        script["chapter_times"] = chapter_times(script, [a.duration for a in audios])
        save_script(script, out_dir)
        log("chapters: " + " | ".join(f"{format_timestamp(c['seconds'])} {c['title']}"
                                      for c in script["chapter_times"]))
    if args.stop_after == "voice":
        _stop(out_dir, "listen to output/.../audio/*.mp3")

    if length == "doodle":
        return _finish_doodle(args, cfg, script, out_dir, audios, visual_source)

    # ---- media candidates + selection ----------------------------------
    from pipeline.visuals import (clips_for_scenes, deserialize_candidates,
                                  fetch_candidates_for_scenes, serialize_candidates)

    # Load existing selections if resuming
    media_plan_path = out_dir / "media_plan.json"
    selections = None
    custom_media = None
    if media_plan_path.exists():
        plan_data = read_json(media_plan_path)
        selections = {int(k): v for k, v in plan_data.get("selections", {}).items() if v}
        custom_media = {int(k): Path(v) for k, v in plan_data.get("custom_media", {}).items() if v}

    # Fetch candidates if not already cached
    candidates_path = out_dir / "media_candidates.json"
    if not candidates_path.exists() or args.force:
        if visual_source == "pixabay":
            from pipeline.ranking import ensure_search_terms
            log("media: writing stock search terms per scene")
            script = ensure_search_terms(script, cfg, out_dir)
        log(f"media: fetching candidates via {visual_source}")
        candidates = fetch_candidates_for_scenes(
            script["scenes"], out_dir, cfg,
            candidates_per_scene=int(cfg["visuals"].get("candidates_per_scene", 8)),
            source=visual_source, script_context=script,
        )
        write_json(candidates_path, serialize_candidates(candidates))
    else:
        # Reconstruct candidate objects from cached JSON
        candidates = deserialize_candidates(read_json(candidates_path))

    if args.stop_after == "media_selection":
        log(f"media: candidates ready for {len(candidates)} scenes -> {out_dir / 'media_candidates.json'}")
        _stop(out_dir, "select media per scene in the UI, then rerun with --slug " + out_dir.name)

    # ---- build final plan from selections or auto -----------------------
    plan = clips_for_scenes(
        script["scenes"], [a.duration for a in audios], out_dir, cfg,
        visual_source, script_context=script,
        selections=selections, custom_media=custom_media,
    )
    from pipeline.motion import plan_to_json
    write_json(out_dir / "clip_plan.json", plan_to_json(plan))
    if args.stop_after == "visuals":
        _stop(out_dir, "check clip_plan.json")

    # ---- captions + assembly -------------------------------------------
    from pipeline.assemble import assemble
    from pipeline.captions import build_ass
    build_ass(audios, out_dir / "captions.ass", cfg)
    log("assemble: rendering video")
    final = assemble(out_dir, plan, audios, cfg)
    log(f"assemble: {final}")
    if args.stop_after == "video":
        _stop(out_dir, f"watch {final}, then rerun with --slug {out_dir.name} for the thumbnail")

    # ---- thumbnail ------------------------------------------------------
    from pipeline.thumbnail import make_thumbnail
    thumb = make_thumbnail(final, script["thumbnail_text"], out_dir / "thumbnail.jpg", cfg)
    log(f"thumbnail: {thumb}")

    print()
    print("REVIEW CHECKLIST")
    print(f"  1. Watch:        {final}")
    print(f"  2. Thumbnail:    {thumb}")
    print(f"  3. Fact-check:   {out_dir / 'script.md'}  (see 'Claims to verify')")
    print(f"  4. Publish:      python run.py publish {out_dir.name}   (add --next-slot to schedule)")
    if length == "long":
        print(f"  5. Shorts:       python run.py cut {out_dir.name}   (excerpts re-rendered vertically)")
    return out_dir


def _doodle_backend(args, cfg, dry: bool) -> str:
    """kaggle | diffusion | placeholder: the flag, else what the project was planned with, else config."""
    from pipeline import doodle
    choice = args.visuals
    state = OUTPUT_DIR / args.slug / "ui_state.json" if args.slug else None
    if not choice and state and state.exists():
        choice = read_json(state).get("visuals")
    choice = choice or ("placeholder" if dry else doodle.settings(cfg)["images"])
    return choice if choice in doodle.BACKENDS else ("placeholder" if dry else "kaggle")


def _finish_doodle(args, cfg, script, out_dir: Path, audios, backend: str) -> Path:
    """Drawings, frames, timing, render and thumbnail for a doodle explainer."""
    from pipeline import doodle
    from pipeline.assemble import assemble
    from pipeline.captions import build_ass
    from pipeline.motion import plan_to_json
    from pipeline.script import save_script
    from pipeline.thumbnail import make_thumbnail
    save_script(script, out_dir)
    doodle.write_beat_index(out_dir, script)
    status = doodle.generate_images(out_dir, script, cfg, backend)
    if status["missing"]:
        sys.exit(f"{len(status['missing'])} of {status['needed']} drawings are still missing "
                 f"({', '.join(status['missing'][:6])}); run the same command again to retry them")
    frames = doodle.compose_frames(out_dir, script, cfg)
    log(f"doodle: {sum(len(r) for r in frames)} frames ready in {doodle.frames_dir(out_dir)}")
    if args.stop_after in ("media_selection", "visuals"):
        _stop(out_dir, "look through doodle/frames, then rerun with --slug " + out_dir.name)
    plan = doodle.clip_plan(script, audios, frames, cfg)
    write_json(out_dir / "clip_plan.json", plan_to_json(plan))
    build_ass(audios, out_dir / "captions.ass", cfg)
    log(f"assemble: rendering {sum(len(r) for r in plan)} drawings")
    final = assemble(out_dir, plan, audios, cfg)
    (out_dir / "final_1x.mp4").unlink(missing_ok=True)
    log(f"assemble: {final}")
    if args.stop_after == "video":
        _stop(out_dir, f"watch {final}, then rerun with --slug {out_dir.name} for the thumbnail")
    thumb = make_thumbnail(final, script["thumbnail_text"], out_dir / "thumbnail.jpg", cfg)
    log(f"thumbnail: {thumb}")
    print()
    print("REVIEW CHECKLIST")
    print(f"  1. Watch:        {final}")
    print(f"  2. Thumbnail:    {thumb}")
    print(f"  3. Fact-check:   {out_dir / 'script.md'}  (see 'Claims to verify')")
    print(f"  4. Publish:      python run.py publish {out_dir.name}   (add --next-slot to schedule)")
    print(f"  5. Shorts:       python run.py cut {out_dir.name}")
    return out_dir


def cmd_cut(args, cfg):
    """Shorts from a finished long video: the hook and a self-contained chapter or two."""
    from pipeline.cuts import cut_shorts
    out_dir = OUTPUT_DIR / args.slug
    if not (out_dir / "final.mp4").exists():
        sys.exit(f"no final.mp4 in {out_dir}; finish the long video first")
    made = cut_shorts(out_dir, cfg, count=args.count, visuals=args.visuals)
    print()
    print("SHORTS")
    for child in made:
        script = read_json(child / "script.json")
        print(f"  {child.name:<70} {script['title']}")
    print("\nReview each in the UI, then: python run.py queue --publish")


def cmd_lofi(args, cfg):
    """An hour of generated lofi music over one gently animated illustrated scene."""
    from pipeline.lofi import make_lofi
    dry = args.dry_run
    cfg = formats.apply(cfg, "lofi")
    idea = args.idea or ("a student studying at a desk by a rainy window at night" if dry else None)
    if args.slug:
        out_dir = OUTPUT_DIR / args.slug
        if not (out_dir / "script.json").exists():
            sys.exit(f"no script.json in {out_dir}")
    else:
        if not idea:
            sys.exit("give the scene with --idea, e.g. --idea \"a cat asleep on a sunny bookshop counter\"")
        suffix = "-dry-run" if dry else ""
        out_dir = _unique_out_dir(f"{date.today().isoformat()}-lofi-{slugify(idea, 36)}{suffix}")
    # Explicit flags win; otherwise a resumed project keeps the settings it was planned with.
    visuals = args.visuals or ("illustration" if dry else None)
    music = "procedural" if dry else args.music
    minutes = args.minutes if args.minutes else (2 if dry else None)
    script = make_lofi(idea or "", out_dir, cfg, minutes=minutes, music_engine=music, visuals=visuals,
                       use_llm=not dry)
    history.add(out_dir.name, script["title"], script.get("idea", idea or ""), kind="lofi", fmt="lofi")
    print()
    print("REVIEW CHECKLIST")
    print(f"  1. Watch:        {out_dir / 'final.mp4'}  ({script.get('duration', 0) / 60:.0f} min)")
    print(f"  2. Thumbnail:    {out_dir / 'thumbnail.jpg'}")
    print(f"  3. Tracklist:    {out_dir / 'script.md'}")
    print(f"  4. Publish:      python run.py publish {out_dir.name}   (add --next-slot to schedule)")


def _stop(out_dir: Path, hint: str):
    raise _Stopped(out_dir, hint)


def _publish_one(slug: str, cfg: dict, privacy: str | None, publish_at: str | None) -> str:
    from pipeline.publish import upload
    out_dir = OUTPUT_DIR / slug
    video, thumb = out_dir / "final.mp4", out_dir / "thumbnail.jpg"
    if not video.exists():
        raise RuntimeError(f"no final.mp4 in {out_dir}; run make first")
    script = read_json(out_dir / "script.json")
    video_id = upload(video, thumb, script, cfg, privacy=privacy, publish_at=publish_at)
    history.mark_published(slug, video_id, publish_at)
    when = f"goes public at {publish_at}" if publish_at else (privacy or cfg["youtube"]["privacy"])
    log(f"published: https://youtu.be/{video_id}  ({when})")
    return video_id


def _resolve_publish_at(args, cfg) -> str | None:
    from pipeline import scheduler
    if getattr(args, "at", None):
        return scheduler.to_rfc3339(scheduler.parse_local(args.at, cfg))
    if getattr(args, "next_slot", False):
        slot = scheduler.next_slots(cfg, 1)[0]
        log(f"schedule: next free slot is {scheduler.describe(slot, cfg)}")
        return scheduler.to_rfc3339(slot)
    return None


def cmd_publish(args, cfg):
    try:
        _publish_one(args.slug, cfg, args.privacy, _resolve_publish_at(args, cfg))
    except RuntimeError as exc:
        sys.exit(str(exc))


def cmd_queue(args, cfg):
    """Finished videos that are not on YouTube yet; --publish schedules them into the next slots."""
    from pipeline import scheduler
    ready = [e for e in history.unpublished() if (OUTPUT_DIR / e["slug"] / "final.mp4").exists()]
    if not ready:
        print("nothing finished and unpublished; run: python run.py batch --count 3")
        return
    slots = scheduler.next_slots(cfg, len(ready))
    for entry, slot in zip(ready, slots):
        print(f"  {entry['slug']:<60}  -> {scheduler.describe(slot, cfg)}")
    if not args.publish:
        print("\nThat is the plan. Review each video, then add --publish to upload them on this schedule.")
        return
    for entry, slot in zip(ready, slots):
        try:
            _publish_one(entry["slug"], cfg, None, scheduler.to_rfc3339(slot))
        except Exception as exc:
            log(f"queue: FAILED {entry['slug']}: {type(exc).__name__}: {str(exc)[:200]}")


def cmd_list(args, cfg):
    for e in history.load():
        if e.get("youtube_id"):
            status = f"https://youtu.be/{e['youtube_id']}"
            if e.get("publish_at"):
                status += f"  (public at {e['publish_at']})"
        else:
            status = "ready, not published" if (OUTPUT_DIR / e["slug"] / "final.mp4").exists() else "unfinished"
        print(f"{e.get('created', '')}  {e.get('slug', ''):<60}  {status}")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    sub.add_parser("ideas").set_defaults(fn=cmd_ideas)
    sub.add_parser("voices", help="list narrators per engine and when each was last used").set_defaults(fn=cmd_voices)

    m = sub.add_parser("make")
    m.add_argument("--idea", "--topic", dest="idea",
                   help="rough short-story premise or fact topic")
    m.add_argument("--slug", help="resume an existing output/<slug>")
    m.add_argument("--stop-after", choices=STAGES[1:], help="pause after this stage")
    m.add_argument("--dry-run", action="store_true",
                   help="no API keys: sample story, built-in voice, placeholders")
    m.add_argument("--voice", choices=["kokoro", "gemini", "elevenlabs", "local"],
                   help="override config voice engine")
    m.add_argument("--voice-name", "--narrator", dest="voice_name",
                   help="narrator for this video (see: run.py voices); default rotates through the pool")
    m.add_argument("--format", choices=["auto", "story", "fact"], default="auto",
                   help="choose an original story, a fact explainer, or infer from your idea")
    m.add_argument("--length", choices=["short", "long", "doodle"], default="short",
                   help="short = vertical Short under a minute; long = 16:9 ten-minute video with chapters; "
                        "doodle = 9-11 minute doodle explainer with a drawing per sentence")
    m.add_argument("--visuals", choices=["pixabay", "diffusion", "illustration", "placeholder", "kaggle"],
                   help="override config visuals source (doodle videos: kaggle, diffusion or placeholder)")
    m.add_argument("--force", action="store_true", help="wipe an existing output folder with the same slug")
    m.set_defaults(fn=cmd_make)

    b = sub.add_parser("batch", help="render several videos with fresh topics and rotating narrators")
    b.add_argument("--count", type=int, default=3, help="how many videos to make (default 3)")
    b.add_argument("--format", choices=["auto", "story", "fact"], default="auto",
                   help="auto alternates story and fact")
    b.add_argument("--length", choices=["short", "long", "doodle"], default="short")
    b.add_argument("--voice", choices=["kokoro", "gemini", "elevenlabs", "local"])
    b.add_argument("--visuals", choices=["pixabay", "diffusion", "illustration", "placeholder", "kaggle"])
    b.add_argument("--stop-after", choices=STAGES[1:], help="pause every video after this stage")
    b.add_argument("--dry-run", action="store_true", help="no API keys: the sample story, repeated")
    b.set_defaults(fn=cmd_batch)

    c = sub.add_parser("cut", help="Shorts cut from a finished long video")
    c.add_argument("slug")
    c.add_argument("--count", type=int, help="how many excerpts (default from config: cuts.count)")
    c.add_argument("--visuals", choices=["pixabay", "diffusion", "illustration", "placeholder"],
                   help="which candidates to reuse; default: what the long video used")
    c.set_defaults(fn=cmd_cut)

    lo = sub.add_parser("lofi", help="an hour of generated lofi music over an animated scene")
    lo.add_argument("--idea", help="the scene, e.g. \"a student at a desk by a rainy window at night\"")
    lo.add_argument("--slug", help="resume an existing output/<slug>")
    lo.add_argument("--minutes", type=float, help="video length (default from config: lofi.minutes)")
    lo.add_argument("--music", choices=["procedural", "acestep"], help="music engine (default: music.engine)")
    lo.add_argument("--visuals", choices=["diffusion", "illustration", "placeholder"],
                    help="how the scene is drawn (default: diffusion, your GPU)")
    lo.add_argument("--dry-run", action="store_true", help="no keys or GPU: drawn scene, 2 minutes, procedural music")
    lo.set_defaults(fn=cmd_lofi)

    pub = sub.add_parser("publish")
    pub.add_argument("slug")
    pub.add_argument("--privacy", choices=["private", "unlisted", "public"])
    pub.add_argument("--at", help="go public at this local time, e.g. '2026-11-06 18:00' (uploads private until then)")
    pub.add_argument("--next-slot", action="store_true", help="go public at the next free slot from youtube.schedule")
    pub.set_defaults(fn=cmd_publish)

    q = sub.add_parser("queue", help="list finished, unpublished videos and the slots they would get")
    q.add_argument("--publish", action="store_true", help="upload them now, scheduled into those slots")
    q.set_defaults(fn=cmd_queue)

    sub.add_parser("list").set_defaults(fn=cmd_list)

    args = p.parse_args()
    args.fn(args, load_config())


if __name__ == "__main__":
    main()
