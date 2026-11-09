"""Long-form scripts: a chaptered ten-minute video written in several passes.

One request cannot reliably return 1,500 words of narration with a storyboard
for forty scenes, so the script is built the way a writer would: an outline
first (title, hook, chapters with their beats, closing), then one request per
chapter that turns its beats into narrated scenes while seeing where the
previous chapter left off. The result has the same shape as a Short's script
(flat ``scenes`` list) plus ``chapters`` with the first scene of each, which
become YouTube chapter timestamps once the narration is timed.
"""
from __future__ import annotations

import time

from .config import env
from .llm import generate_json
from .script import SCENE_SCHEMA, _as_str, _as_str_list, normalize_scenes
from .util import log

# A small free model writes fine Shorts but thin ten-minute explainers, so long-form writing goes
# to Gemini whenever a GEMINI_API_KEY is set; LONG_SCRIPT_PROVIDER in .env (gemini, anthropic,
# nemotron) overrides that choice.
THIN_CHAPTER_RATIO = 0.7
# A chapter with high thinking can take minutes; a timeout throws the work away and restarts on
# the fallback model, so long scripts wait longer than the 180 s default.
LONG_TIMEOUT = 360

OUTLINE_SCHEMA = {
    "type": "object",
    "properties": {
        "content_type": {"type": "string", "enum": ["story", "fact"]},
        "title": {"type": "string"},
        "alternative_titles": {"type": "array", "items": {"type": "string"}},
        "thumbnail_text": {"type": "string"},
        "description": {"type": "string"},
        "tags": {"type": "array", "items": {"type": "string"}},
        "main_character": {"type": "string"},
        "hook": {"type": "string"},
        "chapters": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "title": {"type": "string"},
                    "summary": {"type": "string"},
                    "beats": {"type": "array", "items": {"type": "string"}},
                },
                "required": ["title", "summary", "beats"],
                "additionalProperties": False,
            },
        },
        "closing": {"type": "string"},
        "claims_to_verify": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["content_type", "title", "alternative_titles", "thumbnail_text", "description",
                 "tags", "main_character", "hook", "chapters", "closing", "claims_to_verify"],
    "additionalProperties": False,
}

CHAPTER_SCHEMA = {
    "type": "object",
    "properties": {
        "scenes": {"type": "array", "items": SCENE_SCHEMA},
        "claims_to_verify": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["scenes", "claims_to_verify"],
    "additionalProperties": False,
}

STORYBOARD_RULES = """Rules for the visual storyboard (every scene):
- visual_description: one specific, visually distinct moment that real stock footage OR an illustration could show. Concrete people, objects, action, setting. No abstract concepts.
- visual_query: 2 to 4 concrete nouns from that moment, usable as a stock-footage search (e.g. "old sailing ship", "rainy street", "wheat field").
- setting, visual_subject, action: short concrete phrases that agree with the narration. Describe people by role ("older farmer", "young clerk") as well as any name.
- visual_objects: 1 to 3 visible props or symbols, or an empty array.
- on_screen_text: at most 5 words in ALL CAPS, or an empty string. Use it for a date, a number, or a name worth seeing."""


def long_script_provider() -> str | None:
    """Who writes ten-minute scripts: LONG_SCRIPT_PROVIDER, else Gemini when a key exists, else LLM_PROVIDER."""
    explicit = (env("LONG_SCRIPT_PROVIDER") or "").lower()
    if explicit:
        return explicit
    return "gemini" if env("GEMINI_API_KEY") else None


DOODLE_OUTLINE_RULE = """
- This is a hand-drawn doodle explainer: simple cartoon drawings on a white background, a new drawing every few seconds. main_character is required for every video, story or fact: one recurring doodle character who acts out the ideas, described in one phrase by look only (for example "a round-headed caveman with messy brown hair and a fur tunic"). Fit it to the topic."""

DOODLE_RULES = """Rules for the doodle storyboard (every scene is a list of beats):
- A beat is one short sentence or clause of the narration, about 5 to 14 words, read in order. The scene's narration is its beats joined with spaces, so put every word of the narration into exactly one beat and split long sentences at a comma.
- picture: the simple drawing on screen while that beat is spoken, in at most 25 words: who or what, doing what, where. One clear subject, drawable as a flat cartoon. Turn abstract ideas into a concrete visual metaphor (a crowd of tiny skeletons for a death toll, a ticking clock for running out of time). Never ask for words, letters or numbers inside the drawing.
- character: true when the recurring main character is in the picture. Use the character often; it is what makes the channel recognisable. The character lives in the pictures only: the narration never mentions "the main character" or "our character"; it speaks to the viewer ("you") or about the people of the topic.
- label: on roughly one beat in four, a handwritten note of at most 3 words naming something in the picture (e.g. "COOLS DOWN", "17 HOURS A WEEK"); otherwise an empty string.
- card: about once per chapter (its first beat, or a big turn), a full-screen handwritten title card of at most 5 words in capitals that sums up the moment (e.g. "THE WORKLOAD EXPLODES"); otherwise an empty string. A card beat still has its narration text."""

BEAT_SCHEMA = {
    "type": "object",
    "properties": {
        "text": {"type": "string"},
        "picture": {"type": "string"},
        "character": {"type": "boolean"},
        "label": {"type": "string"},
        "card": {"type": "string"},
    },
    "required": ["text", "picture", "character", "label", "card"],
    "additionalProperties": False,
}

DOODLE_CHAPTER_SCHEMA = {
    "type": "object",
    "properties": {
        "scenes": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {"beats": {"type": "array", "items": BEAT_SCHEMA}},
                "required": ["beats"],
                "additionalProperties": False,
            },
        },
        "claims_to_verify": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["scenes", "claims_to_verify"],
    "additionalProperties": False,
}


def settings(cfg: dict) -> dict:
    vid = cfg["video"]
    chapters = max(3, int(vid.get("chapters", 6) or 6))
    per_chapter = max(3, int(vid.get("scenes_per_chapter", 6) or 6))
    target_words = int(float(vid.get("target_minutes", 10)) * int(vid.get("words_per_minute", 140)))
    return {"chapters": chapters, "scenes_per_chapter": per_chapter, "target_words": target_words,
            "words_per_chapter": max(120, target_words // chapters),
            "words_per_scene": max(30, target_words // (chapters * per_chapter)),
            "provider": long_script_provider()}


def write_long_script(cfg: dict, topic: dict, doodle: bool = False) -> dict:
    """Outline, then one pass per chapter; returns a script dict with ``chapters``.

    ``doodle`` writes every scene as a list of one-sentence beats, each with its own
    picture, for the doodle explainer format (a new drawing every few seconds).
    """
    s = settings(cfg)
    ch = cfg["channel"]
    requested = topic.get("format", "auto")
    if requested == "story":
        kind_rule = ("Write an original fictional story in chapters: a character with a goal, rising "
                     "obstacles, a turn, and a resolved ending. Do not copy existing characters or plots.")
    elif requested == "fact":
        kind_rule = ("Write a well-researched explainer. Prefer stable, well-established facts, never "
                     "invent names, dates or numbers, and list every claim a person should verify.")
    else:
        kind_rule = "Choose story for a premise or character situation; choose fact for a question, subject, or event."

    system = f"""You plan long-form YouTube videos (about {s['target_words'] // 140} minutes) for this channel.
Channel: {ch['name']}. Niche: {ch['niche']}. Audience: {ch['audience']}. Tone: {ch['tone']}.
Requested format: {requested}. {kind_rule}

What a ten-minute video needs that a Short does not:
- A hook (first 20 seconds) that states the stakes or the question and promises a specific payoff, with no greeting and no "in this video".
- Exactly {s['chapters']} chapters that each advance the story or answer one sub-question, each ending on a small payoff or an open question that pulls into the next chapter.
- A closing that pays off the hook, then one short line inviting the viewer to the next video. No summary of what was just said.
- The title (under 60 characters) and thumbnail_text (at most 4 words, capitals) must be specific and honest; curiosity is fine, bait is not.
- For facts: name what a person must check before publishing in claims_to_verify.{DOODLE_OUTLINE_RULE if doodle else ''}"""
    prompt = f"""Plan a long-form video from this rough idea.

User's idea or topic:
{topic['title']}
Angle: {topic.get('angle', '')}
Why it will perform: {topic.get('why_it_will_perform', '')}

Return: content_type, title, 3 alternative_titles, thumbnail_text, a YouTube description (2 short paragraphs, no timestamps), 8 to 15 tags, main_character (the recurring protagonist's look in one phrase, or an empty string), the hook narration (about 60 words), {s['chapters']} chapters each with a title (3 to 6 words, usable as a YouTube chapter name), a one-sentence summary and 4 to 6 beats (what happens or what is explained, in order), the closing narration (about 50 words), and claims_to_verify."""
    log("script: writing the outline" + (f" with {s['provider']}" if s["provider"] else ""))
    outline = generate_json(system, prompt, OUTLINE_SCHEMA, effort="high", provider=s["provider"], timeout=LONG_TIMEOUT)
    chapters_raw = [c for c in (outline.get("chapters") or []) if isinstance(c, dict) and _as_str(c.get("title"))]
    if len(chapters_raw) < 2:
        raise RuntimeError("The model returned an outline with fewer than two chapters")
    content_type = _as_str(outline.get("content_type")).lower()
    if content_type not in ("story", "fact"):
        content_type = requested if requested in ("story", "fact") else "fact"
    title = _as_str(outline.get("title")) or _as_str(topic.get("title")) or "Untitled video"
    main_character = _as_str(outline.get("main_character"))
    hook, closing = _as_str(outline.get("hook")), _as_str(outline.get("closing"))
    claims = _as_str_list(outline.get("claims_to_verify"))

    scene_system = f"""You write narration and a visual storyboard for one chapter of a long-form YouTube video.
Channel tone: {ch['tone']}. The narration is read aloud by a text-to-speech voice.
Format: {content_type}. {kind_rule}

Rules for the narration:
- Exactly {s['scenes_per_chapter']} scenes of about {s['words_per_scene']} words each (never under 30), about {s['words_per_chapter']} words in all. Each scene is one beat a viewer can picture.
- Be concrete: names, places, dates, quantities, what things looked like, what people did and why. Vague generalities ("this was important for trade") are the one thing a long video cannot survive. If you are not sure of a detail, say what is known and list it in claims_to_verify rather than leaving the scene abstract.
- Short natural sentences. Explain any niche term in passing. Numbers written the way they are spoken.
- No markdown, no bullets, no stage directions, no chapter titles inside the narration, no "in this chapter".
- Continue from where the previous chapter stopped; do not repeat it. End the chapter on its payoff or its open question.
- Facts: do not invent names, dates, quotes or statistics; put every specific claim into claims_to_verify.
- Stories: keep the cast, names and setting consistent with the main character description.

{DOODLE_RULES if doodle else STORYBOARD_RULES}"""
    chapter_schema = DOODLE_CHAPTER_SCHEMA if doodle else CHAPTER_SCHEMA

    scenes: list[dict] = []
    chapters: list[dict] = []
    for n, chapter in enumerate(chapters_raw, 1):
        beats = _as_str_list(chapter.get("beats"))
        previous = " ".join(sc["narration"] for sc in scenes[-2:]) if scenes else "(this is the first chapter)"
        extra = ""
        if n == 1:
            extra = f"\nThis is the opening chapter. Its first scene must be this hook, lightly edited if needed:\n{hook}"
        if n == len(chapters_raw):
            extra += f"\nThis is the final chapter. Its last scene must end with this closing, lightly edited if needed:\n{closing}"
        prompt = f"""Video title: {title}
Main character: {main_character or '(none; factual video)'}
All chapters, in order: {'; '.join(_as_str(c.get('title')) for c in chapters_raw)}

Write chapter {n} of {len(chapters_raw)}: "{_as_str(chapter.get('title'))}"
Summary: {_as_str(chapter.get('summary'))}
Beats, in order:
{chr(10).join('- ' + b for b in beats) or '- (use the summary)'}

The previous chapter ended with: {previous}{extra}"""
        log(f"script: chapter {n}/{len(chapters_raw)}: {_as_str(chapter.get('title'))}")
        began = time.time()
        result = generate_json(scene_system, prompt, chapter_schema, effort="high", provider=s["provider"], timeout=LONG_TIMEOUT)
        new_scenes = _scenes_of(result, doodle)
        if not new_scenes:
            raise RuntimeError(f"Chapter {n} came back without narrated scenes")
        words = sum(len(sc["narration"].split()) for sc in new_scenes)
        log(f"script: chapter {n} written in {time.time() - began:.0f}s ({len(new_scenes)} scenes, {words} words)")
        if words < THIN_CHAPTER_RATIO * s["words_per_chapter"]:
            # Small models answer briefly; one more pass with the draft and the shortfall spelled out.
            log(f"script: chapter {n} is thin ({words} words, wanted {s['words_per_chapter']}); expanding")
            redo = prompt + (f"\n\nYour previous draft had only {words} words in total; it must have about "
                             f"{s['words_per_chapter']}. Keep its scene order and storyboard, and expand every "
                             f"scene to about {s['words_per_scene']} words with concrete detail: who, where, "
                             "when, how much, what it looked like, what happened next.\nPrevious draft:\n"
                             + "\n".join(f"- {sc['narration']}" for sc in new_scenes))
            try:
                again = _scenes_of(generate_json(scene_system, redo, chapter_schema, effort="high",
                                                 provider=s["provider"], timeout=LONG_TIMEOUT), doodle)
                if sum(len(sc["narration"].split()) for sc in again) > words and again:
                    new_scenes = again
            except Exception as exc:
                log(f"script: expansion failed ({type(exc).__name__}); keeping the short chapter")
        chapters.append({"title": _as_str(chapter.get("title")), "first_scene": len(scenes)})
        scenes.extend(new_scenes)
        for claim in _as_str_list(result.get("claims_to_verify")):
            if claim not in claims:
                claims.append(claim)

    thumbnail_text = _as_str(outline.get("thumbnail_text")) or " ".join(title.split()[:4]).upper()
    words = sum(len(sc["narration"].split()) for sc in scenes)
    log(f"script: {len(chapters)} chapters, {len(scenes)} scenes, {words} words "
        f"(target {s['target_words']})")
    return {
        "format": "doodle" if doodle else "long",
        "content_type": content_type,
        "title": title,
        "alternative_titles": _as_str_list(outline.get("alternative_titles")),
        "thumbnail_text": thumbnail_text,
        "description": _as_str(outline.get("description")) or title,
        "tags": _as_str_list(outline.get("tags")),
        "main_character": main_character,
        "chapters": chapters,
        "scenes": scenes,
        "claims_to_verify": claims,
    }


def _scenes_of(result: dict, doodle: bool) -> list[dict]:
    if not doodle:
        return normalize_scenes(result.get("scenes"))
    from .doodle import normalize_doodle_scenes
    return normalize_doodle_scenes(result.get("scenes"))


def chapter_times(script: dict, durations: list[float]) -> list[dict]:
    """[{title, seconds}] for each chapter from the timed narration (first chapter at 0)."""
    starts = [0.0]
    for d in durations:
        starts.append(starts[-1] + float(d))
    times = []
    for chapter in script.get("chapters", []) or []:
        first = int(chapter.get("first_scene", 0))
        if 0 <= first < len(durations):
            times.append({"title": str(chapter.get("title", "")), "seconds": round(starts[first], 2)})
    if times:
        times[0]["seconds"] = 0.0
    return times


def format_timestamp(seconds: float) -> str:
    seconds = int(max(0.0, float(seconds)))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m}:{s:02d}"


def chapter_lines(script: dict) -> list[str]:
    """YouTube chapter lines ("0:00 Title"); empty unless there are at least three."""
    times = script.get("chapter_times") or []
    if len(times) < 3:
        return []
    return [f"{format_timestamp(t.get('seconds', 0))} {t.get('title', '')}".rstrip() for t in times]
