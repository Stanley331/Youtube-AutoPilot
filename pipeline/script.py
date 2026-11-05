"""Stage 2: expand a rough idea into structured scenes and video metadata."""
from pathlib import Path

from .llm import generate_json
from .util import write_json

SCENE_SCHEMA = {
    "type": "object",
    "properties": {
        "narration": {"type": "string"},
        "visual_query": {"type": "string"},
        "visual_description": {"type": "string"},
        "setting": {"type": "string"},
        "visual_subject": {"type": "string"},
        "action": {"type": "string"},
        "visual_objects": {"type": "array", "items": {"type": "string"}},
        "on_screen_text": {"type": "string"},
    },
    "required": ["narration", "visual_query", "visual_description", "setting",
                 "visual_subject", "action", "visual_objects", "on_screen_text"],
    "additionalProperties": False,
}

SCRIPT_SCHEMA = {
    "type": "object",
    "properties": {
        "content_type": {"type": "string", "enum": ["story", "fact"]},
        "title": {"type": "string"},
        "alternative_titles": {"type": "array", "items": {"type": "string"}},
        "thumbnail_text": {"type": "string"},
        "description": {"type": "string"},
        "tags": {"type": "array", "items": {"type": "string"}},
        "main_character": {"type": "string"},
        "scenes": {"type": "array", "items": SCENE_SCHEMA},
        "claims_to_verify": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["content_type", "title", "alternative_titles", "thumbnail_text", "description",
                 "tags", "main_character", "scenes", "claims_to_verify"],
    "additionalProperties": False,
}


def write_script(cfg: dict, topic: dict) -> dict:
    ch, vid = cfg["channel"], cfg["video"]
    target_words = int(vid["target_minutes"] * vid["words_per_minute"])
    requested_format = topic.get("format", "auto")
    if requested_format == "story":
        format_rule = "Write an original fictional short story with a clear character, goal, obstacle, turn, and resolved ending. Do not copy existing characters or stories."
    elif requested_format == "fact":
        format_rule = "Write a concise factual explainer about one interesting fact. Prefer stable, well-established facts; do not invent details, and list every claim a person should verify."
    else:
        format_rule = "Choose story for a premise or character situation; choose fact for a question, subject, or claim. If unclear, make the most useful choice and state it in content_type."
    system = f"""You write original YouTube Shorts for a channel about short stories and interesting facts.
Channel: {ch['name']}. Niche: {ch['niche']}. Audience: {ch['audience']}. Tone: {ch['tone']}.

The user's input is a rough idea, premise, question, or fact topic, not finished narration. Preserve its central idea and expand it into one complete Short.
Requested format: {requested_format}. {format_rule}

Rules for the narration (it will be read aloud by a text-to-speech voice):
- Open with a strong hook in the first sentence. Skip greetings and setup. Keep one clear story or idea, deliver a satisfying payoff, and end with a natural short follow prompt.
- Target about {target_words} words, roughly 45 to 55 seconds at the configured narration speed. Keep the pace brisk and every sentence useful.
- Use short, natural sentences. Explain any niche term briefly.
- Write numbers the way they should be spoken aloud when needed.
- No markdown, no bullet symbols, no stage directions inside narration.
- For factual scripts, do not invent facts, quotes, dates, or statistics. Add specific factual claims to claims_to_verify. Avoid current claims that may have changed.
- For story scripts, keep the cast and setting original. Do not claim fictional events are real.
- Split the narration into exactly {vid['max_scenes']} short scenes (no more, no fewer), generally 15 to 35 words each.

Rules for the visual storyboard:
- main_character: describe the recurring original protagonist's appearance in one short phrase, or an empty string for a fact video without a recurring person. Keep that design consistent across scenes.
- visual_description: describe one specific, visually distinct scene that can be depicted with real stock footage OR a simple illustration. Include concrete people, objects, action, and setting. Avoid abstract concepts.
- visual_query: 2 to 4 key concrete nouns from that scene; used as a search label for stock footage (Pixabay) or as an illustration prompt. Choose terms likely to return good results (e.g., "soldier trench", "old book", "rainy street").
- setting, visual_subject, action: concise concrete descriptions that agree with the narration. Describe people with a human role (for example, "older woman", "young inventor", "soldier") as well as any name.
- visual_objects: 1 to 3 specific visible props or symbols that help depict this scene; use an empty array only if none fit.
- on_screen_text: at most 5 short words in ALL CAPS summarizing the scene, or an empty string.

Also produce: a clear title under 60 characters, 3 alternative titles, thumbnail text of at most 4 words in capital letters, a concise YouTube description, 5 to 10 relevant tags, and every specific factual claim that a human should verify before publishing. Stories may have an empty claims_to_verify list."""
    prompt = f"""Turn this rough idea into a complete vertical short video script with a matching visual storyboard.
The idea is not finished narration. Make the result engaging, coherent, and short enough for the configured runtime.

User's idea or topic:
{topic['title']}
Angle: {topic.get('angle', '')}
Why it will perform: {topic.get('why_it_will_perform', '')}"""
    result = generate_json(system, prompt, SCRIPT_SCHEMA, effort="high")
    return normalize_script(result, topic, vid["max_scenes"], requested_format)


def _as_str(value) -> str:
    if value is None:
        return ""
    if isinstance(value, (list, tuple)):
        return " ".join(_as_str(v) for v in value)
    return str(value).strip()


def _as_str_list(value) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value.strip()] if value.strip() else []
    if isinstance(value, (list, tuple)):
        return [_as_str(v) for v in value if _as_str(v)]
    return [str(value)]


def normalize_scenes(raw_scenes) -> list[dict]:
    """Scene dicts with every field downstream stages read, from whatever the model sent."""
    if isinstance(raw_scenes, dict):
        raw_scenes = list(raw_scenes.values())
    scenes = []
    for raw in raw_scenes or []:
        if isinstance(raw, str):
            raw = {"narration": raw}
        if not isinstance(raw, dict):
            continue
        narration = _as_str(raw.get("narration"))
        if not narration:
            continue
        query = _as_str(raw.get("visual_query")) or " ".join(narration.split()[:4])
        scenes.append({
            "narration": narration,
            "visual_query": query,
            "visual_description": _as_str(raw.get("visual_description")) or query,
            "setting": _as_str(raw.get("setting")),
            "visual_subject": _as_str(raw.get("visual_subject")),
            "action": _as_str(raw.get("action")),
            "visual_objects": _as_str_list(raw.get("visual_objects")),
            "on_screen_text": _as_str(raw.get("on_screen_text")),
        })
    return scenes


def normalize_script(result: dict, topic: dict, max_scenes: int,
                     requested_format: str = "auto") -> dict:
    """Fill in every field downstream stages rely on, whatever the model returned.

    Providers that are prompted for JSON (Nemotron) sometimes omit optional-looking
    keys; this keeps a missing field from crashing voice, captions, or publishing.
    """
    if not isinstance(result, dict):
        raise RuntimeError("The model did not return a script object")

    title = _as_str(result.get("title")) or _as_str(topic.get("title")) or "Untitled short"
    content_type = _as_str(result.get("content_type")).lower()
    if content_type not in ("story", "fact"):
        content_type = requested_format if requested_format in ("story", "fact") else "story"

    scenes = normalize_scenes(result.get("scenes"))
    if not scenes:
        raise RuntimeError("The model returned a script with no narrated scenes")
    # Enforce max_scenes: trim if the LLM returned too many
    if max_scenes and len(scenes) > max_scenes:
        scenes = scenes[:max_scenes]

    thumbnail_text = _as_str(result.get("thumbnail_text"))
    if not thumbnail_text:
        thumbnail_text = " ".join(title.split()[:4]).upper()

    return {
        "content_type": content_type,
        "title": title,
        "alternative_titles": _as_str_list(result.get("alternative_titles")),
        "thumbnail_text": thumbnail_text,
        "description": _as_str(result.get("description")) or title,
        "tags": _as_str_list(result.get("tags")),
        "main_character": _as_str(result.get("main_character")),
        "scenes": scenes,
        "claims_to_verify": _as_str_list(result.get("claims_to_verify")),
    }


def save_script(script: dict, out_dir: Path) -> None:
    write_json(out_dir / "script.json", script)
    voice = script.get("voice") or {}
    kind = script.get("format", "short")
    lines = [f"# {script['title']}", "",
             f"**Format:** {kind} {script.get('content_type', '')}".rstrip(), "",
             f"**Narrator:** {voice.get('name') or 'default'} ({voice.get('engine', 'not chosen yet')})", "",
             f"**Thumbnail text:** {script.get('thumbnail_text', '')}", "",
             "**Alternative titles:**"]
    lines += [f"- {t}" for t in script.get("alternative_titles", [])]
    chapters = {int(c.get("first_scene", -1)): c for c in script.get("chapters", []) if isinstance(c, dict)}
    if script.get("tracks"):
        lines += ["", "## Tracklist", ""]
        lines += [f"- {t.get('title', '')}  ({float(t.get('seconds', 0)) / 60:.1f} min, "
                  f"{t.get('bpm', '')} bpm, {t.get('key', '')})" for t in script["tracks"]]
    lines += ["", "## Script", ""]
    for i, s in enumerate(script.get("scenes", []), 1):
        if (i - 1) in chapters:
            lines += [f"## Chapter: {chapters[i - 1].get('title', '')}", ""]
        lines += [f"### Scene {i}  `{s.get('visual_query', '')}`  ({s.get('on_screen_text') or 'no text'})",
                  "", s.get("narration", ""), "",
                  f"**Illustration:** {s.get('visual_description', s.get('visual_query', ''))}", ""]
    lines += ["## Claims to verify before publishing", ""]
    lines += [f"- [ ] {c}" for c in script.get("claims_to_verify", [])]
    lines += ["", "## Description", "", script.get("description", ""), "",
              "## Tags", "", ", ".join(script.get("tags", []))]
    (out_dir / "script.md").write_text("\n".join(lines), encoding="utf-8")
