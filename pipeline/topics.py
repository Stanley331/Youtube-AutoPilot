"""Stage 1: pick a topic with the configured LLM; skip anything already made."""
import re

from . import history
from .llm import generate_json

IDEAS_SCHEMA = {
    "type": "object",
    "properties": {
        "ideas": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "title": {"type": "string"},
                    "angle": {"type": "string"},
                    "why_it_will_perform": {"type": "string"},
                    "kind": {"type": "string", "enum": ["story", "fact"]},
                    "search_demand": {"type": "string", "enum": ["high", "medium", "low"]},
                },
                "required": ["title", "angle", "why_it_will_perform", "kind", "search_demand"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["ideas"],
    "additionalProperties": False,
}


def generate_ideas(cfg: dict, count: int = 10, length: str = "short") -> list[dict]:
    ch = cfg["channel"]
    past = history.past_titles()
    past_text = "\n".join(f"- {t}" for t in past) or "- (none yet)"
    if length in ("long", "doodle"):
        system = (
            "You are a YouTube idea editor for ten-minute videos: original stories told in chapters and "
            "well-researched explainers that answer one big question in depth. Suggest a balanced mix. "
            "A long video needs a subject with enough substance for six chapters, a clear question or "
            "stakes in the title, and a payoff worth staying for. Use curiosity without clickbait; do not "
            "invent claims or copy existing fictional characters. Titles must be under 60 characters."
        )
    else:
        system = (
            "You are a YouTube Shorts idea editor for short original stories and interesting facts. "
            "Suggest a balanced mix of compact, emotionally satisfying fictional story premises and "
            "surprising, stable facts that can be explained accurately in under a minute. "
            "Use curiosity without clickbait; do not invent claims or copy existing fictional characters. "
            "Titles must be under 60 characters."
        )
    prompt = f"""Channel: {ch['name']}
Niche: {ch['niche']}
Audience: {ch['audience']}
Tone: {ch['tone']}

Videos already published (do not repeat these topics or close variants):
{past_text}

Propose {count} {'long-form video' if length in ('long', 'doodle') else 'short'} ideas, ordered best first, with a roughly even mix of story and fact ideas.
Facts should be evergreen and easy for a human to verify; stories should be original and have a strong emotional turn.
Each idea needs a title, kind (story or fact), the specific angle, one sentence on why it will perform, and search demand."""
    return generate_json(system, prompt, IDEAS_SCHEMA, effort="medium")["ideas"]


_STOPWORDS = {"the", "a", "an", "of", "and", "to", "in", "on", "that", "who", "how", "why", "what",
              "is", "was", "for", "with", "his", "her", "its", "their", "from", "at", "by"}


def _keywords(title: str) -> set[str]:
    text = re.sub(r"['’]s\b", "", title.lower()).replace("'", "")
    return {w for w in re.findall(r"[a-z0-9]+", text) if w not in _STOPWORDS and len(w) > 2}


def near_duplicate(title: str, others: list[str], threshold: float = 0.5) -> bool:
    """True when a title shares most of its meaningful words with an earlier one."""
    words = _keywords(title)
    if not words:
        return False
    for other in others:
        theirs = _keywords(other)
        if not theirs:
            continue
        overlap = len(words & theirs) / min(len(words), len(theirs))
        if overlap >= threshold or title.lower() == other.lower():
            return True
    return False


def pick_topic(cfg: dict, kind: str | None = None, length: str = "short") -> dict:
    return pick_topics(cfg, 1, kind=kind, length=length)[0]


def pick_topics(cfg: dict, count: int, kind: str | None = None, length: str = "short") -> list[dict]:
    """Fresh, distinct ideas for a batch; story and fact alternate when no kind is forced.

    Ideas that repeat a past title or each other are dropped. The channel's last
    few videos decide which kind goes first, so a batch continues the rhythm.
    """
    ideas = generate_ideas(cfg, count=max(10, min(count * 3, 24)), length=length)
    past = history.past_titles()
    fresh, chosen_titles = [], []
    for idea in ideas:
        if near_duplicate(idea["title"], past + chosen_titles):
            continue
        fresh.append(idea)
        chosen_titles.append(idea["title"])
    if not fresh:
        fresh = ideas[:1]
    if kind:
        matching = [i for i in fresh if i["kind"] == kind]
        return (matching or fresh)[:count]

    recent = [k for k in history.recent_kinds(1) if k in ("story", "fact")]
    want = "fact" if recent and recent[-1] == "story" else "story"
    picked: list[dict] = []
    while len(picked) < count and fresh:
        pool = [i for i in fresh if i["kind"] == want] or fresh
        picked.append(pool[0])
        fresh.remove(pool[0])
        want = "fact" if want == "story" else "story"
    return picked
