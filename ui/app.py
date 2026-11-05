"""YouTube Autopilot studio: one project at a time, one step at a time.

    streamlit run ui/app.py

Three screens:
  Library    every project as a card with where it stands
  New video  pick a kind (Short, Long video, Lofi music), then the one form it needs
  Project    a wizard: the stepper on top, finished steps folded to one line,
             the current step open, later steps greyed out. Every action shows
             its error in place with the log one click away.

Nothing here touches disk directly; ``ui/helpers.py`` does, so a step's state
is always what is in output/<slug>/ and the page can be reloaded at any time.
"""
from __future__ import annotations

import html
import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent))
import helpers as h  # noqa: E402

st.set_page_config(page_title="YouTube Autopilot", page_icon="🎬", layout="wide",
                   initial_sidebar_state="expanded")

ACCENT = "#ff4b4b"
st.markdown(f"""
<style>
  .block-container {{ max-width: 1240px; padding-top: 1.6rem; }}
  h1, h2, h3 {{ letter-spacing: -0.01em; }}
  .kind-chip {{ display:inline-block; padding:2px 10px; border-radius:999px; font-size:0.75rem;
                font-weight:600; background:rgba(128,128,128,0.18); margin-right:6px; }}
  .muted {{ color: rgba(128,128,128,0.95); font-size:0.9rem; }}
  .mono {{ font-family: ui-monospace, Menlo, Consolas, monospace; font-size:0.8rem; color:rgba(128,128,128,0.9); }}
  .stepper {{ display:flex; align-items:flex-start; margin: 0.6rem 0 1.4rem 0; }}
  .step {{ flex:1; text-align:center; position:relative; }}
  .step .dot {{ width:34px; height:34px; border-radius:50%; margin:0 auto 6px auto; display:flex;
                align-items:center; justify-content:center; font-weight:700; font-size:0.95rem;
                border:2px solid rgba(128,128,128,0.45); color:rgba(128,128,128,0.9); background:transparent; }}
  .step.done .dot {{ background:#2e7d32; border-color:#2e7d32; color:white; }}
  .step.current .dot {{ background:{ACCENT}; border-color:{ACCENT}; color:white;
                        box-shadow:0 0 0 5px rgba(255,75,75,0.18); }}
  .step .label {{ font-size:0.85rem; color:rgba(128,128,128,0.9); }}
  .step.current .label {{ color:inherit; font-weight:700; }}
  .step.done .label {{ color:inherit; }}
  .step:not(:last-child)::after {{ content:""; position:absolute; top:17px; left:calc(50% + 20px);
                                   width:calc(100% - 40px); height:2px; background:rgba(128,128,128,0.35); }}
  .step.done:not(:last-child)::after {{ background:#2e7d32; }}
  .done-row {{ padding:0.55rem 0.9rem; border-left:4px solid #2e7d32; margin:0.35rem 0;
               background:rgba(46,125,50,0.08); border-radius:6px; }}
  .todo-row {{ padding:0.55rem 0.9rem; border-left:4px solid rgba(128,128,128,0.3); margin:0.35rem 0;
               color:rgba(128,128,128,0.75); border-radius:6px; }}
  .card-title {{ font-weight:700; font-size:1.02rem; margin:0.2rem 0 0.1rem 0; }}
  .ok {{ color:#2e7d32; font-weight:600; }}
  .warn {{ color:#b26a00; font-weight:600; }}
</style>
""", unsafe_allow_html=True)

PICTURE_LABELS = {
    "pixabay": "Pixabay animation: a ready-made lofi loop (instant)",
    "diffusion": "AI panel drawn on your GPU (about a minute)",
    "illustration": "Drawn panel (instant)",
    "placeholder": "Plain slide (test)",
}
MUSIC_LABELS = {
    "pixabay": "Pixabay library: tracks you downloaded (instant)",
    "acestep": "ACE-Step 1.5: AI music, about 9 minutes per track",
    "procedural": "Composer: synthesized here, about 1 minute per track",
}
KIND_BLURB = {
    "short": ("Vertical, under a minute", "A script of about six scenes, a narrator, stock footage or AI "
                                           "panels per scene, big captions. About 10 minutes of your time."),
    "long": ("16:9, about ten minutes, with chapters", "An outline and six chapters written one by one, "
                                                       "chapter timestamps in the description, and Shorts cut "
                                                       "from it afterwards. Plan half an hour."),
    "lofi": ("An hour of music over one animated loop", "Describe a cozy scene, approve a Pixabay animation, "
                                                        "pick the music, then render in the background. "
                                                        "Everything is free to use commercially."),
}
VISUALS_BLURB = {
    "pixabay": "stock footage and photos, ranked by a vision model (needs PIXABAY_API_KEY)",
    "diffusion": "AI panels drawn on your GPU, about a minute each",
    "illustration": "simple drawn panels, instant, no key or GPU",
    "placeholder": "plain colour slides, for testing the flow",
}


# ---------------------------------------------------------------------------
# Navigation state
# ---------------------------------------------------------------------------

def _init() -> None:
    st.session_state.setdefault("view", "library")
    st.session_state.setdefault("slug", None)
    st.session_state.setdefault("new_kind", None)
    st.session_state.setdefault("step_override", {})


def go(view: str, slug: str | None = None, kind: str | None = None) -> None:
    st.session_state["view"] = view
    if slug is not None:
        st.session_state["slug"] = slug
    st.session_state["new_kind"] = kind
    st.rerun()


def set_step(slug: str, step: int | None) -> None:
    overrides = st.session_state["step_override"]
    if step is None:
        overrides.pop(slug, None)
    else:
        overrides[slug] = step


# ---------------------------------------------------------------------------
# Small widgets
# ---------------------------------------------------------------------------

def failed(title: str, exc: BaseException) -> None:
    st.error(f"{title}: {exc}")
    with st.expander("Details (log and traceback)"):
        st.code(h.detail_of(exc), language="text")


def act(label: str, fn, *args, **kwargs):
    """Run an action under a spinner; on failure show it in place and return None."""
    try:
        with st.spinner(label):
            return fn(*args, **kwargs)
    except Exception as exc:  # the page must stay usable whatever failed
        failed(label.rstrip(".") , exc)
        return None


def chip(text: str) -> str:
    return f'<span class="kind-chip">{html.escape(text)}</span>'


def stepper(steps: list[str], done: list[bool], current: int) -> None:
    parts = []
    for i, name in enumerate(steps):
        cls = "done" if done[i] else ("current" if i == current else "todo")
        mark = "✓" if done[i] else str(i + 1)
        parts.append(f'<div class="step {cls}"><div class="dot">{mark}</div>'
                     f'<div class="label">{html.escape(name)}</div></div>')
    st.markdown(f'<div class="stepper">{"".join(parts)}</div>', unsafe_allow_html=True)


def done_row(slug: str, index: int, title: str, summary: str, editable: bool = True) -> None:
    col_a, col_b = st.columns([6, 1])
    with col_a:
        st.markdown(f'<div class="done-row"><b>✓ {index + 1}. {html.escape(title)}</b>'
                    f' <span class="muted">{html.escape(summary)}</span></div>', unsafe_allow_html=True)
    with col_b:
        if editable and st.button("Change", key=f"edit_{slug}_{index}", width="stretch"):
            set_step(slug, index)
            st.rerun()


def todo_row(index: int, title: str) -> None:
    st.markdown(f'<div class="todo-row">{index + 1}. {html.escape(title)}</div>', unsafe_allow_html=True)


def narrator_picker(key: str, default_engine: str, default_name: str = "") -> tuple[str, str]:
    engines = list(h.ENGINES)
    engine = st.selectbox("Voice engine", engines,
                          index=engines.index(default_engine) if default_engine in engines else 0,
                          key=f"{key}_engine",
                          help="kokoro = free, on this machine. gemini = free tier, needs GEMINI_API_KEY. "
                               "elevenlabs = paid. local = the system voice, for tests.")
    name = ""
    if engine in ("kokoro", "gemini"):
        cfg = h.cfg()
        story, fact = h.pool_for(cfg, engine, "story"), h.pool_for(cfg, engine, "fact")
        names = [""] + h.all_voices(cfg, engine)

        def label(n: str) -> str:
            if not n:
                return "Rotate: the least recently used narrator from the pool"
            tags = "/".join(t for t, pool in (("story", story), ("fact", fact)) if n in pool)
            return h.describe(engine, n) + (f"  [{tags}]" if tags else "")

        name = st.selectbox("Narrator", names, format_func=label,
                            index=names.index(default_name) if default_name in names else 0,
                            key=f"{key}_name")
        if name and st.button("🔊 Play a sample", key=f"{key}_sample"):
            sample = act("Synthesizing a sample...", h.voice_sample, engine, name)
            if sample:
                st.audio(str(sample))
    return engine, name


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

def sidebar(projects: list[dict]) -> None:
    with st.sidebar:
        st.markdown("## 🎬 YouTube Autopilot")
        if st.button("➕ New video", key="nav_new", type="primary", width="stretch"):
            go("new")
        if st.button("📚 Library", key="nav_lib", width="stretch"):
            go("library")
        st.markdown("---")
        st.markdown('<div class="muted">Recent</div>', unsafe_allow_html=True)
        for p in projects[:12]:
            icon = h.KIND_ICONS.get(p["format"], "🎬")
            title = p["title"][:34] + ("…" if len(p["title"]) > 34 else "")
            if st.button(f"{icon} {title}", key=f"nav_{p['slug']}", width="stretch"):
                go("project", slug=p["slug"])
        st.markdown("---")
        st.caption("Projects live in output/. Reload the page at any time; every step reads its state from disk.")


# ---------------------------------------------------------------------------
# Library
# ---------------------------------------------------------------------------

def library(projects: list[dict]) -> None:
    st.title("Your videos")
    if not projects:
        st.info("Nothing yet. Press New video to start a Short, a long video or a lofi mix.")
        if st.button("➕ New video", key="lib_new", type="primary"):
            go("new")
        return
    kinds = ["all", "short", "long", "lofi"]
    choice = st.pills("Show", kinds, default="all", key="lib_filter",
                      format_func=lambda k: "All" if k == "all" else f"{h.KIND_ICONS[k]} {h.KIND_LABELS[k]}")
    shown = [p for p in projects if choice in ("all", None) or p["format"] == choice]
    cols = st.columns(3)
    for i, p in enumerate(shown):
        with cols[i % 3]:
            with st.container(border=True):
                if p.get("card_image"):
                    st.image(p["card_image"], width="stretch")
                st.markdown(f'<div class="card-title">{h.KIND_ICONS.get(p["format"], "🎬")} '
                            f'{html.escape(p["title"])}</div>', unsafe_allow_html=True)
                status = h.status_line(p)
                cls = "ok" if status.startswith(("Published", "Ready")) else ("warn" if "failed" in status else "muted")
                st.markdown(f'<span class="{cls}">{html.escape(status)}</span> '
                            f'<span class="muted">· {html.escape(p.get("created", ""))}</span>',
                            unsafe_allow_html=True)
                if p.get("parent"):
                    st.caption(f"Cut from {p['parent']}")
                if st.button("Open", key=f"open_{p['slug']}", width="stretch"):
                    go("project", slug=p["slug"])


# ---------------------------------------------------------------------------
# New video
# ---------------------------------------------------------------------------

def new_video() -> None:
    st.title("New video")
    kind = st.session_state.get("new_kind")
    stepper(["Kind", "Details"], [bool(kind), False], 1 if kind else 0)
    if not kind:
        st.markdown("#### What do you want to make?")
        cols = st.columns(3)
        for col, k in zip(cols, ("short", "long", "lofi")):
            with col, st.container(border=True):
                st.markdown(f"### {h.KIND_ICONS[k]} {h.KIND_LABELS[k]}")
                st.markdown(f"**{KIND_BLURB[k][0]}**")
                st.markdown(f'<div class="muted">{KIND_BLURB[k][1]}</div>', unsafe_allow_html=True)
                if st.button("Choose", key=f"choose_{k}", type="primary", width="stretch"):
                    st.session_state["new_kind"] = k
                    st.rerun()
        return

    done_row("new", 0, "Kind", f"{h.KIND_ICONS[kind]} {h.KIND_LABELS[kind]}: {KIND_BLURB[kind][0]}")
    if st.session_state["step_override"].get("new") == 0:
        set_step("new", None)
        st.session_state["new_kind"] = None
        st.rerun()
    if kind == "lofi":
        new_lofi_form()
    else:
        new_narrated_form(kind)


def new_narrated_form(kind: str) -> None:
    st.markdown("#### 2. The idea")
    st.caption("A rough premise, question or fact topic is enough; the script is written from it. "
               "You review the script before anything else runs.")
    idea = st.text_area("Topic or idea", key="new_idea", height=110,
                        placeholder="e.g. Why do cats always land on their feet?"
                        if kind == "short" else "e.g. How the Great Hedge of India worked")
    col_a, col_b = st.columns(2)
    with col_a:
        fmt = st.radio("Treatment", ["auto", "story", "fact"], horizontal=True, key="new_fmt",
                       help="story = original fiction with a character and a turn; fact = an explainer with "
                            "claims to verify; auto lets the model decide from the idea.")
    with col_b:
        sources = list(h.SOURCES)
        default = h.cfg()["visuals"]["source"]
        visuals = st.selectbox("Visuals", sources, index=sources.index(default) if default in sources else 0,
                               key="new_visuals", format_func=lambda s: f"{s}: {VISUALS_BLURB[s]}")
    wait = "a few minutes: an outline, then one pass per chapter" if kind == "long" else "about a minute"
    if st.button("✍️ Write the script", key="new_write", type="primary"):
        if not idea.strip():
            st.warning("Type the idea first.")
        else:
            slug = act(f"Writing the script ({wait})...", h.create_script, idea.strip(), fmt, kind, visuals)
            if slug:
                set_step(slug, None)
                go("project", slug=slug)


def _apply_preset() -> None:
    choice = st.session_state.get("lofi_preset")
    if choice:
        st.session_state["lofi_idea"] = choice


def new_lofi_form() -> None:
    st.markdown("#### 2. The scene")
    st.caption("Describe one calm picture: who or what, where, time of day, light, weather. "
               "Planning writes the title, description and track names; drawing comes next, so nothing "
               "slow runs yet.")
    st.pills("Presets", h.lofi.PRESETS, key="lofi_preset", on_change=_apply_preset,
             format_func=lambda s: s.split(",")[0].replace("a ", "", 1).capitalize()[:34],
             label_visibility="collapsed")
    idea = st.text_area("Scene", key="lofi_idea", height=90,
                        placeholder="a student at a desk by a rainy window at night, warm lamp, a cup of tea")
    col_a, col_b, col_c = st.columns(3)
    with col_a:
        minutes = st.select_slider("Length", options=[10, 15, 20, 30, 45, 60, 90, 120],
                                   value=int(h.cfg().get("lofi", {}).get("minutes", 60)), key="lofi_minutes",
                                   format_func=lambda m: f"{m} min")
    with col_b:
        rain = st.selectbox("Rain", ["auto", "always", "never"], key="lofi_rain",
                            help="Rain streaks over the picture and soft rain under the music. "
                                 "auto = when the scene mentions rain.")
    with col_c:
        arts = list(PICTURE_LABELS)
        default_art = h.cfg().get("lofi", {}).get("picture", "pixabay")
        art = st.selectbox("Picture", arts, index=arts.index(default_art) if default_art in arts else 0,
                           key="lofi_art", format_func=PICTURE_LABELS.get)
    music = music_picker("lofi_new_music", h.cfg().get("music", {}).get("engine", "pixabay"), int(minutes))
    if st.button("🗺️ Plan the scene", key="lofi_plan", type="primary"):
        if not idea.strip():
            st.warning("Describe the scene or pick a preset first.")
        else:
            slug = act("Planning the scene (a few seconds)...", h.lofi_plan, idea.strip(), int(minutes),
                       art, music, rain)
            if slug:
                set_step(slug, None)
                go("project", slug=slug)


def music_picker(key: str, default: str, minutes: int = 60) -> str:
    engines = list(MUSIC_LABELS)
    engine = st.selectbox("Music", engines, index=engines.index(default) if default in engines else 0, key=key,
                          format_func=MUSIC_LABELS.get)
    if engine == "pixabay":
        lib = h.music_library()
        if lib["count"] == 0:
            st.warning(f"The music library is empty. Download lofi tracks from {lib['url']} into "
                       f"`{lib['dir']}` (mp3 is fine; keep the licence certificate the site offers), then reload.")
        else:
            enough = lib["unused_minutes"] >= minutes
            note = (f"{lib['count']} tracks ({lib['minutes']:.0f} min), {lib['unused']} never used "
                    f"({lib['unused_minutes']:.0f} min).")
            if enough:
                st.caption("✅ " + note)
            else:
                st.caption("⚠️ " + note + f" Add more from {lib['url']} for a {minutes}-minute video with "
                           "no repeats; otherwise the least-used tracks are reused.")
    if engine == "acestep":
        if h.acestep_running():
            st.caption("✅ ACE-Step server is running.")
        else:
            st.warning("ACE-Step server is not running. Start it (README, 'Lofi music videos') before "
                       "rendering, or every track will be composed procedurally instead.")
    return engine


# ---------------------------------------------------------------------------
# Project wizard: shared
# ---------------------------------------------------------------------------

def project_header(p: dict, script: dict) -> None:
    col_a, col_b = st.columns([5, 1])
    with col_a:
        st.markdown(f"<h1>{html.escape(script.get('title', p['slug']))}</h1>", unsafe_allow_html=True)
        bits = [chip(f"{h.KIND_ICONS.get(p['format'], '🎬')} {h.KIND_LABELS.get(p['format'], p['format'])}")]
        if script.get("content_type") in ("story", "fact"):
            bits.append(chip(script["content_type"]))
        if script.get("voice", {}).get("name"):
            bits.append(chip(f"🎙 {h.describe(script['voice'].get('engine', ''), script['voice']['name'])}"))
        if script.get("parent"):
            bits.append(chip(f"cut from {script['parent']}"))
        st.markdown(" ".join(bits) + f' <span class="mono">{html.escape(p["slug"])}</span>',
                    unsafe_allow_html=True)
    with col_b:
        st.write("")
        if st.button("📚 Library", key="hdr_lib", width="stretch"):
            go("library")


def current_step(slug: str, done: list[bool]) -> int:
    auto = next((i for i, d in enumerate(done) if not d), len(done) - 1)
    chosen = st.session_state["step_override"].get(slug)
    if chosen is not None and 0 <= chosen <= auto:
        return chosen
    set_step(slug, None)
    return auto


def continue_button(slug: str, key: str, label: str = "Continue →", **state) -> None:
    if st.button(label, key=key, type="primary"):
        if state:
            h.set_ui_state(slug, **state)
        set_step(slug, None)
        st.rerun()


def danger_zone(slug: str) -> None:
    with st.expander("Danger zone"):
        st.caption(f"Deletes output/{slug} and its history entry. Cannot be undone.")
        sure = st.checkbox("I want to delete this project", key=f"sure_{slug}")
        if st.button("🗑 Delete project", key=f"del_{slug}", disabled=not sure):
            h.delete_project(slug)
            go("library")


# ---------------------------------------------------------------------------
# Project wizard: narrated (short / long)
# ---------------------------------------------------------------------------

NARRATED_STEPS = ["Script", "Narrator", "Visuals", "Render", "Publish"]


def narrated_wizard(p: dict, script: dict) -> None:
    slug, state = p["slug"], p["state"]
    done = [bool(state.get("script_ok")), p["voice"], p["selected"] and bool(state.get("visuals_ok")),
            p["video"], bool(p["youtube_id"])]
    cur = current_step(slug, done)
    stepper(NARRATED_STEPS, done, cur)

    # 1. Script
    if cur == 0:
        step_script(slug, script)
    elif done[0]:
        words = sum(len(s.get("narration", "").split()) for s in script.get("scenes", []))
        done_row(slug, 0, "Script", f"{p['scene_count']} scenes, {words} words")
    # 2. Narrator
    if cur == 1:
        step_narrator(slug, script, p)
    elif cur > 1 or done[1]:
        v = script.get("voice") or {}
        total = sum(p.get("durations", []))
        done_row(slug, 1, "Narrator", f"{h.describe(v.get('engine', ''), v.get('name', ''))} via "
                                     f"{v.get('engine', '')}, {total / 60:.1f} min of narration")
    else:
        todo_row(1, "Narrator")
    # 3. Visuals
    if cur == 2:
        step_visuals(slug, script, p)
    elif done[2]:
        done_row(slug, 2, "Visuals", f"{p['visuals']}; a clip or picture chosen for every scene")
    elif cur < 2:
        todo_row(2, "Visuals")
    # 4. Render
    if cur == 3:
        step_render(slug, script, p)
    elif done[3]:
        done_row(slug, 3, "Render", f"{h.video_duration(slug) / 60:.1f} min, final.mp4 ready")
    elif cur < 3:
        todo_row(3, "Render")
    # 5. Publish
    if cur == 4:
        step_publish(slug, script, p)
    elif cur < 4:
        todo_row(4, "Publish")
    st.markdown("---")
    danger_zone(slug)


def step_script(slug: str, script: dict) -> None:
    st.markdown("### 1. Read the script")
    st.caption("Everything downstream reads from this. Edit output/…/script.json by hand if you want "
               "changes, then reload the page; the claims list is what to fact-check before publishing.")
    if script.get("alternative_titles"):
        st.markdown("**Other titles the model offered:** " + " · ".join(script["alternative_titles"][:3]))
    st.markdown(f"**Description:** {script.get('description', '')}")
    if script.get("tags"):
        st.caption("Tags: " + ", ".join(script["tags"][:12]))
    chapters = {int(c.get("first_scene", -1)): c.get("title", "") for c in script.get("chapters", []) or []}
    with st.expander(f"Scenes ({len(script.get('scenes', []))})", expanded=True):
        for i, s in enumerate(script.get("scenes", [])):
            if i in chapters:
                st.markdown(f"**Chapter: {chapters[i]}**")
            st.markdown(f"**{i + 1}.** {s.get('narration', '')}")
            st.caption(f"Picture: {s.get('visual_description', '')}  ·  on screen: "
                       f"{s.get('on_screen_text') or 'nothing'}")
    claims = script.get("claims_to_verify") or []
    if claims:
        with st.expander(f"⚠️ {len(claims)} claims to verify before publishing"):
            for c in claims:
                st.markdown(f"- {c}")
    continue_button(slug, f"script_ok_{slug}", "Looks good, choose a narrator →", script_ok=True)


def step_narrator(slug: str, script: dict, p: dict) -> None:
    st.markdown("### 2. Choose the narrator")
    st.caption("Each scene is spoken separately so captions can follow the words. Rotation picks a "
               "different narrator from the pool for every video so the channel never sounds templated.")
    current = script.get("voice") or {}
    engine, name = narrator_picker(f"nar_{slug}", current.get("engine") or h.cfg()["voice"].get("engine", "kokoro"),
                                   current.get("name", ""))
    if p["voice"]:
        st.success("Narration exists. Narrating again replaces it and removes the rendered video.")
        with st.expander("Listen per scene"):
            for i in range(p["scene_count"]):
                mp3 = h.OUTPUT_DIR / slug / "audio" / f"scene_{i:02d}.mp3"
                if mp3.exists():
                    st.caption(f"Scene {i + 1} · {p['durations'][i]:.1f}s")
                    st.audio(str(mp3))
    label = "🎙 Narrate again" if p["voice"] else f"🎙 Narrate {p['scene_count']} scenes"
    col_a, col_b = st.columns([1, 1])
    with col_a:
        if st.button(label, key=f"narrate_{slug}", type="primary"):
            wait = "about a minute" if p["format"] == "short" else "a few minutes"
            if act(f"Narrating ({wait})...", h.narrate, slug, engine, name) is not None:
                set_step(slug, None)
                st.rerun()
    with col_b:
        if p["voice"]:
            continue_button(slug, f"keep_voice_{slug}", "Keep this narration →")


def step_visuals(slug: str, script: dict, p: dict) -> None:
    st.markdown("### 3. Pick the visuals")
    candidates = h.load_candidates(slug)
    sources = list(h.SOURCES)
    if not candidates:
        st.caption("Fetch a handful of candidates per scene, then choose one for each. Stock results are "
                   "ranked by how well they match the scene.")
        source = st.selectbox("Source", sources, index=sources.index(p["visuals"]) if p["visuals"] in sources else 0,
                              key=f"src_{slug}", format_func=lambda s: f"{s}: {VISUALS_BLURB[s]}")
        wait = {"pixabay": "about a minute", "diffusion": "about a minute per scene"}.get(source, "a moment")
        if st.button("🔍 Fetch candidates", key=f"fetch_{slug}", type="primary"):
            if act(f"Fetching candidates via {source} ({wait})...", h.fetch_candidates, slug, source) is not None:
                st.rerun()
        return

    with st.expander(f"Source: {p['visuals']}  (switch and fetch again)"):
        source = st.selectbox("Source", sources, index=sources.index(p["visuals"]) if p["visuals"] in sources else 0,
                              key=f"resrc_{slug}", format_func=lambda s: f"{s}: {VISUALS_BLURB[s]}")
        st.caption("Replaces every scene's candidates and clears your picks; uploads are kept.")
        if st.button("Fetch again", key=f"refetch_{slug}"):
            if act(f"Fetching candidates via {source}...", h.refetch_candidates, slug, source) is not None:
                st.rerun()

    chosen = p["chosen"]
    missing = [i + 1 for i in range(p["scene_count"]) if i not in chosen]
    if missing:
        st.info(f"Choose a clip or picture for scene{'s' if len(missing) > 1 else ''} "
                f"{', '.join(map(str, missing[:8]))}{'…' if len(missing) > 8 else ''}.")
    else:
        st.success("Every scene has a pick.")
        continue_button(slug, f"vis_ok_top_{slug}", "Continue to render →", visuals_ok=True)

    for i, scene in enumerate(script.get("scenes", [])):
        scene_chooser(slug, i, scene, candidates.get(i, []), chosen.get(i), p)

    if not missing:
        continue_button(slug, f"vis_ok_{slug}", "Continue to render →", visuals_ok=True)


def scene_chooser(slug: str, i: int, scene: dict, cands: list[dict], current: str | None, p: dict) -> None:
    with st.container(border=True):
        head, dur, actions = st.columns([5, 1, 2])
        with head:
            st.markdown(f"**Scene {i + 1}**  {'✅' if current else '⬜'}")
            st.caption(scene.get("narration", "")[:220])
        with dur:
            d = p["durations"][i] if i < len(p.get("durations", [])) else 0
            st.markdown(f'<span class="kind-chip">{d:.1f}s</span>', unsafe_allow_html=True)
        with actions:
            src = p["visuals"]
            more = {"pixabay": "More stock", "diffusion": "New take"}.get(src, "Fetch more")
            if st.button(f"🔄 {more}", key=f"more_{slug}_{i}", width="stretch"):
                page_key = f"page_{slug}_{i}"
                st.session_state[page_key] = st.session_state.get(page_key, 1) + 1
                got = act("Fetching more...", h.regenerate_scene, slug, i, src, st.session_state[page_key])
                if got:
                    st.rerun()
                elif got == []:
                    st.warning("Nothing more came back for this scene.")
            if src != "diffusion" and h.diffusion.available():
                if st.button("🎨 AI panel", key=f"ai_{slug}_{i}", width="stretch",
                             help="Draw this scene on your GPU and add it here (about a minute)."):
                    if act("Drawing an AI panel...", h.regenerate_scene, slug, i, "diffusion", 1):
                        st.rerun()
        if cands:
            best = max((c.get("metadata", {}).get("score", -1) for c in cands), default=-1)
            if 0 <= best < h.LOW_MATCH_SCORE:
                st.warning(f"Best stock match scored {best}/10. Try an AI panel or upload your own.")
            cols = st.columns(4)   # a lone vertical panel must not fill the page
            for k, cand in enumerate(cands):
                with cols[k % len(cols)]:
                    thumb = h.get_media_thumb(cand["path"])
                    picked = bool(current) and Path(current).resolve() == Path(cand["path"]).resolve()
                    if thumb and thumb.exists():
                        st.image(str(thumb), width="stretch")
                    meta = cand.get("metadata", {})
                    if cand["source"] == "diffusion":
                        label = f"AI panel, take {int(meta.get('variant', 0)) + 1}"
                    else:
                        kind = "photo" if cand.get("type") == "image" else "video"
                        label = f"{cand['source']} {kind}" + (f" · {meta['duration']:.0f}s" if meta.get("duration") else "")
                    score = meta.get("score")
                    if isinstance(score, int) and score >= 0:
                        label += f" · match {score}/10"
                    st.caption(label, help=meta.get("reason") or None)
                    if st.button("✓ Chosen" if picked else "Use this", key=f"pick_{slug}_{i}_{k}",
                                 type="primary" if picked else "secondary", width="stretch"):
                        h.select_media(slug, i, str(cand["path"]))
                        st.rerun()
        else:
            st.warning("No candidates for this scene; fetch more or upload your own.")
        with st.expander("Upload your own clip or picture"):
            up = st.file_uploader(f"Scene {i + 1}", type=["mp4", "mov", "jpg", "jpeg", "png", "webp"],
                                  key=f"up_{slug}_{i}", label_visibility="collapsed")
            if up is not None:
                sig = f"{i}:{up.name}:{up.size}"
                if st.session_state.get(f"upsig_{slug}_{i}") != sig:
                    h.save_upload(slug, i, up.name, up.getbuffer().tobytes())
                    st.session_state[f"upsig_{slug}_{i}"] = sig
                    st.rerun()
        if current:
            st.caption(f"Chosen: {Path(current).name}")


def step_render(slug: str, script: dict, p: dict) -> None:
    st.markdown("### 4. Render")
    cfg = h.project_cfg(slug)
    w, hgt = cfg["video"]["resolution"]
    total = sum(p.get("durations", []))
    st.markdown(f"- **{p['scene_count']} scenes**, {total / 60:.1f} minutes of narration\n"
                f"- **{w}×{hgt}**, captions {'on' if cfg['captions'].get('enabled', True) else 'off'}, "
                f"stills get camera moves, stock clips keep their motion\n"
                f"- Music bed: {'on' if cfg['music'].get('enabled') else 'off'} (assets/music)")
    wait = "a minute or two" if p["format"] == "short" else "several minutes"
    label = "🎬 Render again" if p["video"] else "🎬 Render the video"
    if st.button(label, key=f"render_{slug}", type="primary"):
        if act(f"Rendering with ffmpeg ({wait})...", h.render_video, slug) is not None:
            set_step(slug, None)
            st.rerun()
    if p["video"]:
        continue_button(slug, f"keep_render_{slug}", "Keep this render →")


def step_publish(slug: str, script: dict, p: dict) -> None:
    st.markdown("### 5. Review and publish")
    out_dir = h.OUTPUT_DIR / slug
    col_v, col_t = st.columns(2)
    with col_v:
        st.video(str(out_dir / "final.mp4"))
    with col_t:
        if (out_dir / "thumbnail.jpg").exists():
            st.image(str(out_dir / "thumbnail.jpg"), caption="Thumbnail (replace the file to use your own)",
                     width="stretch")
    if script.get("chapter_times"):
        from pipeline.longform import format_timestamp
        with st.expander("Chapters written into the description"):
            for c in script["chapter_times"]:
                st.markdown(f"`{format_timestamp(c.get('seconds', 0))}` {c.get('title', '')}")
        with st.expander("✂️ Cut Shorts from this video"):
            st.caption("The hook and a self-contained chapter or two, re-rendered vertically with the same "
                       "narration and media. Each becomes its own project.")
            n = st.number_input("How many", 1, 6, int(h.cfg().get("cuts", {}).get("count", 2)), key=f"cutn_{slug}")
            if st.button("Cut Shorts", key=f"cut_{slug}"):
                made = act("Choosing excerpts and rendering them (a minute or two each)...", h.cut_shorts, slug, int(n))
                if made is not None:
                    st.success("Made: " + ", ".join(made) if made else "Done; see the library.")
    with st.expander("Fine-tune the timing"):
        speed = st.slider("Whole video speed", 0.5, 2.0, 1.0, 0.1, key=f"spd_{slug}")
        if st.button("Apply speed", key=f"spd_btn_{slug}"):
            if act("Re-timing the video...", h.apply_speed, slug, speed) is not None:
                st.rerun()
        idx = st.number_input("Scene to re-time", 1, max(1, p["scene_count"]), 1, key=f"sc_{slug}")
        sspeed = st.slider("Scene speed", 0.5, 2.0, 1.0, 0.1, key=f"sspd_{slug}")
        if st.button("Re-time scene and render again", key=f"sspd_btn_{slug}"):
            if act("Re-timing the scene and rendering...", h.retime_scene, slug, int(idx) - 1, sspeed) is not None:
                st.rerun()
    publish_controls(slug, p)


def publish_controls(slug: str, p: dict) -> None:
    if p["youtube_id"]:
        when = f" (public at {p['publish_at']})" if p.get("publish_at") else ""
        st.success(f"Published: https://youtu.be/{p['youtube_id']}{when}")
    st.markdown("#### Publish")
    col_a, col_b = st.columns(2)
    with col_a:
        privacy = st.selectbox("Privacy", ["private", "unlisted", "public"], key=f"priv_{slug}")
    with col_b:
        when = st.selectbox("When", ["now", "slot", "at"], key=f"when_{slug}",
                            format_func=lambda w: {"now": "Now", "slot": f"Next free slot: {h.next_slot_label()}",
                                                   "at": "A time I pick"}[w])
    at = st.text_input("Go public at (local time)", placeholder="2026-11-06 18:00", key=f"at_{slug}") if when == "at" else ""
    st.caption("The description gets the AI disclosure from config.yaml and the upload sets YouTube's "
               "altered-or-synthetic content flag. Scheduled uploads are private until their time.")
    label = "🚀 Publish again" if p["youtube_id"] else "🚀 Publish to YouTube"
    if st.button(label, key=f"pub_{slug}", type="primary"):
        url = act("Uploading to YouTube...", h.publish, slug, privacy, when, at)
        if url:
            st.success(f"Published: {url}")
            st.rerun()


# ---------------------------------------------------------------------------
# Project wizard: lofi
# ---------------------------------------------------------------------------

LOFI_STEPS = ["Scene", "Look", "Music", "Render", "Publish"]


def lofi_wizard(p: dict, script: dict) -> None:
    slug, state = p["slug"], p["state"]
    rendering = p["render"] == "running"
    done = [bool(state.get("scene_ok")), bool(p.get("scene_image")) and bool(state.get("look_ok")),
            bool(state.get("music_ok")), p["video"] and not rendering, bool(p["youtube_id"])]
    cur = 3 if rendering else current_step(slug, done)
    stepper(LOFI_STEPS, done, cur)
    settings = p.get("settings", {})

    if cur == 0:
        lofi_step_scene(slug, script)
    elif done[0]:
        done_row(slug, 0, "Scene", (script.get("lofi_plan") or {}).get("visual_description", "")[:110], editable=not rendering)
    if cur == 1:
        lofi_step_look(slug, script, p)
    elif done[1]:
        done_row(slug, 1, "Look", f"{settings.get('visuals', '')}, take {int(settings.get('variant', 0)) + 1}", editable=not rendering)
    elif cur < 1:
        todo_row(1, "Look")
    if cur == 2:
        lofi_step_music(slug, script, p)
    elif done[2]:
        done_row(slug, 2, "Music", f"{settings.get('music', '')}, rain {settings.get('rain', 'auto')}", editable=not rendering)
    elif cur < 2:
        todo_row(2, "Music")
    if cur == 3:
        lofi_step_render(slug, script, p)
    elif done[3]:
        done_row(slug, 3, "Render", f"{h.video_duration(slug) / 60:.0f} min, {len(p.get('tracks', []))} tracks")
    elif cur < 3:
        todo_row(3, "Render")
    if cur == 4:
        lofi_step_publish(slug, script, p)
    elif cur < 4:
        todo_row(4, "Publish")
    st.markdown("---")
    if not rendering:
        danger_zone(slug)


def lofi_step_scene(slug: str, script: dict) -> None:
    st.markdown("### 1. The scene and the words")
    st.caption("The plan from your idea. Fix anything here; the tracklist is appended to the description "
               "with timestamps when the video is rendered.")
    plan = script.get("lofi_plan") or {}
    title = st.text_input("Title", value=script.get("title", ""), key=f"lt_{slug}")
    thumb = st.text_input("Thumbnail text", value=script.get("thumbnail_text", ""), key=f"lth_{slug}")
    desc = st.text_area("Description", value=script.get("description", ""), height=130, key=f"ld_{slug}")
    tags = st.text_input("Tags (comma separated)", value=", ".join(script.get("tags", [])), key=f"ltg_{slug}")
    st.markdown(f"**Picture to draw:** {plan.get('visual_description') or script.get('idea', '')}")
    if plan.get("track_titles"):
        st.caption("Track names: " + ", ".join(plan["track_titles"]))
    if st.button("Save and continue →", key=f"scene_ok_{slug}", type="primary"):
        h.lofi_save_details(slug, title, desc, thumb, [t.strip() for t in tags.split(",")])
        h.set_ui_state(slug, scene_ok=True)
        set_step(slug, None)
        st.rerun()


def lofi_step_look(slug: str, script: dict, p: dict) -> None:
    st.markdown("### 2. Approve the picture")
    st.caption("This one still becomes the whole video: a breathing camera, rain if the scene is rainy, "
               "lamp flicker and grain. Draw it, ask for another take until you like it, then continue.")
    settings = p.get("settings", {})
    arts = list(PICTURE_LABELS)
    art = st.selectbox("Picture", arts, index=arts.index(settings.get("visuals")) if settings.get("visuals") in arts else 0,
                       key=f"art_{slug}", format_func=PICTURE_LABELS.get)
    image = p.get("scene_image")
    clip = h.lofi_clip(slug) if art == "pixabay" else None
    if clip:
        st.video(clip["path"])
        st.caption(f"Pixabay animation {clip.get('id', '')} by {clip.get('user', '')}, "
                   f"{float(clip.get('duration', 0)):.0f}s loop · {clip.get('tags', '')[:80]}")
    elif image and art != "pixabay":
        st.image(image, width="stretch")
    have = bool(clip) if art == "pixabay" else bool(image)
    col_a, col_b, col_c = st.columns(3)
    with col_a:
        first = "🔍 Find a loop" if art == "pixabay" else "🖼 Draw the scene"
        again = "🔍 Search again" if art == "pixabay" else "🖼 Redraw"
        if st.button(again if have else first, key=f"draw_{slug}", type="secondary" if have else "primary",
                     width="stretch"):
            wait = "Finding a Pixabay loop..." if art == "pixabay" else "Drawing the scene..."
            if act(wait, h.lofi_draw, slug, art, 0 if art == "pixabay" else None) is not None:
                st.rerun()
    with col_b:
        if art in ("pixabay", "diffusion") and have and st.button("🎲 Another one", key=f"take_{slug}", width="stretch",
                                                                   help="The next candidate loop, or a new seed."):
            if act("Fetching another...", h.lofi_draw, slug, art, int(settings.get("variant", 0)) + 1) is not None:
                st.rerun()
    with col_c:
        if have and st.button("Use this →", key=f"look_ok_{slug}", type="primary", width="stretch"):
            h.set_ui_state(slug, look_ok=True)
            set_step(slug, None)
            st.rerun()


def lofi_step_music(slug: str, script: dict, p: dict) -> None:
    st.markdown("### 3. Choose the music")
    settings = p.get("settings", {})
    engine = music_picker(f"eng_{slug}", settings.get("music", "pixabay"), int(settings.get("minutes", 60)))
    rains = ["auto", "always", "never"]
    rain = st.selectbox("Rain under the music", rains, index=rains.index(settings.get("rain")) if settings.get("rain") in rains else 0,
                        key=f"rain_{slug}", help="Kept quiet under the music (lofi.ambience_volume).")
    lib = h.music_library()
    if lib["ambience_files"]:
        st.caption(f"Rain sound: {lib['ambience_files']} file(s) in the ambience library, least-used first.")
    else:
        st.caption(f"Rain sound: synthesized. For real rain, drop a file from {lib['ambience_url']} into "
                   f"`{lib['ambience_dir']}`.")
    titles = (script.get("lofi_plan") or {}).get("track_titles") or []
    if titles and engine != "pixabay":
        st.caption("Track names: " + ", ".join(titles[:8]) + (" …" if len(titles) > 8 else ""))
    key = f"prev_{slug}"
    col_a, col_b = st.columns(2)
    with col_a:
        if st.button("🎵 Preview the first track", key=f"pv_{slug}", width="stretch",
                     help="The full render reuses it, so nothing is wasted."):
            wait = {"acestep": "about 9 minutes with ACE-Step", "pixabay": "instant"}.get(engine, "about a minute")
            track = act(f"Composing the first track ({wait})...", h.lofi_preview_track, slug, engine)
            if track:
                st.session_state[key] = track
    with col_b:
        if st.button("Continue →", key=f"music_ok_{slug}", type="primary", width="stretch"):
            h.lofi_save_settings(slug, music=engine, rain=rain)
            h.set_ui_state(slug, music_ok=True)
            set_step(slug, None)
            st.rerun()
    track = st.session_state.get(key)
    if track and (h.OUTPUT_DIR / slug / "music" / track["path"]).exists():
        st.audio(str(h.OUTPUT_DIR / slug / "music" / track["path"]))
        extra = f" · {track['bpm']} bpm, {track['key']}" if track.get("bpm") else ""
        st.caption(f"{track['title']} ({track['engine']}){extra}, {track['seconds'] / 60:.1f} min")


@st.fragment(run_every="5s")
def lofi_progress(slug: str) -> None:
    log_path = h.OUTPUT_DIR / slug / "render.log"
    st.code(h.log_tail(log_path, 2500) or "(starting...)", language="text")
    if h.log_state(log_path) != "running":
        st.rerun(scope="app")


def lofi_step_render(slug: str, script: dict, p: dict) -> None:
    st.markdown("### 4. Render")
    if p["render"] == "running":
        st.info("Rendering in the background. This page follows the log; you can open other videos meanwhile.")
        lofi_progress(slug)
        return
    if p["render"] == "failed":
        st.error("The last render failed. The log is below; fix the cause and render again.")
        with st.expander("Render log", expanded=True):
            st.code(h.log_tail(h.OUTPUT_DIR / slug / "render.log", 5000), language="text")
    settings = p.get("settings", {})
    minutes = st.select_slider("Length", options=[5, 10, 15, 20, 30, 45, 60, 90, 120],
                               value=int(settings.get("minutes", 60)) if int(settings.get("minutes", 60)) in (5, 10, 15, 20, 30, 45, 60, 90, 120) else 60,
                               key=f"min_{slug}", format_func=lambda m: f"{m} min")
    engine = settings.get("music", "pixabay")
    if engine == "pixabay":
        music_line = f"- **Music from the Pixabay library**, least-used tracks first; no generation time"
    else:
        n = h.tracks_needed(minutes)
        per = 9 if engine == "acestep" else 1
        music_line = f"- **{n} tracks** with {engine}, roughly {n * per} minutes of generation"
    st.markdown(f"{music_line}\n"
                f"- Picture: {PICTURE_LABELS.get(settings.get('visuals', ''), settings.get('visuals', ''))}, "
                f"rain {settings.get('rain', 'auto')}\n"
                f"- The video itself assembles in seconds and runs in the background; nothing already made is redone")
    if settings.get("music") == "acestep" and not h.acestep_running():
        st.warning("ACE-Step server is not running; every track would be composed procedurally. Start it "
                   "first, or go back to Music and choose the composer.")
    label = "🎬 Render again" if p["video"] else "🎬 Render in the background"
    col_a, col_b = st.columns(2)
    with col_a:
        if st.button(label, key=f"lrender_{slug}", type="primary", width="stretch"):
            h.lofi_save_settings(slug, minutes=float(minutes))
            if act("Starting the render...", h.lofi_start_render, slug) is not None:
                st.rerun()
    with col_b:
        if p["video"]:
            continue_button(slug, f"keep_lofi_{slug}", "Keep this render →")


def lofi_step_publish(slug: str, script: dict, p: dict) -> None:
    st.markdown("### 5. Review and publish")
    out_dir = h.OUTPUT_DIR / slug
    col_v, col_t = st.columns(2)
    with col_v:
        st.video(str(out_dir / "final.mp4"))
    with col_t:
        if (out_dir / "thumbnail.jpg").exists():
            st.image(str(out_dir / "thumbnail.jpg"), caption="Thumbnail", width="stretch")
    with st.expander("Tracklist (written into the description with timestamps)", expanded=True):
        for line, track in zip(script.get("tracklist", []), script.get("tracks", [])):
            extra = f" · {track.get('bpm')} bpm, {track.get('key')}" if track.get("bpm") else ""
            st.markdown(f"`{html.escape(str(line))}`{html.escape(extra)}  ({track.get('engine', '')})")
        for credit in script.get("credits", []):
            st.caption(credit)
    publish_controls(slug, p)


# ---------------------------------------------------------------------------
# Router
# ---------------------------------------------------------------------------

def project_view(slug: str) -> None:
    script = h.read_script(slug)
    if not script:
        st.error(f"output/{slug} has no script.json any more.")
        if st.button("Back to the library"):
            go("library")
        return
    p = h.progress(slug)
    project_header(p, script)
    if p["format"] == "lofi":
        lofi_wizard(p, script)
    else:
        narrated_wizard(p, script)


def main() -> None:
    _init()
    projects = h.list_projects()
    sidebar(projects)
    view = st.session_state["view"]
    if view == "new":
        new_video()
    elif view == "project" and st.session_state.get("slug"):
        project_view(st.session_state["slug"])
    else:
        library(projects)


main()
