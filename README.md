# YouTube Autopilot

One command turns a topic into a finished, captioned, voiced YouTube video with a thumbnail,
then stops so you can review it. Publishing is a separate command, on purpose.

```
idea    ->  script  ->  voice  ->  visuals  ->  video + captions  ->  thumbnail  ->  (you review)  ->  publish
                   Gemini by default; Anthropic optional             Kokoro or Gemini TTS, narrator rotates    stock, AI panels, drawn  ffmpeg  Pillow  YouTube API
```

## What is automated and what is not

| Step | Automated | Your job |
|---|---|---|
| Topic selection | Yes, the selected LLM proposes ideas and the pipeline skips topics already made | Optional: pass `--idea` |
| Script, title, description, tags, storyboard | Yes, it expands your rough story premise or fact topic | Read `script.md`, check the visuals and verify factual claims |
| Voiceover | Yes: a different narrator per video, rotated from a pool of free voices (Kokoro locally, or Gemini TTS) | Listen, or pick another narrator in the UI |
| Visuals per scene | Yes: Pixabay stock footage by default, or original flat panels drawn locally (`--visuals illustration`) | Pick the clip per scene in the browser UI, or review the automatic choice |
| Editing, captions, music, export | Yes (ffmpeg) | Nothing |
| Thumbnail | Yes | Optional: replace `thumbnail.jpg` with your own |
| Publishing | On request: now, at a time you give, or at the next free slot from `youtube.schedule` | Review first; `run.py queue --publish` ships everything you approved |
| Upload | On request only | Watch the video, then run `publish` |

Review the script and final video before publishing. Facts can be misstated, and the local
motion-comic renderer uses simple shapes rather than detailed AI artwork. Automation does not
guarantee views or monetization.

## Setup (once, about 20 minutes)

Works on Windows, macOS, and Linux. The only system tools are Python 3.12 and ffmpeg.

### Windows

1. Open PowerShell and install the tools (then close and reopen PowerShell so ffmpeg is on the PATH):

```powershell
winget install Python.Python.3.12
```

```powershell
winget install Gyan.FFmpeg
```

   Check with `ffmpeg -version` and `py -3.12 --version`. If winget is unavailable, download Python from python.org (tick "Add to PATH") and the ffmpeg "essentials" build from gyan.dev, unzip it, and add its `bin` folder to your PATH.

2. In the project folder, create the environment and install dependencies:

```powershell
py -3.12 -m venv .venv; .venv\Scripts\pip install -r requirements.txt
```

   Every command below uses `.venv\Scripts\python` on Windows where the examples show `.venv/bin/python`.

### macOS

1. Install the tools:

```bash
brew install python@3.12 ffmpeg
```

2. Create the environment and install dependencies:

```bash
python3.12 -m venv .venv && .venv/bin/pip install -r requirements.txt
```

### Both

3. Copy `.env.example` to `.env` and add keys:
   - `GEMINI_API_KEY` from [Google AI Studio](https://aistudio.google.com/api-keys). Gemini is the default script model; `gemini-3.8-flash` currently has free input and output on the free tier, subject to changing quotas and rate limits. If it is busy, the app retries and falls back to `gemini-3.6-flash`.
   - Gemini free-tier prompts and responses may be reviewed and used to improve Google products. Do not submit private or sensitive material. See [Google's Gemini API terms](https://ai.google.dev/gemini-api/terms).
   - `ANTHROPIC_API_KEY` from console.anthropic.com only if you set `LLM_PROVIDER=anthropic`.
   - `ELEVENLABS_API_KEY` and `ELEVENLABS_VOICE_ID` only if you want to switch to ElevenLabs. The default Kokoro voices are local and need no voice API key; `voice.engine: gemini` reuses `GEMINI_API_KEY`.
   - `PIXABAY_API_KEY` from [Pixabay's video API](https://pixabay.com/api/docs/) for the default `pixabay` visuals. Set `visuals.source` to `illustration` in `config.yaml` (or pass `--visuals illustration`) to draw panels locally with no image API key.
   - `KAGGLE_API_TOKEN` only for doodle explainers drawn on Kaggle's free GPUs: on [kaggle.com/settings](https://www.kaggle.com/settings), under API, press "Generate New Token" and paste it here. Also verify your phone number on Kaggle (Settings, Phone verification); without it, notebooks get neither a GPU nor the internet access they need to download the image model. See "Doodle explainers" below.

   Kokoro's ONNX runtime is installed with `requirements.txt`. On the first Kokoro run, the app downloads the model and voice files once (a few hundred MB) into `cache/tts/kokoro/`; it then synthesizes locally on the CPU.

4. Drop a few royalty-free `.mp3` tracks into `assets/music/`. YouTube Audio Library is the safest source. Leave the folder empty to skip music.

5. Edit `config.yaml`: channel name, niche, audience, tone. The script quality depends heavily on these four lines.

6. For publishing, follow the setup comment at the top of `pipeline/publish.py` to create `client_secrets.json`.

7. Prove the install with a dry run. It needs no API keys and uses the built-in Windows or macOS voice:

```bash
.venv/bin/python run.py make --dry-run
```

The dry run uses a sample story, the built-in system voice, and locally drawn illustrations. To use stock footage instead, run:

```bash
.venv/bin/python run.py make --dry-run --visuals pixabay
```

On Windows PowerShell:

```powershell
.\.venv\Scripts\python.exe run.py make --dry-run
```

Add `--visuals pixabay` only if you want stock footage and have configured a Pixabay API key.

## Browser UI (pick the clip for every scene)

```bash
.venv/bin/python -m streamlit run ui/app.py
```

On Windows PowerShell:

```powershell
.\.venv\Scripts\python.exe -m streamlit run ui/app.py
```

The studio is a step-by-step wizard. **Library** lists every project in `output/` as a card with where it stands. **New video** asks what to make (Short, Long video, Doodle explainer, Lofi music) and then shows the one form that kind needs. A **project** page has a stepper on top: finished steps fold to one line with a Change button, the current step is open, later steps are grey.

- Short and long videos: **Script** (read it, check the claims) → **Narrator** (pick, play a sample, narrate) → **Visuals** (fetch candidates per scene, choose one, upload your own, ask for an AI panel) → **Render** → **Publish** (review, fine-tune timing, cut Shorts from a long video, schedule the upload).
- Doodle explainers: **Script** (every sentence with the drawing planned for it) → **Narrator** (shows whether the narration lands in 9-11 minutes) → **Drawings** (send them to Kaggle in the background, look through the gallery, tick bad ones and redraw just those) → **Render** (in the background) → **Publish**.
- Lofi: **Scene** (title, description, tags) → **Look** (draw the picture, ask for another take, approve) → **Music** (engine, rain, preview the first track) → **Render** (length; runs in the background while the page follows the log) → **Publish**.

Every action shows its error in place with the log one click away, and every step reads its state from the project folder, so you can reload the page or come back tomorrow. The studio releases the GPU before any ACE-Step work, because an AI panel left loaded would starve the music model of VRAM.

## Daily use

```bash
.venv/bin/python run.py make
```

To start from your own story premise or fact topic on Windows PowerShell:

The default `auto` format infers story or fact from your idea. Add `--format story` or `--format fact` to choose explicitly.

```powershell
.\.venv\Scripts\python.exe run.py make --idea "A tiny lighthouse keeper finds a message inside a bottle" --format story
.\.venv\Scripts\python.exe run.py make --idea "Why do cats always seem to land on their feet?" --format fact
```

`--visuals` accepts `pixabay`, `illustration`, or `placeholder` and overrides `config.yaml`.

### How stock footage is chosen

With `pixabay` visuals the pipeline no longer searches one phrase per scene:

1. One LLM call writes three concrete search phrasings per scene (`search_terms` in `script.json`), keeping the scene's era so a historical scene never pulls modern offices or jets.
2. Each phrasing is searched on Pixabay for videos and photos; photos get a slow zoom in the final video. Results are merged and de-duplicated into eight candidates per scene.
3. A vision model (Gemini Flash by default, free tier) scores every candidate frame 0 to 10 against the scene and sorts them. The top candidate is used automatically unless you pick another in the browser UI, which shows each score and a one-line reason. A best score under 4 means the library simply lacks the subject; upload your own image or switch that video to `illustration`.

### AI panels drawn on your own GPU (free)

Stock libraries have nothing for mythology, history, or invented characters. The `diffusion` source renders one panel per scene locally with Stable Diffusion 1.5 (DreamShaper 8), using the script's visual description, setting, props, and recurring character. It needs the optional packages once:

```powershell
.\.venv\Scripts\pip install -r requirements-diffusion.txt --extra-index-url https://download.pytorch.org/whl/cu126
```

The first run downloads about 2 GB of weights into `cache/diffusion/`. On a 4 GB card such as a GTX 1050 Ti a panel takes well under a minute in the default fast mode (LCM, 8 steps); set `diffusion.fast: false` for 25-step quality at a few minutes per panel. Without a CUDA GPU it falls back to the CPU, which is slow but works.

Use it two ways:

- `--visuals diffusion` (or `visuals.source: diffusion`) draws every scene. Regenerate in the UI renders another take with a new seed.
- Keep `pixabay` and set `visuals.diffusion_fallback_score` (default 4): scenes whose best stock match scores below it get an AI panel ranked first, so real footage is used where it exists and drawn panels fill the gaps.

Every panel's prompt and seed are saved next to it in `illustrations/`, so you can tweak `diffusion.style` in `config.yaml` and rerender. The checkpoint is licensed CreativeML OpenRAIL-M, which permits commercial use.

Ranking packs three scenes per request, so a six-scene short costs two vision requests. Gemini 3.8 Flash's free tier allows about 20 requests per day; when that runs out the pipeline switches to the fallback model, and if no vision key works it keeps the search order. Tune this under `visuals:` in `config.yaml` (`search_terms_per_scene`, `include_photos`, `rank_with_llm`, `rank_batch_scenes`) and with `VISION_PROVIDER` in `.env`.
`--stop-after` accepts `script`, `voice`, `media_selection`, `visuals`, or `video`.

To generate and review only the script before rendering:

```powershell
.\.venv\Scripts\python.exe run.py make --idea "Your rough idea goes here" --stop-after script
```

After editing and fact-checking script.md or script.json, continue the video using the slug printed by the command:

```powershell
.\.venv\Scripts\python.exe run.py make --slug <slug>
```

To regenerate narration for an existing script with a particular narrator:

```powershell
.\.venv\Scripts\python.exe run.py make --slug <slug> --voice kokoro --voice-name bm_fable --stop-after voice
```

### Narrators rotate so the channel does not sound templated

Thirty shorts read by one voice look mass-produced. Each new video therefore gets the least recently used narrator from a pool, chosen by content type (story or fact), and the choice is stored in `script.json` so re-runs keep it. Two free engines are built in:

- `kokoro` (default): 28 English voices run locally; the pools use the ones the model card grades C or better. The model also reports when each phoneme is spoken, so captions follow the real word timing.
- `gemini`: Gemini 3.8 Flash TTS on the Gemini free tier, 30 prebuilt voices, with `gemini_style` in `config.yaml` as a delivery hint.

### Thirty a month: batch, queue, schedule

```powershell
.\.venv\Scripts\python.exe run.py batch --count 5        # five fresh videos, left in output/ for review
.\.venv\Scripts\python.exe run.py queue                   # finished, unpublished videos and the slot each would get
.\.venv\Scripts\python.exe run.py queue --publish         # upload them, each going public at its slot
```

`batch` asks the model for distinct topics in one call, drops anything close to a past title (shared keywords), alternates story and fact, gives every video a different narrator, and with AI panels rotates the art style from `diffusion.styles`. One failing video is logged and skipped; the summary at the end lists what is ready. The browser UI has the same under "Batch" in the sidebar, running in the background with its log shown.

Publishing is still your call. `run.py publish <slug> --next-slot` or `--at "2026-11-06 18:00"` uploads the video as private with a YouTube `publishAt`, so it goes public by itself at that time; slots come from `youtube.schedule` in `config.yaml` (daily times, timezone, lead time) and ones already promised are skipped. `run.py list` shows what is unfinished, ready, published, or scheduled.

Every upload sets YouTube's altered-or-synthetic-content flag (`youtube.contains_synthetic_media`) and appends `youtube.disclosure` to the description, because the narration and any AI panels are generated. Keep that honest: YouTube's monetization rules penalise mass-produced, templated uploads more than disclosed AI use, which is why the review step stays in the loop and why narrators, topics, and art styles rotate.

### Four kinds of video

The same pipeline makes four formats; `formats:` in `config.yaml` holds what differs between them (resolution, caption size, script length, ranking budget) and the chosen format is written into each project's `script.json`.

| Format | Command | What you get |
|---|---|---|
| `short` (default) | `run.py make` | vertical 9:16, under a minute, up to 6 scenes |
| `long` | `run.py make --length long` | 16:9, about ten minutes, six chapters with timestamps in the description, mid-roll eligible |
| `doodle` | `run.py make --length doodle --idea "..."` | 16:9, 9 to 11 minutes, a hand-drawn cartoon for every sentence with one recurring character, drawn on Kaggle's free GPUs |
| `lofi` | `run.py lofi --idea "..."` | 16:9, an hour of generated music over one gently animated illustrated scene, tracklist in the description |

The UI puts the three side by side: the sidebar's Create panel has Short, Long video and Lofi music, each with its own form, and a finished long video has a "Cut Shorts" button. Lofi is done in stages so nothing expensive runs before you have seen it: pick a preset or describe the scene and press Plan (title, description, tags and track names, a few seconds); on the project page draw the scene (and ask for another take), listen to the first track, then render the full video in the background while the page follows the log; when it is ready the same page shows the video, the tracklist and the publish controls, and "Change and render again" reuses the tracks and loop already made.

**Long videos** are written the way a writer would: one request for the outline (title, hook, chapters with their beats, closing), then one request per chapter that turns its beats into six narrated scenes while seeing where the previous chapter stopped. Expect 1,300 to 1,600 words over about 36 scenes; a chapter that comes back thin is sent back once with the shortfall spelled out. Ten minutes of narration needs a model that knows the subject: the small Nemotron model that is fine for Shorts wrote a vague 860-word explainer in testing, so long and doodle scripts go to Gemini whenever `GEMINI_API_KEY` is set, whatever `LLM_PROVIDER` says (about seven requests per video, within the free tier's daily allowance). Set `LONG_SCRIPT_PROVIDER` in `.env` (`gemini`, `anthropic` or `nemotron`) to choose another. Chapter start times are computed from the timed narration and go into the description as `0:00 Title` lines, which YouTube turns into chapters. Stock ranking is capped per video (`formats.long.visuals.rank_max_requests`, six scenes per request) so a long video spends about six vision requests instead of twelve. Everything else (narrators, Whisper captions, camera moves on stills, scheduling) is the same as for Shorts; AI panels are generated in landscape.

**Doodle explainers** are the calm, well-sourced "what did ancient humans do all day" kind of video: a narrator, ten minutes of explanation, and a new simple cartoon on a white page every few seconds, with one recurring character and the odd handwritten title card.

```powershell
.\.venv\Scripts\python.exe run.py make --length doodle --format fact --idea "What did ancient humans actually do all day?"
.\.venv\Scripts\python.exe run.py make --length doodle --dry-run    # no keys: sample script, placeholder drawings
```

1. **Script (Gemini).** The same outline-then-chapters writer as long videos, but every scene comes back as *beats*: one sentence of narration each, with the picture to draw while it is spoken, whether the recurring character is in it, an optional handwritten label ("17 HOURS A WEEK"), and, about once a chapter, a full-screen title card ("THE WORKLOAD EXPLODES"). A ten-minute script has about 150 to 200 beats. Edit `script.json` by hand if you like: when a scene's narration no longer matches its beats, the beats are rebuilt from the narration and keep their pictures in order.
2. **Narration**, as for any video, then re-timed evenly (never more than 12%, `doodle.max_speed_change`) so the video lands inside `doodle.fit_minutes` (9 to 11). If even that is not enough, the UI says the script is too short or too long.
3. **Drawings on Kaggle.** The app writes a small Python script with every prompt and seed embedded, pushes it to your Kaggle account as one private notebook (`kaggle.kernel_slug`, reused for every video as a new version), and runs it on Kaggle's free T4 GPUs. Kaggle draws the character once, then every beat with SDXL plus the SDXL-Lightning 8-step LoRA (a few seconds a picture, both GPUs in parallel), with IP-Adapter holding the character's look. The app polls, downloads one zip, checks the job id, and unpacks the drawings into `doodle/raw/`. Expect a few minutes of queueing and model download, then roughly 10 to 20 minutes for a whole video, well inside the weekly free GPU quota. If the studio is closed meanwhile, pressing Draw again picks up the same Kaggle run instead of starting a new one. Every model involved allows commercial use of its output: SDXL base and SDXL-Lightning (CreativeML OpenRAIL++-M), the fp16-fix VAE (MIT), IP-Adapter (Apache 2.0).
4. **Review.** The Drawings step shows every picture by chapter; tick the off-model or wrong ones and "Redraw marked" sends only those back with new seeds.
5. **Render.** Title cards and labels are drawn here in a handwriting font (Ink Free on Windows; set `doodle.font_path` or drop a `.ttf` into `assets/fonts/`). Each drawing stays on screen exactly while its sentence is spoken (word timings from the narrator) with a slow drift, and no burned captions by default (YouTube's own captions work). Shorts cut from a doodle video keep a drawing per sentence, set on a white vertical page.

No Kaggle account? Choose "This PC's GPU" (Stable Diffusion 1.5, about 45 seconds a picture on a GTX 1050 Ti, so a few hours per video) or "Placeholders" to try the whole flow.

**Shorts cut from a long video** reuse its narration, its chosen media and its ranked stills, so they cost no API calls beyond one request that picks the excerpts:

```powershell
.\.venv\Scripts\python.exe run.py cut 2026-11-05-your-long-video --count 2
```

The model picks consecutive runs of scenes that stand alone (the hook first, then a chapter with a payoff), between `cuts.min_seconds` and `cuts.max_seconds`; if it cannot, chapter starts are used. Each cut is rendered vertically (landscape footage is centre-cropped), gets Shorts-sized captions, its own title and thumbnail, and a description that links to the full video once that is published. They appear as `<slug>-short-1`, `-short-2` in `output/` and in the UI, and `queue --publish` schedules them like any other video.

### Lofi music videos

The quick way, and the default: everything comes from Pixabay, which is free for commercial use under its Content License.

- **The picture** is an animated loop from Pixabay's video API: searching `video_type=animation` for the scene you describe returns the genre's staples (a girl studying by a lamp, a cat asleep by a rainy window), usually 60 seconds at 1080p. The clip is made seamless by cross-fading its tail into its own head, then repeated by stream copy, so an hour of picture takes seconds. "Another one" in the studio moves to the next candidate; the ones used before sort last.
- **The music** comes from a folder you fill: Pixabay serves music only through its website and forbids automated downloads, so search https://pixabay.com/music/search/lofi/ , download the tracks you like into `assets/lofi_music/` (mp3 is fine; keep the licence certificate the site offers next to the download), and the pipeline strings the least-used ones together with crossfades until the video is full. Titles come from the file names. Pixabay music is allowed in monetised videos, but some tracks are registered in YouTube's Content ID; a claim is cleared by disputing it with that certificate.
- **Rain** likewise comes from `assets/ambience/` (https://pixabay.com/sound-effects/search/rain/ ), looped quietly under the music when the scene is rainy; with the folder empty the pipeline synthesizes rain instead. `lofi.ambience_volume` keeps it under the music.
- **No repeats.** `lofi_usage.json` records which clip, tracks and rain file each video used, and every pick prefers what has been used least, so two videos never share a loop or a tracklist while there is unused material. The studio's Music step shows how many unused minutes the library holds and warns when a video would have to reuse tracks.

The description gets the tracklist with timestamps plus credits for the animation and the music. The slower options remain for when you want original material: `lofi.picture: diffusion` draws the scene on your GPU, and `music.engine: acestep` or `procedural` generates the music.


```powershell
.\.venv\Scripts\python.exe run.py lofi --idea "a student at a desk by a rainy window at night" --minutes 60
```

**Drawing the scene yourself** (`lofi.picture: diffusion` or `illustration`): one illustrated scene (an AI panel in the lofi style from `lofi.style`, or `--visuals illustration` without a GPU) becomes a seamless 24-second loop: the camera breathes in and out, rain falls past the window when the scene is rainy, the lamp light flickers, film grain moves. The hour-long picture is that loop repeated by stream copy, so the render takes seconds. The music is a tracklist generated on this machine, joined with crossfades, with soft rain underneath when the scene calls for it. The description carries the tracklist with timestamps; the upload uses the Music category and a disclosure that the music and artwork were generated.

Two music engines, both free for commercial use:

- `procedural`: a composer and synthesizer in `pipeline/lofi_music.py`. It picks a key, a tempo between 68 and 88, a jazz chord loop, and voices an electric piano, bass, swung drums with ghost notes and a sparse pentatonic melody, then applies the lofi treatment (low-pass, tape wobble, vinyl crackle, side-chain ducking). No model, no download, nothing to license. About thirty seconds per track on a CPU.
- `kaggle`: the same ACE-Step 1.5 model, run on Kaggle's free T4 GPU instead of yours. One job generates the whole tracklist (`pipeline/kaggle_kernel/acestep_music.py` is pushed as a private notebook, like the doodle drawings): a few minutes to install ACE-Step and download its weights, then roughly a minute per 2.5-minute track, and the mp3s are downloaded into the project's music folder. Needs `KAGGLE_API_TOKEN` and a phone-verified Kaggle account. Measured on 2026-11-08: Kaggle's image runs Python 3.13, so the notebook builds a Python 3.12 environment with uv and installs ACE-Step there (about three minutes), then generates a one-minute track in about seven seconds on the T4; a two-track probe took four and a half minutes from push to downloaded mp3s. Only the zip of tracks comes back; the environment and the weights stay in Kaggle's `/tmp`. On any failure the composer takes over, and the Kaggle log tail is written into the render log.
- `acestep`: [ACE-Step 1.5](https://github.com/ACE-Step/ACE-Step-1.5), an MIT-licensed open model whose output may be used commercially, on this PC's GPU. It is a separate install in its own environment and this pipeline talks to its local API (`music.acestep_url`); when the server is not running or a track fails the pipeline composes that track procedurally instead (`music.fallback_to_procedural`). It is the default (`music.engine: acestep`), so start its server before a render or the whole tracklist is composed procedurally; `--music procedural` or the UI's Music picker chooses the composer for one video.

  Install it outside this project, for example in `D:\Personal Projects\ACE-Step-1.5`:

  ```powershell
  git clone https://github.com/ACE-Step/ACE-Step-1.5
  cd ACE-Step-1.5
  uv sync --python 3.12
  ```

  Then start the API server. On this machine (GTX 1050 Ti, 4 GB, Pascal) it needs the settings ACE-Step's documentation recommends for cards under 6 GB: the 2B turbo DiT, no language model, INT8 quantization and full CPU offload. The API server reads them from environment variables:

  ```powershell
  $env:ACESTEP_INIT_LLM = "false"; $env:ACESTEP_CONFIG_PATH = "acestep-v15-turbo"
  $env:ACESTEP_OFFLOAD_TO_CPU = "true"; $env:ACESTEP_OFFLOAD_DIT_TO_CPU = "true"
  $env:ACESTEP_QUANTIZATION = "w8a8_dynamic"; $env:ACESTEP_COMPILE_MODEL = "false"
  $env:ACESTEP_GENERATION_TIMEOUT = "1500"   # its default of 600 s is too short for a 2.5-minute track here
  .\.venv\Scripts\acestep-api.exe --host 127.0.0.1 --port 8001   # run from the ACE-Step folder; no uv needed
  ```

  Measured here on 2026-11-04 (ACE-Step commit ca1e85f, torch 2.7.1+cu128, which still ships Pascal kernels, so the launcher's legacy-torch swap was not needed):

  - It runs. The first request downloads about 10 GB of weights into `ACE-Step-1.5/checkpoints` and then loads and quantizes the DiT, which takes about two minutes; later requests skip that.
  - fp16 works on Pascal. ACE-Step detects a pre-Ampere card, switches from bf16 to float16 and from fused to eager attention; the output has no NaN and normal levels.
  - Peak use of the whole card was 3.7 GB; torch itself reported a maximum of 3.1 GB allocated for the 30-second test and 3.5 GB for a 2.5-minute track, so a 2.5-minute track is near the ceiling and longer `music.track_minutes` will probably not fit. Close other GPU users first; another lofi run drawing with Stable Diffusion at the same time left only 100 MB free and the request would have failed.
  - Time: a 30-second test track took 109 s; a 2.5-minute track (the default `music.track_minutes`) took 7.5 and 9 minutes in two runs at 8 steps, almost all of it diffusion (the quantized DiT is copied to the GPU and back for each track, which took between 2 s and 15 s). A 5-minute video is therefore a 25-minute job and an hour of music about four hours on this card; `music.acestep_timeout_minutes` (20) is the per-track ceiling on this side, and ACE-Step's own `ACESTEP_GENERATION_TIMEOUT` (600 s by default) must be raised or a full-length track is abandoned at ten minutes and the pipeline composes that track procedurally instead.
  - Two small changes to the ACE-Step checkout were needed, both left as uncommitted edits there (`git diff` in that folder shows them): the stock `acestep-api` never passes a quantization option to the model loader (only the Gradio launcher does), so `acestep/api/startup_model_init.py` now reads `ACESTEP_QUANTIZATION`; and its CPU-offload mover only recognised `AffineQuantizedTensor`, while `w8a8_dynamic` wraps weights in `LinearActivationQuantizedTensor`, which on torch 2.7 cannot be moved by assigning `param.data`, so `acestep/core/generation/handler/init_service_memory_basic.py` treats any torchao wrapper tensor as quantized. Without the second change every generation failed with `NotImplementedError ... _has_compatible_shallow_copy_type`, with torchao 0.16 and 0.11 alike (0.11.0 is what is installed now, since 0.16 also warns it is incompatible with torch 2.7.1).

  The procedural engine stays the default: it is instant and needs nothing, while ACE-Step on this card is slow enough that it is a deliberate choice per video.

Lofi channels are judged on mood and on the picture, so expect to iterate on the scene idea and on `lofi.style`, and read YouTube's rules on repetitive content before uploading many of them: a tracklist of distinct generated tracks, a hand-chosen scene and an honest title keep them on the right side of it.

### Stills move, and scenes cut

A photo or AI panel is never held still. Each scene is cut into shots of about three seconds (`video.seconds_per_still`), and every shot is a different camera move: a push in, a pull back, or a pan, each framed on a different part of the picture. When a scene has more than one usable still (its AI panels, or stock photos the ranker scored 6 or higher) the shots alternate between them; `visuals.stills_per_scene` caps how many. The moves are tuned under `motion:` in `config.yaml`. Videos keep their own motion and cut every `video.seconds_per_cut`.

In the media editor each scene has two buttons: "More stock" (or "New take" for AI panels) fetches another page of candidates, and "AI panel" draws the scene on your GPU and adds the panel to that scene's candidates without changing the rest of the project. That is the fix for a scene where nothing stock fits.

### Captions follow the speech

Captions show a few words at a time, timed to the narration. Kokoro reports when each phoneme is spoken. For voices that only return audio (Gemini TTS, ElevenLabs without alignment data, the system voice) the pipeline runs a local Whisper pass (`faster-whisper`, `small.en`, CPU, a few seconds per scene, downloaded once into `cache/asr/`) with the script as the decoding prompt, then maps the recognised words back onto the script's own words so spelling and punctuation stay yours. If Whisper hears something else entirely, timing falls back to an even spread. Tune `captions.align*` and `hold_seconds` in `config.yaml`.

`python run.py voices` lists every narrator, its pool, and when it was last used. Set `voice.rotate: false` to always use `kokoro_voice` or `gemini_voice`, edit `voice.pools` to curate the rotation, or pass `--voice-name` for one video. In the browser UI the sidebar has engine and narrator pickers with a play-a-sample button, and each project's script panel has a Narrator section that re-voices every scene.

That runs everything and prints a review checklist. Output lands in `output/<date>-<slug>/`:

```
script.md        human-readable script with the fact-check list
script.json      what the pipeline actually uses
audio/           one mp3 per scene
illustrations/   locally drawn scene panels (illustration visuals only)
final.mp4        the video
thumbnail.jpg    the thumbnail
clip_plan.json   the source visual selected for each scene
```

Pixabay clips are cached in `cache/clips/`; media candidates and your per-scene choices are stored in `media_candidates.json` and `media_plan.json` next to the script.

Recommended workflow while you are tuning the prompts and checking generated facts:

```bash
.venv/bin/python run.py make --stop-after script     # writes the script and stops
# edit output/<slug>/script.json (narration, title, thumbnail_text)
.venv/bin/python run.py make --slug <slug>           # continues from the edited script
```

Then watch `final.mp4` and publish:

```bash
.venv/bin/python run.py publish <slug>
```

Uploads are `private` by default so you can also check it on YouTube before flipping to public.

Other commands:

```bash
.venv/bin/python run.py ideas            # 10 topic ideas, no video
.venv/bin/python run.py list             # everything made so far
.venv/bin/python run.py make --dry-run   # test the whole chain with no API keys at all
```

## Running it on a schedule

Have a draft waiting every morning.

### Windows (Task Scheduler)

Run once in PowerShell from the project folder, which registers a 6:00 daily task:

```powershell
schtasks /Create /TN "YouTubeAutopilot" /SC DAILY /ST 06:00 /TR "cmd /c cd /d \"$PWD\" && .venv\Scripts\python run.py make >> autopilot.log 2>&1"
```

Remove it with `schtasks /Delete /TN "YouTubeAutopilot"`. The laptop has to be awake at that time; set the task to "wake the computer" in Task Scheduler's GUI if needed.

### macOS (launchd)

Create `~/Library/LaunchAgents/com.ytautopilot.plist`:

```xml
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0"><dict>
  <key>Label</key><string>com.ytautopilot</string>
  <key>ProgramArguments</key><array>
    <string>/FULL/PATH/TO/yt-autopilot/.venv/bin/python</string>
    <string>/FULL/PATH/TO/yt-autopilot/run.py</string>
    <string>make</string>
  </array>
  <key>WorkingDirectory</key><string>/FULL/PATH/TO/yt-autopilot</string>
  <key>StartCalendarInterval</key><dict>
    <key>Hour</key><integer>6</integer><key>Minute</key><integer>0</integer>
  </dict>
  <key>StandardOutPath</key><string>/FULL/PATH/TO/yt-autopilot/autopilot.log</string>
  <key>StandardErrorPath</key><string>/FULL/PATH/TO/yt-autopilot/autopilot.log</string>
</dict></plist>
```

Then `launchctl load ~/Library/LaunchAgents/com.ytautopilot.plist`.

Each morning a new folder is waiting in `output/`; you review and publish the ones you like.

## Cost per video (approximate)

| Item | Cost |
|---|---|
| Gemini 3.8 Flash (ideas + script) | Free within the current free-tier quota; limits can change |
| Kokoro or Gemini TTS (voice) | Free: Kokoro runs locally (model files download once); Gemini TTS is on the free tier |
| ElevenLabs (about 1 minute of speech) | optional; check the current plan limits |
| Motion-comic illustrations | Free; drawn locally with Pillow |
| Pixabay (optional) | free |
| YouTube API | free |

## Tuning quality

- Script voice and structure: the `system` prompt in `pipeline/script.py`. Edit it after your first few videos to lock in what works.
- Pacing: `seconds_per_cut` in `config.yaml`. Lower is punchier.
- Caption look: the `Style:` line in `pipeline/captions.py`.
- Thumbnail: `pipeline/thumbnail.py`. Thumbnails matter more than anything else here; replace the generated one by hand whenever you can.
- Shorts: built in by default. Output is vertical 1080x1920 and targets about 45 to 55 seconds. YouTube classifies square or vertical videos up to 3 minutes as Shorts; see [YouTube's current Shorts guidance](https://support.google.com/youtube/answer/15424877).

## Known limits

- Stock footage may not match a scene exactly, which is why the browser UI lets you choose the clip per scene. The optional motion-comic source creates original flat panels locally with a slow camera zoom; it is not frame-by-frame animation or AI-generated artwork, so unusual details may need manual adjustment.
- ElevenLabs limits a single request to a few thousand characters, which is why audio is generated per scene.
- Kokoro model weights use Apache 2.0 and the ONNX wrapper uses MIT. Review the [Kokoro model card](https://huggingface.co/hexgrad/Kokoro-82M) and its voice details before commercial use. Kokoro word timing comes from the model's phoneme timings; other voices are timed with a local Whisper pass.
- The `local` voice (Windows SAPI or macOS `say`) is for testing only.
