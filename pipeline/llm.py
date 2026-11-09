"""Provider-neutral helper for generating schema-constrained JSON."""
import base64
import json
import re
import time
from urllib.parse import quote

import requests

from .config import env
from .util import log

GEMINI_API_URL = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
DEFAULT_GEMINI_MODEL = "gemini-3.8-flash"
DEFAULT_GEMINI_FALLBACK_MODEL = "gemini-3.6-flash"
DEFAULT_ANTHROPIC_MODEL = "claude-opus-5-5"
DEFAULT_NEMOTRON_MODEL = "nvidia/nemotron-3.5-bolt-e4b"
DEFAULT_NEMOTRON_BASE_URL = "https://integrate.api.nvidia.com/v1/chat/completions"
TRANSIENT_GEMINI_STATUSES = {500, 502, 503, 504}
# Quota exhausted on one model: move straight to the fallback model, which has
# its own free-tier allowance, instead of retrying the same one.
SWITCH_MODEL_GEMINI_STATUSES = {429}
DEFAULT_TIMEOUT = 180
GEMINI_BENCH_SECONDS = 10 * 60
_GEMINI_BENCHED: dict[str, float] = {}   # model -> time until which it is skipped
TRANSIENT_NEMOTRON_STATUSES = {429, 500, 502, 503, 504}


def generate_json(system: str, prompt: str, schema: dict,
                  max_tokens: int = 16000, effort: str = "high",
                  images: list[bytes] | None = None,
                  timeout: int = DEFAULT_TIMEOUT,
                  provider: str | None = None) -> dict:
    """Generate one JSON object using the provider selected in .env.

    ``images`` are JPEG bytes attached to the prompt in order (for ranking
    footage candidates). They go to VISION_PROVIDER when set, otherwise to
    Gemini when a key exists (its Flash models see images on the free tier),
    otherwise to the text provider. ``provider`` overrides LLM_PROVIDER for
    one call (long scripts can use a stronger model than the cheap daily one).
    """
    provider = (provider or env("LLM_PROVIDER", "gemini") or "gemini").lower()
    if images:
        provider = vision_provider()
    if provider == "gemini":
        return _generate_gemini_json(system, prompt, schema, max_tokens, effort, images, timeout)
    if provider == "anthropic":
        return _generate_anthropic_json(system, prompt, schema, max_tokens, effort, images, timeout)
    if provider == "nemotron":
        return _generate_nemotron_json(system, prompt, schema, max_tokens, effort, images, timeout)
    raise ValueError("LLM_PROVIDER must be 'gemini', 'anthropic', or 'nemotron'")


def vision_provider() -> str:
    """Provider used for prompts that include images."""
    explicit = (env("VISION_PROVIDER") or "").lower()
    if explicit:
        return explicit
    if env("GEMINI_API_KEY"):
        return "gemini"
    return (env("LLM_PROVIDER", "gemini") or "gemini").lower()


def vision_available() -> bool:
    provider = vision_provider()
    keys = {"gemini": "GEMINI_API_KEY", "anthropic": "ANTHROPIC_API_KEY", "nemotron": "NEMOTRON_API_KEY"}
    return bool(env(keys.get(provider, "")))


def _b64(data: bytes) -> str:
    return base64.b64encode(data).decode("ascii")


def _extract_json_from_text(text: str) -> dict:
    """Pull a JSON object out of a model response, handling markdown fences."""
    text = text.strip()
    # Try direct parse first
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    # Look for ```json ... ``` or ``` ... ``` fences
    fence = re.search(r"```(?:json)?\s*\n(.*?)\n```", text, re.DOTALL)
    if fence:
        try:
            return json.loads(fence.group(1).strip())
        except json.JSONDecodeError:
            pass
    # Fall back to finding the first { ... } block
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        try:
            return json.loads(text[start:end + 1])
        except json.JSONDecodeError:
            pass
    raise json.JSONDecodeError("Could not extract JSON from model response", text, 0)


def _generate_nemotron_json(system: str, prompt: str, schema: dict,
                            max_tokens: int, effort: str,
                            images: list[bytes] | None = None,
                            timeout: int = DEFAULT_TIMEOUT) -> dict:
    """Generate schema-constrained JSON using NVIDIA Nemotron (OpenAI-compatible)."""
    key = env("NEMOTRON_API_KEY")
    if not key:
        raise RuntimeError(
            "NEMOTRON_API_KEY is missing. Get a free key from build.nvidia.com "
            "and add it to .env, or switch LLM_PROVIDER back to gemini."
        )
    model = env("NEMOTRON_MODEL", DEFAULT_NEMOTRON_MODEL) or DEFAULT_NEMOTRON_MODEL
    base_url = env("NEMOTRON_BASE_URL", DEFAULT_NEMOTRON_BASE_URL) or DEFAULT_NEMOTRON_BASE_URL
    temperature = 0.2 if effort == "high" else 0.4

    # Nemotron may not support native response_format JSON schema, so we ask
    # for JSON in the prompt and parse it back out.
    schema_prompt = (
        prompt
        + "\n\nRespond with valid JSON only (no markdown, no explanation). "
        "Match this JSON schema exactly:\n"
        + json.dumps(schema, indent=2)
    )

    if images:
        user_content: list | str = [{"type": "text", "text": schema_prompt}] + [
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{_b64(img)}"}}
            for img in images
        ]
    else:
        user_content = schema_prompt
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user_content},
        ],
        "max_tokens": max_tokens,
        "temperature": temperature,
        "stream": False,
    }

    headers = {
        "Authorization": f"Bearer {key}",
        "Content-Type": "application/json",
        "Accept": "application/json",
    }

    resp = None
    last_error = None
    for attempt, delay in enumerate((0, 2, 5)):
        if delay:
            log(f"Nemotron temporarily unavailable; retrying in {delay}s")
            time.sleep(delay)
        try:
            resp = requests.post(
                base_url,
                headers=headers,
                json=payload,
                timeout=timeout,
            )
        except (requests.Timeout, requests.ConnectionError) as exc:
            last_error = exc
            resp = None
            log(f"Nemotron request failed ({type(exc).__name__})")
            continue
        except requests.RequestException as exc:
            raise RuntimeError(
                f"Could not reach the Nemotron API ({type(exc).__name__})"
            ) from exc

        last_error = None
        if resp.ok or resp.status_code not in TRANSIENT_NEMOTRON_STATUSES:
            break

    if resp is None:
        raise RuntimeError(
            f"Could not reach the Nemotron API ({type(last_error).__name__}). "
            "Check your internet connection and NEMOTRON_BASE_URL."
        ) from last_error

    if not resp.ok:
        detail = (resp.text or "").replace(key, "[redacted]")
        detail = " ".join(detail.split())[:500]
        raise RuntimeError(
            f"Nemotron model {model} returned HTTP {resp.status_code}: {detail}"
        )

    try:
        data = resp.json()
    except ValueError as exc:
        raise RuntimeError("Nemotron returned a response that was not valid JSON") from exc

    try:
        text = data["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as exc:
        raise RuntimeError(f"Nemotron response had unexpected shape: {data}") from exc

    if not text:
        raise RuntimeError("Nemotron returned no text")

    try:
        return _extract_json_from_text(text)
    except json.JSONDecodeError as exc:
        raise RuntimeError("Nemotron response did not match the expected JSON format") from exc


def _generate_gemini_json(system: str, prompt: str, schema: dict,
                          max_tokens: int, effort: str,
                          images: list[bytes] | None = None,
                          timeout: int = DEFAULT_TIMEOUT) -> dict:
    key = env("GEMINI_API_KEY")
    if not key:
        raise RuntimeError(
            "GEMINI_API_KEY is missing. Create a key in Google AI Studio and add it to .env."
        )
    model = env("GEMINI_MODEL", DEFAULT_GEMINI_MODEL) or DEFAULT_GEMINI_MODEL
    fallback = env("GEMINI_FALLBACK_MODEL", DEFAULT_GEMINI_FALLBACK_MODEL)
    models = [model]
    if fallback and fallback != model:
        models.append(fallback)
    thinking_level = "HIGH" if effort == "high" else "MEDIUM"
    payload = {
        "systemInstruction": {"parts": [{"text": system}]},
        "contents": [{"role": "user", "parts": [{"text": prompt}] + [
            {"inline_data": {"mime_type": "image/jpeg", "data": _b64(img)}}
            for img in (images or [])
        ]}],
        "generationConfig": {
            "maxOutputTokens": max_tokens,
            "thinkingConfig": {"thinkingLevel": thinking_level},
            "responseFormat": {
                "text": {
                    "mimeType": "APPLICATION_JSON",
                    "schema": schema,
                }
            },
        },
    }

    # A model that just failed (quota, overload, timeout) is skipped for a while: a ten-minute
    # script makes seven calls, and retrying the dead model first cost minutes on every one.
    benched = [m for m in models if _GEMINI_BENCHED.get(m, 0) > time.time()]
    if benched and len(benched) < len(models):
        models = [m for m in models if m not in benched]
        log(f"Gemini {', '.join(benched)} failed a few minutes ago; using {models[0]} for now")

    response = None
    last_transport_error = None
    active_model = model
    for model_index, candidate_model in enumerate(models):
        if model_index:
            _GEMINI_BENCHED[models[model_index - 1]] = time.time() + GEMINI_BENCH_SECONDS
        active_model = candidate_model
        response = None
        url = GEMINI_API_URL.format(model=quote(candidate_model, safe="-._"))
        for attempt, delay in enumerate((0, 2, 5)):
            if delay:
                log(f"Gemini {candidate_model} is temporarily unavailable; retrying in {delay}s")
                time.sleep(delay)
            try:
                response = requests.post(
                    url,
                    headers={"x-goog-api-key": key},
                    json=payload,
                    timeout=timeout,
                )
            except (requests.Timeout, requests.ConnectionError) as exc:
                last_transport_error = exc
                log(f"Gemini {candidate_model} request failed ({type(exc).__name__})")
                break
            except requests.RequestException as exc:
                raise RuntimeError(
                    f"Could not reach the Gemini API ({type(exc).__name__})"
                ) from exc

            last_transport_error = None
            if response.ok or response.status_code not in TRANSIENT_GEMINI_STATUSES:
                break
        if response is None:
            if model_index + 1 < len(models):
                log(f"Gemini {candidate_model} could not be reached; switching to {models[model_index + 1]}")
                continue
            raise RuntimeError(
                f"Could not reach the Gemini API ({type(last_transport_error).__name__})"
            ) from last_transport_error
        if response.ok:
            if model_index:
                log(f"Gemini is using fallback model {candidate_model}")
            break
        if response.status_code in SWITCH_MODEL_GEMINI_STATUSES and model_index + 1 < len(models):
            log(f"Gemini {candidate_model} quota exhausted (HTTP 429); switching to {models[model_index + 1]}")
            continue
        if response.status_code not in TRANSIENT_GEMINI_STATUSES:
            break
        if model_index + 1 < len(models):
            log(f"Gemini {candidate_model} remains unavailable; switching to {models[model_index + 1]}")

    if not response.ok:
        detail = (response.text or "").replace(key, "[redacted]")
        detail = " ".join(detail.split())[:500]
        raise RuntimeError(
            f"Gemini model {active_model} returned HTTP {response.status_code}: {detail}"
        )

    try:
        data = response.json()
    except ValueError as exc:
        raise RuntimeError("Gemini returned a response that was not valid JSON") from exc

    candidates = data.get("candidates", [])
    if not candidates:
        reason = data.get("promptFeedback", {}).get("blockReason", "no candidate returned")
        raise RuntimeError(f"Gemini did not generate a script: {reason}")

    parts = candidates[0].get("content", {}).get("parts", [])
    text = "".join(part.get("text", "") for part in parts if isinstance(part, dict))
    if not text:
        finish = candidates[0].get("finishReason", "unknown")
        raise RuntimeError(f"Gemini returned no text (finish reason: {finish})")
    try:
        return _extract_json_from_text(text)
    except json.JSONDecodeError as exc:
        raise RuntimeError("Gemini response did not match the expected JSON format") from exc


def _generate_anthropic_json(system: str, prompt: str, schema: dict,
                             max_tokens: int, effort: str,
                             images: list[bytes] | None = None,
                             timeout: int = DEFAULT_TIMEOUT) -> dict:
    import anthropic

    model = env("ANTHROPIC_MODEL", DEFAULT_ANTHROPIC_MODEL) or DEFAULT_ANTHROPIC_MODEL
    client = anthropic.Anthropic(timeout=float(timeout))
    with client.messages.stream(
        model=model,
        max_tokens=max_tokens,
        system=system,
        messages=[{"role": "user", "content": _anthropic_content(prompt, images)}],
        output_config={
            "format": {"type": "json_schema", "schema": schema},
            "effort": effort,
        },
    ) as stream:
        message = stream.get_final_message()

    if message.stop_reason == "refusal":
        detail = getattr(message, "stop_details", None)
        raise RuntimeError(f"Claude declined this request: {detail}")
    if message.stop_reason == "max_tokens":
        raise RuntimeError("Claude's response was cut off; raise max_tokens")

    text = "".join(block.text for block in message.content if block.type == "text")
    if not text:
        raise RuntimeError(f"Claude returned no text (stop reason: {message.stop_reason})")
    try:
        return json.loads(text)
    except json.JSONDecodeError as exc:
        raise RuntimeError("Claude's response did not match the expected JSON format") from exc


def _anthropic_content(prompt: str, images: list[bytes] | None):
    if not images:
        return prompt
    blocks = [
        {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": _b64(img)}}
        for img in images
    ]
    return blocks + [{"type": "text", "text": prompt}]
