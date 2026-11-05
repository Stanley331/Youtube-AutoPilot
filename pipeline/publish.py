"""Stage 7 (manual trigger only): upload to YouTube with the Data API.

Setup, once:
  1. console.cloud.google.com -> new project -> enable "YouTube Data API v3"
  2. OAuth consent screen (External, add yourself as a test user)
  3. Credentials -> OAuth client ID -> Desktop app -> download JSON -> save as client_secrets.json here
First upload opens a browser to authorize; the token is cached in token.json.
"""
from pathlib import Path

from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload

from .config import ROOT
from .util import log

SCOPES = ["https://www.googleapis.com/auth/youtube.upload",
          "https://www.googleapis.com/auth/youtube"]


def _service():
    token = ROOT / "token.json"
    secrets = ROOT / "client_secrets.json"
    creds = Credentials.from_authorized_user_file(str(token), SCOPES) if token.exists() else None
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            if not secrets.exists():
                raise RuntimeError("client_secrets.json is missing; see pipeline/publish.py for setup")
            creds = InstalledAppFlow.from_client_secrets_file(str(secrets), SCOPES).run_local_server(port=0)
        token.write_text(creds.to_json())
    return build("youtube", "v3", credentials=creds)


def description_timestamps(script: dict) -> list[str]:
    """Chapter lines for a long video, the tracklist for a lofi video; YouTube reads both as chapters."""
    if script.get("tracklist"):
        lines = ["Tracklist:"] + [str(line) for line in script["tracklist"]]
        if script.get("credits"):
            lines += [""] + [str(c) for c in script["credits"]]
        return lines
    from .longform import chapter_lines
    lines = chapter_lines(script)
    return ["Chapters:"] + lines if lines else []


def build_body(script: dict, cfg: dict, privacy: str | None = None, publish_at: str | None = None) -> dict:
    """The videos.insert payload: metadata, privacy, optional schedule, AI disclosure."""
    yt = cfg["youtube"]
    description = str(script.get("description", "")).strip()
    timestamps = description_timestamps(script)
    if timestamps and timestamps[0] not in description:
        description = f"{description}\n\n" + "\n".join(timestamps) if description else "\n".join(timestamps)
    disclosure = str(yt.get("disclosure", "") or "").strip()
    if disclosure and disclosure not in description:
        description = f"{description}\n\n{disclosure}" if description else disclosure
    status = {
        "privacyStatus": privacy or yt["privacy"],
        "selfDeclaredMadeForKids": bool(yt.get("made_for_kids", False)),
    }
    if yt.get("contains_synthetic_media") is not None:
        # YouTube's altered-or-synthetic-content disclosure; the voice and panels are generated.
        status["containsSyntheticMedia"] = bool(yt["contains_synthetic_media"])
    if publish_at:
        # YouTube only honours a schedule on a private upload; it flips to public itself.
        status["privacyStatus"] = "private"
        status["publishAt"] = publish_at
    return {
        "snippet": {
            "title": script["title"][:100],
            "description": description[:5000],
            "tags": list(script.get("tags", []))[:30],
            "categoryId": str(script.get("category_id") or yt["category_id"]),
        },
        "status": status,
    }


def upload(video: Path, thumbnail: Path, script: dict, cfg: dict, privacy: str | None = None,
           publish_at: str | None = None) -> str:
    service = _service()
    body = build_body(script, cfg, privacy, publish_at)
    media = MediaFileUpload(str(video), chunksize=8 * 1024 * 1024, resumable=True)
    request = service.videos().insert(part="snippet,status", body=body, media_body=media)
    response = None
    while response is None:
        status, response = request.next_chunk()
        if status:
            log(f"upload: {int(status.progress() * 100)}%")
    video_id = response["id"]
    if thumbnail.exists():
        service.thumbnails().set(videoId=video_id, media_body=str(thumbnail)).execute()
    return video_id
