"""Put Chinese words on a YouTube cover, using the image API from the settings.

The English on a thumbnail has to become Chinese before it can be used as a
Bilibili cover. Two services are involved and they are deliberately separate:

* the text model (``openai.*`` settings, currently gpt-5.6-sol) translates the
  English line when you do not supply the Chinese yourself;
* the image model (``image.*`` settings, a gpt-image key) redraws the cover with
  the Chinese in place, keeping the layout, colours and picture intact.

The generated picture comes back at a 3:2 size, so it is cropped to 16:9 and
resized to 1280x720 - what the upload page wants.

Examples
--------
    # English -> Chinese handled by the text model
    python tools/cover_localize.py cover.jpg --en "HYPERSONIC ICE BALLS" -o cover.zh.jpg

    # supply the Chinese and skip the translation call
    python tools/cover_localize.py cover.jpg --en "HYPERSONIC ICE BALLS" --zh "超音速冰球"

    # start from a task session and use its original cover
    python tools/cover_localize.py --session workfolder/xxx/yyy__ID --en "..." -o cover.zh.jpg

Without ``--en`` the cover is left alone and the tool exits - detecting whether
a cover even has English text needs OCR, which is not wired in yet.
"""

from __future__ import annotations

import argparse
import base64
import subprocess
import sys
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backend.app import database  # noqa: E402
from backend.app.adapters.openai_client import normalize_openai_base_url  # noqa: E402
from backend.app.config import ffmpeg_binary  # noqa: E402
from tools.fetch_cover import fetch_cover, resolve_proxy, video_id_from_session  # noqa: E402

DEFAULT_BASE_URL = "https://www.aicost.me/v1"
DEFAULT_MODEL = "gpt-image-2"
# ``auto`` makes the model follow the input's aspect ratio. The covers are 16:9,
# so this returns 16:9 back and the crop below stays lossless - a fixed 3:2 size
# ends up cutting the title that sits at the very top of the picture.
DEFAULT_SIZE = "auto"

TRANSLATE_SYSTEM = (
    "你是封面文案翻译。把用户给的英文封面标题翻成中文，要求：保留原文的冲击力和口语感，"
    "不超过 12 个汉字，不要引号、不要解释、不要标点堆叠。只输出译文本身。"
)


def translate_title(english: str) -> str:
    """Turn the English cover line into Chinese with the configured text model."""
    from openai import OpenAI

    base_url = database.get_setting("openai.base_url") or "https://api.openai.com/v1"
    api_key = database.get_setting("openai.api_key")
    model = database.get_setting("openai.model") or "gpt-4o-mini"
    if not api_key:
        raise RuntimeError("openai.api_key is not configured; pass --zh instead")
    client = OpenAI(api_key=api_key, base_url=normalize_openai_base_url(base_url), timeout=60.0)
    reply = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": TRANSLATE_SYSTEM},
            {"role": "user", "content": english},
        ],
        max_tokens=60,
    )
    text = (reply.choices[0].message.content or "").strip().strip("《》\"'“”‘’")
    if not text:
        raise RuntimeError("the text model returned an empty translation; pass --zh instead")
    return text


def build_prompt(english: str, chinese: str) -> str:
    return (
        "Edit this YouTube thumbnail: replace the English text "
        f"'{english}' with the Chinese text '{chinese}'. "
        "Keep the exact same layout and style: same font weight, colour, alignment, "
        "size and position as the original English text, on whatever banner or "
        "background it sat on. Keep the whole title at least 5% of the image height "
        "away from every edge of the picture, so no character is ever clipped. "
        "Do not change anything else - the scene, the people, "
        "the products, the sky, the colours and the composition must stay identical. "
        "The Chinese characters must be crisp, correct and clearly legible."
    )


def request_redraw(
    image: Path,
    prompt: str,
    *,
    base_url: str,
    api_key: str,
    model: str,
    size: str,
    proxy: str | None,
) -> bytes:
    if not api_key:
        raise RuntimeError("image.api_key is not configured (settings key image.api_key)")
    proxies = {"http": proxy, "https": proxy} if proxy else None
    with image.open("rb") as handle:
        resp = requests.post(
            f"{base_url.rstrip('/')}/images/edits",
            headers={"Authorization": f"Bearer {api_key}"},
            files={"image": (image.name, handle, "image/png" if image.suffix.lower() == ".png" else "image/jpeg")},
            data={"model": model, "prompt": prompt, "size": size},
            proxies=proxies,
            timeout=600,
        )
    if not resp.ok:
        raise RuntimeError(f"image API HTTP {resp.status_code}: {resp.text[:400]}")
    payload = resp.json()
    items = payload.get("data") or []
    if not items:
        raise RuntimeError(f"image API returned no image: {str(payload)[:300]}")
    item = items[0]
    if item.get("b64_json"):
        return base64.b64decode(item["b64_json"])
    if item.get("url"):
        download = requests.get(item["url"], proxies=proxies, timeout=300)
        download.raise_for_status()
        return download.content
    raise RuntimeError(f"image API returned neither b64_json nor url: {str(payload)[:300]}")


def crop_to_16x9(source: Path, target: Path) -> None:
    """Centre-crop whatever the model returned to 16:9 and scale to 1280x720."""
    expression = (
        "crop='min(iw,ih*16/9)':'min(ih,iw*9/16)':(iw-ow)/2:(ih-oh)/2,"
        "scale=1280:720:flags=lanczos"
    )
    result = subprocess.run(
        [
            ffmpeg_binary(),
            "-y",
            "-loglevel",
            "error",
            "-i",
            str(source),
            "-vf",
            expression,
            "-q:v",
            "2",
            str(target),
        ],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg crop failed: {(result.stderr or '').strip()[:300]}")


def main(argv: list[str] | None = None) -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:  # noqa: BLE001 - cosmetic only
        pass

    parser = argparse.ArgumentParser(description="Replace the English on a cover with Chinese.")
    parser.add_argument("image", nargs="?", type=Path, help="cover image to work on")
    parser.add_argument("--session", type=Path, help="task session: fetch its original cover first")
    parser.add_argument("--en", default="", help="the English text printed on the cover")
    parser.add_argument("--zh", default="", help="the Chinese text to use (skips translation)")
    parser.add_argument("-o", "--output", type=Path, help="where to write the 16:9 JPEG")
    parser.add_argument("--size", default=DEFAULT_SIZE, help=f"image API size (default {DEFAULT_SIZE})")
    parser.add_argument("--model", default="", help="image model (default: image.model setting)")
    parser.add_argument("--base-url", default="", help="image API base url (default: image.base_url)")
    parser.add_argument("--api-key", default="", help="image API key (default: image.api_key)")
    parser.add_argument("--keep-raw", action="store_true", help="keep the uncropped model output")
    args = parser.parse_args(argv)

    session = args.session
    if session and not session.is_dir():
        parser.error(f"--session {session} is not a directory")
    if not args.image and not session:
        parser.error("give an image file or --session")

    proxy = resolve_proxy("") or None

    # 1. Resolve the source cover.
    if args.image:
        image = args.image
        if not image.exists():
            parser.error(f"{image} does not exist")
    else:
        video_id = video_id_from_session(session)
        image = session / "metadata" / "cover.jpg"
        result = fetch_cover(video_id, image, proxy=proxy or "", session=session)
        if result is None:
            print("error: could not download the original cover", file=sys.stderr)
            return 1
        print(f"cover    : downloaded {result['width']}x{result['height']} -> {image}")

    if not args.en and not args.zh:
        print("error: give --en (and optionally --zh); OCR is not wired in yet", file=sys.stderr)
        return 2

    # 2. Work out the Chinese line.
    chinese = args.zh.strip()
    if chinese:
        print(f"chinese  : {chinese} (given)")
    else:
        chinese = translate_title(args.en)
        print(f"chinese  : {chinese} (translated from {args.en!r})")

    # 3. Redraw through the image API.
    base_url = args.base_url or database.get_setting("image.base_url") or DEFAULT_BASE_URL
    api_key = args.api_key or database.get_setting("image.api_key")
    model = args.model or database.get_setting("image.model") or DEFAULT_MODEL
    print(f"image api: {base_url} model={model} size={args.size}")
    print(f"prompt   : {build_prompt(args.en or chinese, chinese)[:140]}...")

    raw_bytes = request_redraw(
        image,
        build_prompt(args.en or chinese, chinese),
        base_url=base_url,
        api_key=api_key,
        model=model,
        size=args.size,
        proxy=proxy,
    )
    output = args.output or image.with_name(f"{image.stem}.zh.jpg")
    output.parent.mkdir(parents=True, exist_ok=True)
    raw_path = output.with_suffix(".raw.png")
    raw_path.write_bytes(raw_bytes)
    print(f"redrawn  : {raw_path}  {len(raw_bytes) / 1024:.0f} KB")

    # 4. Crop to what the upload page wants.
    crop_to_16x9(raw_path, output)
    print(f"saved    : {output}  1280x720  {output.stat().st_size / 1024:.0f} KB")
    if not args.keep_raw:
        raw_path.unlink(missing_ok=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
