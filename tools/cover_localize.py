"""Put Chinese words on a YouTube cover, using the image API from the settings.

The English on a thumbnail has to become Chinese before it can be used as a
Bilibili cover. Two services, one job each:

* local OCR (``tools/cover_ocr.py``) answers one question only - *is there
  English on this cover, and what does it say*. If there is none, the original
  cover is fine and nothing else happens;
* the image model (``image.*`` settings, aicost.me gpt-image-2) gets the whole
  picture and does both halves: it translates the English into Chinese and
  redraws the cover with that Chinese in place, keeping the layout, colours and
  picture intact. No text model is involved, so a dead translation relay cannot
  block this.

The generated picture is cropped to 16:9 and resized to 1280x720 - what the
upload page wants - and then read again by the OCR to confirm the English is
actually gone and Chinese characters are actually there.

Examples
--------
    # the whole way: find the cover, check for English, redraw, check the result
    python tools/cover_localize.py --session workfolder/xxx/yyy__ID

    # a local file
    python tools/cover_localize.py cover.jpg -o cover.zh.jpg

    # force the Chinese wording instead of letting the image model translate
    python tools/cover_localize.py cover.jpg --en "HYPERSONIC ICE BALLS" --zh "超音速冰球"

    # only look at what the OCR sees
    python tools/cover_localize.py cover.jpg --list

Every English block on the cover is replaced when ``--all`` is given; otherwise
the biggest one (the title) is replaced and the rest are reported so you can
pick another with ``--ocr-index``.

``localize_cover()`` is the same thing without the command line: the publish
stage calls it through ``backend/app/adapters/cover.py``.
"""

from __future__ import annotations

import argparse
import base64
import re
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backend.app import database  # noqa: E402
from backend.app.config import ffmpeg_binary  # noqa: E402
from tools.cover_ocr import TextBlock, english_blocks, read_blocks  # noqa: E402
from tools.fetch_cover import fetch_cover, resolve_proxy, video_id_from_session  # noqa: E402

DEFAULT_BASE_URL = "https://www.aicost.me/v1"
DEFAULT_MODEL = "gpt-image-2"
# ``auto`` makes the model follow the input's aspect ratio. The covers are 16:9,
# so this returns 16:9 back and the crop below stays lossless - a fixed 3:2 size
# ends up cutting the title that sits at the very top of the picture.
DEFAULT_SIZE = "auto"

CJK_RE = re.compile(r"[\u4e00-\u9fff]")

#: Shared tail: what must not move, and how the Chinese has to look.
KEEP_RULES = (
    "Keep the exact same layout and style: same font weight, colour, alignment and "
    "position as the English it replaces, on whatever banner or background it sat on. "
    "Keep the text at least 5% of the image height away from every edge of the picture, "
    "so no character is ever clipped. Do not change anything else - the scene, the "
    "people, the products, the sky, the colours and the composition must stay identical. "
    "The Chinese characters must be crisp, correct and clearly legible."
)


@dataclass
class LocalizeResult:
    """What came out of one run - enough for the caller to decide to keep it."""

    output: Path
    targets: list[str]
    chinese: str
    latin_before: int
    latin_after: int
    cjk_after: int
    leftover: list[str] = field(default_factory=list)
    others: list[str] = field(default_factory=list)

    @property
    def usable(self) -> bool:
        """Chinese is on the picture and none of the target English is left."""
        return self.cjk_after > 0 and not self.leftover


def side_of(block: TextBlock, width: int) -> str:
    """Which third of the picture the block sits in."""
    centre = (block.x0 + block.x1) / 2 / max(width, 1)
    if centre < 0.34:
        return "left"
    if centre > 0.66:
        return "right"
    return "centre"


def build_prompt(targets: list[str], chinese: str = "", sides: list[str] | None = None) -> str:
    """The instruction handed to the image model.

    With one line of English and no Chinese given, the model is asked to do the
    translation itself - that is the normal path, and it means no text model and
    no extra round trip. ``chinese`` only appears when the wording was forced.
    """
    if len(targets) == 1:
        english = targets[0]
        where = f"It sits on the {sides[0]} side of the picture. " if sides else ""
        if chinese:
            action = (
                f"the English text on it says '{english}'. Replace that English with "
                f"the Chinese text '{chinese}'. "
            )
        else:
            action = (
                f"the English text on it says '{english}'. Translate that into natural, "
                "punchy Chinese and replace the English with the Chinese text, in the "
                "very same place. Keep the Chinese short - at most 12 characters - so it "
                "fits the same width, and write it on one line if that fits, otherwise "
                "keep the same number of lines. "
            )
        return f"Edit this YouTube thumbnail: {action}{where}{KEEP_RULES}"

    if chinese:
        lines = "\n".join(
            f'{index}. "{en}" -> "{zh}"' for index, (en, zh) in enumerate(zip(targets, [chinese] * len(targets)), 1)
        )
        action = (
            "replace each English text below with its Chinese translation, one for one, "
            f"every one staying exactly where its English original was:\n{lines}\n"
        )
    else:
        lines = "\n".join(f'{index}. "{en}"' for index, en in enumerate(targets, 1))
        action = (
            "translate each of these English texts into natural, punchy Chinese and put "
            "each Chinese text exactly where its English original was, one for one, "
            f"short enough to fit the same width (at most 12 characters each):\n{lines}\n"
        )
    return f"Edit this YouTube thumbnail: {action}{KEEP_RULES}"


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


def count_scripts(path: Path) -> tuple[int, int]:
    """Latin letters and Chinese characters the OCR can still read on ``path``."""
    latin = cjk = 0
    for block in read_blocks(path):
        latin += block.latin
        cjk += len(CJK_RE.findall(block.text))
    return latin, cjk


def still_there(target: str, blocks: list[TextBlock]) -> bool:
    """Is the English we asked to replace still readable on the result?

    Latin left over somewhere else is normal - a channel name, a logo, a product
    label - so the check is on the words of the target line, not on "any Latin".
    """
    words = re.findall(r"[A-Za-z]{3,}", target)
    if not words:
        return False
    haystack = " ".join(block.text for block in blocks).lower()
    hits = sum(1 for word in words if word.lower() in haystack)
    return hits / len(words) >= 0.6


def localize_cover(
    image: Path,
    output: Path,
    *,
    english: str = "",
    chinese: str = "",
    all_blocks: bool = False,
    index: int = 0,
    size: str = DEFAULT_SIZE,
    model: str = "",
    base_url: str = "",
    api_key: str = "",
    proxy: str | None = None,
    verify: bool = True,
    keep_raw: bool = False,
    log=print,
) -> LocalizeResult | None:
    """Read ``image``, redraw its English in Chinese, write ``output``.

    Returns ``None`` when the cover has no English at all - the caller should
    then keep the original picture, no credits spent. Raises when the image API
    fails, so the caller can fall back to something else.
    """
    detected: list[TextBlock] = []
    if english:
        targets = [english.strip()]
        log(f"english  : {english!r} (given)")
    else:
        detected = english_blocks(image)
        if not detected:
            log("english  : none on the cover - keep the original, nothing to redraw")
            return None
        if all_blocks:
            chosen = detected
        else:
            if not 0 <= index < len(detected):
                raise ValueError(f"index {index} is out of range (0-{len(detected) - 1})")
            chosen = [detected[index]]
            for position, block in enumerate(detected):
                mark = "->" if position == index else "  "
                log(f"  {mark} [{position}] h={int(block.height):>3}px | {block.text}")
        targets = [block.text for block in chosen]
        log(f"english  : {targets[0]!r}" + (f" + {len(targets) - 1} more" if len(targets) > 1 else "") + " (OCR)")

    if chinese and len(targets) > 1:
        raise ValueError("--zh gives one line of Chinese; drop --all to translate a single block")

    sides: list[str] = []
    if detected:
        from PIL import Image

        with Image.open(image) as handle:
            picture = handle.width
        sides = [side_of(block, picture) for block in detected if block.text in targets]

    prompt = build_prompt(targets, chinese, sides or None)
    api_base = base_url or database.get_setting("image.base_url") or DEFAULT_BASE_URL
    key = api_key or database.get_setting("image.api_key")
    chosen_model = model or database.get_setting("image.model") or DEFAULT_MODEL
    log(f"image api: {api_base} model={chosen_model} size={size}")
    log(f"prompt   : {prompt[:200]}...")

    raw_bytes = request_redraw(
        image,
        prompt,
        base_url=api_base,
        api_key=key,
        model=chosen_model,
        size=size,
        proxy=proxy,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    raw_path = output.with_suffix(".raw.png")
    raw_path.write_bytes(raw_bytes)
    log(f"redrawn  : {raw_path}  {len(raw_bytes) / 1024:.0f} KB")

    crop_to_16x9(raw_path, output)
    log(f"saved    : {output}  1280x720  {output.stat().st_size / 1024:.0f} KB")
    if not keep_raw:
        raw_path.unlink(missing_ok=True)

    result = LocalizeResult(output=output, targets=targets, chinese=chinese, latin_before=0, latin_after=0, cjk_after=0)
    if verify:
        try:
            result.latin_before, _ = count_scripts(image)
            after_blocks = read_blocks(output)
            result.latin_after = sum(block.latin for block in after_blocks)
            result.cjk_after = sum(len(CJK_RE.findall(block.text)) for block in after_blocks)
            result.leftover = [target for target in targets if still_there(target, after_blocks)]
            result.others = [block.text for block in after_blocks if block.has_english]
        except Exception as exc:  # noqa: BLE001 - a failed check must not fail the run
            log(f"check    : skipped ({exc})")
        else:
            log(f"check    : chinese {result.cjk_after} chars, latin {result.latin_before} -> {result.latin_after}")
            if result.cjk_after == 0:
                log("warning  : no Chinese found on the result - look at the picture before publishing")
            elif result.leftover:
                log(f"warning  : still readable on the result: {result.leftover!r} - try again")
            if result.others:
                log(f"leftover : {' | '.join(result.others)}  (not the title; left alone on purpose)")
    return result


def main(argv: list[str] | None = None) -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:  # noqa: BLE001 - cosmetic only
        pass

    parser = argparse.ArgumentParser(description="Replace the English on a cover with Chinese.")
    parser.add_argument("image", nargs="?", type=Path, help="cover image to work on")
    parser.add_argument("--session", type=Path, help="task session: fetch its original cover first")
    parser.add_argument("--en", default="", help="English text on the cover (default: read it with OCR)")
    parser.add_argument("--zh", default="", help="force this Chinese wording instead of letting the image model translate")
    parser.add_argument("-o", "--output", type=Path, help="where to write the 16:9 JPEG")
    parser.add_argument("--size", default=DEFAULT_SIZE, help=f"image API size (default {DEFAULT_SIZE})")
    parser.add_argument("--model", default="", help="image model (default: image.model setting)")
    parser.add_argument("--base-url", default="", help="image API base url (default: image.base_url)")
    parser.add_argument("--api-key", default="", help="image API key (default: image.api_key)")
    parser.add_argument("--keep-raw", action="store_true", help="keep the uncropped model output")
    parser.add_argument("--list", action="store_true", help="only print what the OCR reads, then exit")
    parser.add_argument("--all", action="store_true", help="replace every English block, not just the title")
    parser.add_argument("--ocr-index", type=int, default=0, help="which detected block to use (default 0 = biggest)")
    parser.add_argument("--no-verify", action="store_true", help="skip reading the result back")
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

    # 2. Only look at the OCR, if that is all that was asked.
    if args.list:
        blocks = english_blocks(image)
        if not blocks:
            print("no English found on the cover")
            return 1
        for index, block in enumerate(blocks):
            print(f"  [{index}] h={int(block.height):>3}px conf={block.score:.2f} | {block.text}")
        return 0

    # 3. Redraw, then report.
    output = args.output or image.with_name(f"{image.stem}.zh.jpg")
    try:
        result = localize_cover(
            image,
            output,
            english=args.en,
            chinese=args.zh.strip(),
            all_blocks=args.all,
            index=args.ocr_index,
            size=args.size,
            model=args.model,
            base_url=args.base_url,
            api_key=args.api_key,
            proxy=proxy,
            verify=not args.no_verify,
            keep_raw=args.keep_raw,
        )
    except Exception as exc:  # noqa: BLE001 - the command line reports, it does not trace
        print(f"error    : {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    if result is None:
        return 0
    return 0 if result.usable else 3


if __name__ == "__main__":
    raise SystemExit(main())
