"""Read the text printed on a cover, so the localiser knows what to replace.

The OCR is local (RapidOCR on onnxruntime, PP-OCR models shipped inside the
package) - no network call, no API key, no tesseract install. A 1280x720 cover
takes about one second.

What comes back from the engine is one box per *visual fragment*: a stacked
title like

    The Right
    Way to
    Sharpen a
    Pencil

arrives as four separate boxes, and a row of words ("HYPERSONIC" "ICE" "BALLS")
is three. Feeding a single fragment to the image model would replace only that
fragment, so this module merges boxes into rows and rows into blocks first, and
the block is what gets handed to the translator.

Usage
-----
    ./.venv/Scripts/python.exe tools/cover_ocr.py cover.jpg
    ./.venv/Scripts/python.exe tools/cover_ocr.py cover.jpg --json
    ./.venv/Scripts/python.exe tools/cover_ocr.py --session workfolder/... --top 1
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

LATIN_RE = re.compile(r"[A-Za-z]")
#: Anything that is a letter in either script - used as the denominator when
#: asking "how much of this text is Latin?".
LETTER_RE = re.compile(r"[A-Za-z\u4e00-\u9fff]")

#: Below this the engine is guessing at noise (a logo, a face).
MIN_SCORE = 0.55
#: Latin letters needed before a block counts as "has English on it".
MIN_LATIN = 3

#: A cover is small; upscaling makes the thin strokes of a condensed font
#: readable for the detector, and it is the difference between readable and
#: gibberish on the 480x360 fallback thumbnail. Costs about a second.
UPSCALE = 2.0

_engine = None


@dataclass
class TextLine:
    """One fragment as the engine saw it, in source-image pixels."""

    text: str
    score: float
    x0: float
    y0: float
    x1: float
    y1: float

    @property
    def height(self) -> float:
        return self.y1 - self.y0

    @property
    def width(self) -> float:
        return self.x1 - self.x0

    @property
    def mid_y(self) -> float:
        return (self.y0 + self.y1) / 2

    @property
    def mid_x(self) -> float:
        return (self.x0 + self.x1) / 2


@dataclass
class TextBlock:
    """One logical piece of copy: a stacked title, a channel name, a logo."""

    lines: list[TextLine] = field(default_factory=list)

    @property
    def text(self) -> str:
        return " ".join(line.text for line in self.lines)

    @property
    def score(self) -> float:
        return sum(line.score for line in self.lines) / len(self.lines) if self.lines else 0.0

    @property
    def x0(self) -> float:
        return min(line.x0 for line in self.lines)

    @property
    def y0(self) -> float:
        return min(line.y0 for line in self.lines)

    @property
    def x1(self) -> float:
        return max(line.x1 for line in self.lines)

    @property
    def y1(self) -> float:
        return max(line.y1 for line in self.lines)

    @property
    def height(self) -> float:
        """Tallest line - a proxy for the font size, which is what draws the eye."""
        return max(line.height for line in self.lines)

    @property
    def area(self) -> float:
        return (self.x1 - self.x0) * (self.y1 - self.y0)

    @property
    def latin(self) -> int:
        return len(LATIN_RE.findall(self.text))

    @property
    def latin_ratio(self) -> float:
        letters = len(LETTER_RE.findall(self.text))
        return self.latin / letters if letters else 0.0

    @property
    def has_english(self) -> bool:
        return self.latin >= MIN_LATIN and self.latin_ratio >= 0.4

    def as_dict(self) -> dict:
        return {
            "text": self.text,
            "lines": [line.text for line in self.lines],
            "score": round(self.score, 3),
            "latin_ratio": round(self.latin_ratio, 3),
            "has_english": self.has_english,
            "box": [int(self.x0), int(self.y0), int(self.x1), int(self.y1)],
            "font_height": int(self.height),
        }


def load_engine():
    """Load the OCR engine once per process (weights come with the package)."""
    global _engine
    if _engine is None:
        from rapidocr_onnxruntime import RapidOCR

        _engine = RapidOCR()
    return _engine


def read_lines(image: Path, *, min_score: float = MIN_SCORE, upscale: float = UPSCALE) -> list[TextLine]:
    """Every readable fragment on ``image``, coordinates mapped back to the original."""
    import numpy as np
    from PIL import Image

    engine = load_engine()
    with Image.open(image) as handle:
        picture = handle.convert("RGB")
        if upscale and upscale != 1.0:
            picture = picture.resize(
                (round(picture.width * upscale), round(picture.height * upscale)),
                Image.LANCZOS,
            )
        # The engine follows the OpenCV convention for arrays: BGR, not RGB.
        array = np.asarray(picture)[:, :, ::-1]

    result, _ = engine(array)
    lines: list[TextLine] = []
    for item in result or []:
        points, text, score = item[0], str(item[1]), float(item[2])
        text = text.strip()
        if not text or score < min_score:
            continue
        scale = upscale or 1.0
        xs = [float(point[0]) / scale for point in points]
        ys = [float(point[1]) / scale for point in points]
        lines.append(TextLine(text, score, min(xs), min(ys), max(xs), max(ys)))
    return lines


def _same_row(a: TextLine, b: TextLine) -> bool:
    """Two fragments sitting side by side on one visual line."""
    overlap = min(a.y1, b.y1) - max(a.y0, b.y0)
    if overlap <= 0:
        return False
    if overlap < 0.5 * min(a.height, b.height):
        return False
    gap = max(b.x0 - a.x1, a.x0 - b.x1)
    return gap <= 1.5 * max(a.height, b.height)


def merge_rows(lines: list[TextLine]) -> list[TextLine]:
    """Join fragments that share a row, left to right, into one line."""
    ordered = sorted(lines, key=lambda line: (line.mid_y, line.x0))
    rows: list[TextLine] = []
    for line in ordered:
        for row in rows:
            if _same_row(row, line):
                left, right = (row, line) if row.x0 <= line.x0 else (line, row)
                rows[rows.index(row)] = TextLine(
                    f"{left.text} {right.text}",
                    min(left.score, right.score),
                    min(left.x0, right.x0),
                    min(left.y0, right.y0),
                    max(left.x1, right.x1),
                    max(left.y1, right.y1),
                )
                break
        else:
            rows.append(line)
    return sorted(rows, key=lambda row: (row.y0, row.x0))


def _stacks_under(row: TextLine, block: TextBlock) -> bool:
    """A row belongs to a block when it sits tight under it and lines up with it."""
    gap = row.y0 - block.y1
    if gap < -0.3 * row.height:
        return False
    if gap > 0.8 * max(row.height, block.height):
        return False
    # Aligned left, right, or centred - anything but a wild horizontal jump.
    overlap = min(row.x1, block.x1) - max(row.x0, block.x0)
    if overlap > 0:
        return True
    return abs(row.mid_x - (block.x0 + block.x1) / 2) <= max(row.height, block.height)


def merge_blocks(rows: list[TextLine]) -> list[TextBlock]:
    """Stack rows that are visually one piece of copy into blocks."""
    blocks: list[TextBlock] = []
    for row in sorted(rows, key=lambda item: item.y0):
        for block in blocks:
            if _stacks_under(row, block):
                block.lines.append(row)
                break
        else:
            blocks.append(TextBlock([row]))
    return blocks


def read_blocks(image: Path, *, min_score: float = MIN_SCORE) -> list[TextBlock]:
    """All text blocks on the cover, biggest font first."""
    blocks = merge_blocks(merge_rows(read_lines(image, min_score=min_score)))
    return sorted(blocks, key=lambda block: (block.height, block.area), reverse=True)


def english_blocks(image: Path, *, min_score: float = MIN_SCORE) -> list[TextBlock]:
    """Only the blocks that carry Latin letters, biggest font first."""
    return [block for block in read_blocks(image, min_score=min_score) if block.has_english]


def main(argv: list[str] | None = None) -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:  # noqa: BLE001 - cosmetic only
        pass

    parser = argparse.ArgumentParser(description="Read the text on a cover image.")
    parser.add_argument("image", nargs="?", type=Path, help="cover image")
    parser.add_argument("--session", type=Path, help="task session (uses metadata/cover.jpg)")
    parser.add_argument("--json", action="store_true", help="machine-readable output")
    parser.add_argument("--top", type=int, default=0, help="keep only the N biggest English blocks")
    parser.add_argument("--min-score", type=float, default=MIN_SCORE, help=f"engine confidence floor (default {MIN_SCORE})")
    args = parser.parse_args(argv)

    image = args.image or (args.session / "metadata" / "cover.jpg" if args.session else None)
    if image is None:
        parser.error("give an image file or --session")
    if not image.exists():
        parser.error(f"{image} does not exist")

    blocks = english_blocks(image, min_score=args.min_score)
    if args.top:
        blocks = blocks[: args.top]

    if args.json:
        print(json.dumps([block.as_dict() for block in blocks], ensure_ascii=False, indent=2))
        return 0 if blocks else 1

    if not blocks:
        print("no English found on the cover")
        return 1

    print(f"{image}  ({len(blocks)} English block{'s' if len(blocks) != 1 else ''})")
    for index, block in enumerate(blocks):
        box = f"{int(block.x0)},{int(block.y0)}"
        print(f"  [{index}] h={int(block.height):>3}px conf={block.score:.2f} box={box}  {block.text}")
        if len(block.lines) > 1:
            for line in block.lines:
                print(f"        | {line.text}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
