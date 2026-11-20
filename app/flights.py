"""Multi-temporal forest change detection over a fixed sample area.

Each registered flight pairs a capture date (YYYY-MM-DD) with an aligned
frame (RGBA/RGB PNG; transparent pixels are unobserved) and a same-size
black/white mask (white = forest, black = non-forest).  Observations are
extracted once at registration and stored per date; the change table compares
adjacent *visible* observations per pixel and records category changes.

Re-registering or deleting a date only recomputes the change records that
date's added/removed/reclassified observations can affect — every other
date's already-extracted observations are reused as-is.
"""

from __future__ import annotations

import hashlib
import io
import re
import sqlite3
from dataclasses import dataclass
from datetime import datetime
from typing import Iterable, Mapping

from PIL import Image

from .mosaic import TileValidationError, flat_pixels

MAX_SAMPLE_DIM = 256

CATEGORY_NONFOREST = 0
CATEGORY_FOREST = 1
CATEGORY_TOKEN = {
    CATEGORY_NONFOREST: "nonforest",
    CATEGORY_FOREST: "forest",
}

DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")

CSV_HEADER = "x,y,from_date,from_category,to_date,to_category"


class FlightValidationError(TileValidationError):
    """Raised when a flight registration or sample area setting is invalid."""


def validate_flight_date(raw: str) -> str:
    """Strict YYYY-MM-DD; ISO dates sort chronologically as plain strings."""
    value = (raw or "").strip()
    if not DATE_RE.match(value):
        raise FlightValidationError("拍摄日期必须是 YYYY-MM-DD 格式。")
    try:
        datetime.strptime(value, "%Y-%m-%d")
    except ValueError:
        raise FlightValidationError(f"拍摄日期 {value} 不是有效的日历日期。") from None
    return value


def validate_sample_size(width: int, height: int) -> None:
    if width < 1 or height < 1:
        raise FlightValidationError("样区宽高必须为正整数。")
    if width > MAX_SAMPLE_DIM or height > MAX_SAMPLE_DIM:
        raise FlightValidationError(
            f"样区宽高不能超过 {MAX_SAMPLE_DIM} 像元（最多 256×256）。"
        )


def decode_png(raw: bytes, label: str) -> Image.Image:
    if not raw:
        raise FlightValidationError(f"{label}为空。")
    try:
        image = Image.open(io.BytesIO(raw))
        image.load()
    except Exception:
        raise FlightValidationError(f"{label}不是可读取的 PNG 图像。") from None
    if image.format != "PNG":
        raise FlightValidationError(f"{label}必须是 PNG 格式。")
    return image


def extract_observations(
    frame: Image.Image, grey_mask: Image.Image
) -> dict[tuple[int, int], int]:
    """Per-pixel category for one flight.

    Transparent frame pixels are unobserved and simply absent from the
    result; everywhere else the mask decides forest (255) vs non-forest (0).
    """
    rgba = frame.convert("RGBA")
    alphas = list(flat_pixels(rgba.getchannel("A")))
    values = list(flat_pixels(grey_mask))
    width, _ = rgba.size
    observations: dict[tuple[int, int], int] = {}
    for index, (alpha, value) in enumerate(zip(alphas, values)):
        if alpha == 0:
            continue
        observations[(index % width, index // width)] = (
            CATEGORY_FOREST if value == 255 else CATEGORY_NONFOREST
        )
    return observations


@dataclass
class ValidatedFlight:
    flight_date: str
    width: int
    height: int
    frame_bytes: bytes
    mask_bytes: bytes
    frame_sha256: str
    mask_sha256: str
    observations: dict[tuple[int, int], int]
    observed_pixels: int
    forest_pixels: int


def validate_flight_upload(
    *,
    flight_date: str,
    sample_width: int,
    sample_height: int,
    frame_bytes: bytes,
    mask_bytes: bytes,
) -> ValidatedFlight:
    """Validate one aligned frame plus same-size strictly black/white mask."""
    date = validate_flight_date(flight_date)
    frame = decode_png(frame_bytes, "航飞画幅")
    mask = decode_png(mask_bytes, "林地掩膜")

    if frame.mode not in ("RGB", "RGBA"):
        raise FlightValidationError(
            f"航飞画幅必须是 RGB 或 RGBA PNG（当前为 {frame.mode}）；"
            "透明像元表示未观测。"
        )
    if mask.mode not in ("1", "L", "P"):
        raise FlightValidationError(
            f"林地掩膜必须是同尺寸的黑白 PNG（当前为 {mask.mode}）。"
        )
    if frame.size != mask.size:
        raise FlightValidationError(
            "林地掩膜尺寸必须与画幅一致"
            f"（画幅 {frame.width}x{frame.height}，掩膜 {mask.width}x{mask.height}）。"
        )
    width, height = frame.size
    if (width, height) != (sample_width, sample_height):
        raise FlightValidationError(
            f"画幅尺寸必须等于固定样区尺寸 {sample_width}x{sample_height}"
            f"（当前 {width}x{height}）。"
        )

    grey = mask.convert("L")
    histogram = grey.histogram()
    if any(
        count for value, count in enumerate(histogram)
        if value not in (0, 255) and count
    ):
        raise FlightValidationError(
            "林地掩膜只能包含纯黑（0=非林地）与纯白（255=林地）两种像元值。"
        )

    observations = extract_observations(frame, grey)
    forest_pixels = sum(
        1 for category in observations.values() if category == CATEGORY_FOREST
    )
    return ValidatedFlight(
        flight_date=date,
        width=width,
        height=height,
        frame_bytes=frame_bytes,
        mask_bytes=mask_bytes,
        frame_sha256=hashlib.sha256(frame_bytes).hexdigest(),
        mask_sha256=hashlib.sha256(mask_bytes).hexdigest(),
        observations=observations,
        observed_pixels=len(observations),
        forest_pixels=forest_pixels,
    )


@dataclass
class ChangeStats:
    affected_pixels: int
    removed_records: int
    added_records: int


def load_observations(
    conn: sqlite3.Connection, flight_date: str
) -> dict[tuple[int, int], int]:
    rows = conn.execute(
        "SELECT x, y, category FROM flight_observations WHERE flight_date = ?",
        (flight_date,),
    ).fetchall()
    return {(row["x"], row["y"]): row["category"] for row in rows}


def _neighbour_observation(conn, x, y, flight_date, direction):
    """Closest stored observation before/after `flight_date` at one pixel."""
    if direction < 0:
        clause = "flight_date < ? ORDER BY flight_date DESC"
    else:
        clause = "flight_date > ? ORDER BY flight_date ASC"
    row = conn.execute(
        f"SELECT flight_date, category FROM flight_observations "
        f"WHERE x = ? AND y = ? AND {clause} LIMIT 1",
        (x, y, flight_date),
    ).fetchone()
    if row is None:
        return None
    return (row["flight_date"], row["category"])


def apply_flight_observations(
    conn: sqlite3.Connection,
    flight_date: str,
    new_observations: Mapping[tuple[int, int], int],
) -> ChangeStats:
    """Swap one date's observations and repair only the affected records.

    A change record can only be influenced by this date's swap if, at a pixel
    whose observation actually changed, the record has this date as an
    endpoint or spans across it.  Those records are deleted and re-derived
    from the stored observations of the neighbouring dates; every other
    record — and every other date's extracted observations — stays untouched.

    Must be called inside the caller's transaction so the flight row, the
    observations and the change table commit or roll back together.
    """
    old_observations = load_observations(conn, flight_date)
    affected = sorted(
        pixel
        for pixel in set(old_observations) | set(new_observations)
        if old_observations.get(pixel) != new_observations.get(pixel)
    )

    conn.execute(
        "DELETE FROM flight_observations WHERE flight_date = ?", (flight_date,)
    )
    conn.executemany(
        "INSERT INTO flight_observations(flight_date, x, y, category) "
        "VALUES (?, ?, ?, ?)",
        [
            (flight_date, x, y, category)
            for (x, y), category in sorted(new_observations.items())
        ],
    )

    removed = 0
    added = 0
    for x, y in affected:
        cursor = conn.execute(
            "DELETE FROM change_records WHERE x = ? AND y = ? AND "
            "(from_date = ? OR to_date = ? OR (from_date < ? AND to_date > ?))",
            (x, y, flight_date, flight_date, flight_date, flight_date),
        )
        removed += cursor.rowcount

        previous = _neighbour_observation(conn, x, y, flight_date, -1)
        following = _neighbour_observation(conn, x, y, flight_date, +1)
        current = new_observations.get((x, y))

        pairs = []
        if current is None:
            # Observation removed: the neighbours become adjacent visible
            # observations and may now form a change pair across the gap.
            if previous is not None and following is not None:
                pairs.append((previous, following))
        else:
            here = (flight_date, current)
            if previous is not None:
                pairs.append((previous, here))
            if following is not None:
                pairs.append((here, following))

        for first, second in pairs:
            if first[1] == second[1]:
                continue
            conn.execute(
                "INSERT INTO change_records("
                "x, y, from_date, from_category, to_date, to_category) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (x, y, first[0], first[1], second[0], second[1]),
            )
            added += 1

    return ChangeStats(
        affected_pixels=len(affected), removed_records=removed, added_records=added
    )


def build_changes_csv(rows: Iterable[sqlite3.Row]) -> str:
    """Deterministic CSV: one line per change record, row-major then date."""
    lines = [CSV_HEADER]
    for row in rows:
        lines.append(
            f"{row['x']},{row['y']},{row['from_date']},"
            f"{CATEGORY_TOKEN[row['from_category']]},{row['to_date']},"
            f"{CATEGORY_TOKEN[row['to_category']]}"
        )
    return "\n".join(lines) + "\n"
