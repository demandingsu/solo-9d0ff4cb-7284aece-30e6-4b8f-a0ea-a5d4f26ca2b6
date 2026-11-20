"""Forest change monitoring: flights, observations and incremental changes."""

from __future__ import annotations

import csv
import io
import random
import sqlite3
from urllib.parse import unquote

import pytest
from PIL import Image

from app import flights
from app.flights import (
    FlightValidationError,
    validate_flight_date,
    validate_sample_size,
)

D1 = "2026-03-01"
D2 = "2026-03-02"
D3 = "2026-03-03"

HEADER = ["x", "y", "from_date", "from_category", "to_date", "to_category"]

ALL_2X2 = {(x, y) for x in range(2) for y in range(2)}
ALL_4X4 = {(x, y) for x in range(4) for y in range(4)}


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def frame_bytes(width, height, transparent=(), colour=(40, 160, 40), mode="RGBA"):
    """Frame image; transparent pixels (alpha 0) mean unobserved."""
    if mode == "RGBA":
        image = Image.new("RGBA", (width, height), tuple(colour) + (255,))
        pixels = image.load()
        for x, y in transparent:
            pixels[x, y] = (0, 0, 0, 0)
    else:
        image = Image.new(mode, (width, height), 0)
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def land_mask(width, height, forest=(), value=None):
    """Black/white mask: 255 = forest, 0 = non-forest."""
    image = Image.new("L", (width, height), 0)
    pixels = image.load()
    for x, y in forest:
        pixels[x, y] = 255
    if value is not None:  # sprinkle a non-binary value for validation tests
        pixels[value[0], value[1]] = value[2]
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def set_sample(client, width, height):
    return client.post(
        "/flights/sample",
        data={"width": str(width), "height": str(height)},
        follow_redirects=False,
    )


def register(client, date, frame, mask):
    files = {
        "frame": ("frame.png", frame, "image/png"),
        "mask": ("mask.png", mask, "image/png"),
        "flight_date": (None, date),
    }
    return client.post("/flights/register", files=files, follow_redirects=False)


def delete_flight(client, date):
    return client.post(f"/flights/{date}/delete", follow_redirects=False)


def fetch_changes(client):
    response = client.get("/flights/changes.csv")
    assert response.status_code == 200, response.text
    rows = list(csv.reader(io.StringIO(response.content.decode("utf-8"))))
    assert rows[0] == HEADER
    return rows[1:]


def db_rows(client, sql, args=()):
    conn = sqlite3.connect(client.app.state.runtime.db_path)
    try:
        return conn.execute(sql, args).fetchall()
    finally:
        conn.close()


def actual_changes(client):
    return set(
        db_rows(
            client,
            "SELECT x, y, from_date, from_category, to_date, to_category "
            "FROM change_records",
        )
    )


def reference_changes(client):
    """Full recompute from the stored observations (the expected table)."""
    rows = db_rows(
        client,
        "SELECT flight_date, x, y, category FROM flight_observations "
        "ORDER BY x, y, flight_date",
    )
    by_pixel = {}
    for date, x, y, category in rows:
        by_pixel.setdefault((x, y), []).append((date, category))
    expected = set()
    for (x, y), series in by_pixel.items():
        for first, second in zip(series, series[1:]):
            if first[1] != second[1]:
                expected.add((x, y, first[0], first[1], second[0], second[1]))
    return expected


@pytest.fixture()
def sample(client):
    set_sample(client, 4, 4)
    return client


# --------------------------------------------------------------------------- #
# Sample area
# --------------------------------------------------------------------------- #


def test_sample_area_must_be_set_before_registration(client):
    response = register(client, D1, frame_bytes(4, 4), land_mask(4, 4))
    assert response.status_code == 303
    assert "err=" in response.headers["location"]
    assert db_rows(client, "SELECT * FROM flights") == []


def test_sample_area_bounds(client):
    for bad in (0, -1, 257, 1000):
        response = set_sample(client, bad, 10)
        assert "err=" in response.headers["location"]
    for good in (1, 256):
        response = set_sample(client, good, good)
        assert "err=" not in response.headers["location"]


def test_sample_area_locked_while_flights_registered(client, sample):
    register(client, D1, frame_bytes(4, 4), land_mask(4, 4))
    response = set_sample(client, 5, 5)
    assert "err=" in unquote(response.headers["location"])
    # Re-posting the same size is a harmless no-op.
    assert "err=" not in set_sample(client, 4, 4).headers["location"]
    # After removing every flight the sample area can change again.
    assert "err=" not in delete_flight(client, D1).headers["location"]
    assert "err=" not in set_sample(client, 5, 5).headers["location"]


def test_sample_size_validation():
    validate_sample_size(1, 256)
    validate_sample_size(256, 256)
    for bad in [(0, 1), (1, 0), (257, 1), (1, 257)]:
        with pytest.raises(FlightValidationError):
            validate_sample_size(*bad)


# --------------------------------------------------------------------------- #
# Flight validation
# --------------------------------------------------------------------------- #


def test_flight_date_validation():
    assert validate_flight_date("2026-03-01") == "2026-03-01"
    for bad in ["", "2026-3-1", "20260301", "2026/03/01", "2026-02-30", "2026-13-01"]:
        with pytest.raises(FlightValidationError):
            validate_flight_date(bad)


def test_register_rejects_bad_date(client, sample):
    response = register(client, "2026-13-40", frame_bytes(4, 4), land_mask(4, 4))
    assert "err=" in response.headers["location"]
    assert db_rows(client, "SELECT * FROM flights") == []


def test_register_rejects_wrong_size(client, sample):
    response = register(client, D1, frame_bytes(3, 4), land_mask(3, 4))
    assert "err=" in unquote(response.headers["location"])
    response = register(client, D1, frame_bytes(4, 4), land_mask(4, 3))
    assert "err=" in unquote(response.headers["location"])


def test_register_rejects_non_binary_mask(client, sample):
    response = register(
        client, D1, frame_bytes(4, 4), land_mask(4, 4, value=(1, 1, 128))
    )
    assert "err=" in unquote(response.headers["location"])


def test_register_rejects_bad_modes(client, sample):
    grey_frame = frame_bytes(4, 4, mode="L")
    assert "err=" in register(client, D1, grey_frame, land_mask(4, 4)).headers[
        "location"
    ]
    rgb_mask = io.BytesIO()
    Image.new("RGB", (4, 4)).save(rgb_mask, format="PNG")
    response = register(client, D1, frame_bytes(4, 4), rgb_mask.getvalue())
    assert "err=" in response.headers["location"]


def test_rgb_frame_is_fully_observed(client, sample):
    register(client, D1, frame_bytes(4, 4, mode="RGB"), land_mask(4, 4))
    (observed,) = db_rows(client, "SELECT observed_pixels FROM flights")[0]
    assert observed == 16


# --------------------------------------------------------------------------- #
# Change detection
# --------------------------------------------------------------------------- #


def test_basic_change_csv(client, sample):
    register(client, D1, frame_bytes(4, 4), land_mask(4, 4, forest=ALL_4X4))
    register(client, D2, frame_bytes(4, 4), land_mask(4, 4))
    rows = fetch_changes(client)
    assert len(rows) == 16
    assert rows[0] == ["0", "0", D1, "forest", D2, "nonforest"]
    assert rows[-1] == ["3", "3", D1, "forest", D2, "nonforest"]
    response = client.get("/flights/changes.csv")
    assert "attachment" in response.headers["content-disposition"]
    assert "changes.csv" in response.headers["content-disposition"]
    # The page summarises harvest (forest -> non-forest) for investigators.
    page = client.get("/flights").text
    assert "采伐" in page and "复绿" in page


def test_empty_change_csv_is_downloadable(client):
    response = client.get("/flights/changes.csv")
    assert response.status_code == 200
    assert response.content.decode("utf-8") == ",".join(HEADER) + "\n"


def test_change_spans_unobserved_date(client):
    """D2 unobserved at a pixel: the change links D1 directly to D3."""
    set_sample(client, 2, 1)
    register(client, D1, frame_bytes(2, 1), land_mask(2, 1, forest={(0, 0), (1, 0)}))
    register(
        client, D2,
        frame_bytes(2, 1, transparent={(0, 0)}),
        land_mask(2, 1, forest={(1, 0)}),
    )
    register(client, D3, frame_bytes(2, 1), land_mask(2, 1))
    rows = fetch_changes(client)
    assert rows == [
        ["0", "0", D1, "forest", D3, "nonforest"],  # gap over D2
        ["1", "0", D2, "forest", D3, "nonforest"],  # D2 visible here
    ]


def test_backfill_between_existing_dates(client):
    """Backfilling D2 rewrites only the records that span or touch D2."""
    set_sample(client, 2, 2)
    register(client, D1, frame_bytes(2, 2), land_mask(2, 2, forest=ALL_2X2))
    register(client, D3, frame_bytes(2, 2), land_mask(2, 2))
    assert {r[3] for r in fetch_changes(client)} == {"forest"}
    assert {(r[2], r[4]) for r in fetch_changes(client)} == {(D1, D3)}

    # Backfill D2 as all-forest: changes move to the D2 -> D3 boundary.
    response = register(client, D2, frame_bytes(2, 2), land_mask(2, 2, forest=ALL_2X2))
    location = unquote(response.headers["location"])
    assert "影响像元 4" in location and "变化记录 -4/+4" in location
    assert {(r[2], r[4]) for r in fetch_changes(client)} == {(D2, D3)}

    # Replace D2 with a fully unobserved frame: changes span the gap again.
    register(client, D2, frame_bytes(2, 2, transparent=ALL_2X2), land_mask(2, 2))
    assert {(r[2], r[4]) for r in fetch_changes(client)} == {(D1, D3)}


def test_same_day_replace_only_touches_that_date(client):
    """Replacing D2's images leaves records not involving D2 untouched."""
    set_sample(client, 2, 1)
    register(client, D1, frame_bytes(2, 1), land_mask(2, 1, forest={(0, 0), (1, 0)}))
    register(
        client, D2,
        frame_bytes(2, 1, transparent={(1, 0)}),
        land_mask(2, 1, forest={(0, 0)}),
    )
    register(client, D3, frame_bytes(2, 1), land_mask(2, 1))
    assert fetch_changes(client) == [
        ["0", "0", D2, "forest", D3, "nonforest"],
        ["1", "0", D1, "forest", D3, "nonforest"],
    ]

    # Swap D2's mask: pixel (0,0) becomes non-forest; (1,0) stays unobserved.
    response = register(
        client, D2,
        frame_bytes(2, 1, transparent={(1, 0)}),
        land_mask(2, 1),
    )
    location = unquote(response.headers["location"])
    assert "已替换" in location
    assert "影响像元 1" in location and "变化记录 -1/+1" in location
    assert fetch_changes(client) == [
        ["0", "0", D1, "forest", D2, "nonforest"],  # rewritten
        ["1", "0", D1, "forest", D3, "nonforest"],  # untouched
    ]


def test_replace_decodes_only_the_new_upload(client, monkeypatch):
    """Other dates' already-extracted observations are reused, not re-read."""
    set_sample(client, 2, 1)
    for date in (D1, D2, D3):
        register(client, date, frame_bytes(2, 1), land_mask(2, 1, forest={(0, 0)}))

    calls = []
    real_decode = flights.decode_png

    def spy(raw, label):
        calls.append(label)
        return real_decode(raw, label)

    monkeypatch.setattr(flights, "decode_png", spy)
    register(client, D2, frame_bytes(2, 1), land_mask(2, 1))
    assert calls == ["航飞画幅", "林地掩膜"]
    assert actual_changes(client) == reference_changes(client)


def test_delete_flight_reconnects_across_the_gap(client):
    set_sample(client, 2, 1)
    register(client, D1, frame_bytes(2, 1), land_mask(2, 1, forest={(0, 0), (1, 0)}))
    register(client, D2, frame_bytes(2, 1), land_mask(2, 1))
    register(client, D3, frame_bytes(2, 1), land_mask(2, 1))
    assert {(r[2], r[4]) for r in fetch_changes(client)} == {(D1, D2)}

    response = delete_flight(client, D2)
    assert "err=" not in response.headers["location"]
    assert fetch_changes(client) == [
        ["0", "0", D1, "forest", D3, "nonforest"],
        ["1", "0", D1, "forest", D3, "nonforest"],
    ]
    assert db_rows(client, "SELECT * FROM flights WHERE flight_date = ?", (D2,)) == []
    assert db_rows(
        client, "SELECT * FROM flight_observations WHERE flight_date = ?", (D2,)
    ) == []
    # The date's stored files are gone too.
    assert not any((client.app.state.runtime.flights_dir / D2).glob("*"))


def test_delete_unknown_date_is_an_error(client, sample):
    response = delete_flight(client, D1)
    assert "err=" in response.headers["location"]


# --------------------------------------------------------------------------- #
# Atomicity
# --------------------------------------------------------------------------- #


def test_failed_registration_keeps_previous_change_table(client, sample, monkeypatch):
    register(client, D1, frame_bytes(4, 4), land_mask(4, 4, forest=ALL_4X4))
    register(client, D2, frame_bytes(4, 4), land_mask(4, 4))
    csv_before = client.get("/flights/changes.csv").content

    def boom(conn, date, observations):
        raise RuntimeError("simulated recompute failure")

    monkeypatch.setattr(flights, "apply_flight_observations", boom)
    response = register(client, D3, frame_bytes(4, 4), land_mask(4, 4, forest=ALL_4X4))
    assert "err=" in unquote(response.headers["location"])

    # Nothing took effect: same CSV, no flight row, no observations, no files.
    assert client.get("/flights/changes.csv").content == csv_before
    assert db_rows(client, "SELECT * FROM flights WHERE flight_date = ?", (D3,)) == []
    assert db_rows(
        client, "SELECT * FROM flight_observations WHERE flight_date = ?", (D3,)
    ) == []
    assert not any((client.app.state.runtime.flights_dir / D3).glob("*"))
    assert D3 not in client.get("/flights").text

    # The system recovers once the failure is gone.
    monkeypatch.undo()
    response = register(client, D3, frame_bytes(4, 4), land_mask(4, 4, forest=ALL_4X4))
    assert "err=" not in response.headers["location"]
    assert {(r[2], r[4]) for r in fetch_changes(client)} == {(D1, D2), (D2, D3)}


# --------------------------------------------------------------------------- #
# Incremental recompute vs full reference
# --------------------------------------------------------------------------- #


def test_incremental_updates_match_full_recompute(client):
    """Random registers/replacements/deletes always equal a full recompute."""
    rng = random.Random(20261120)
    set_sample(client, 6, 4)
    dates = ["2026-03-01", "2026-03-15", "2026-04-01", "2026-04-20", "2026-05-10"]
    live: list[str] = []
    for step in range(40):
        if live and rng.random() < 0.25:
            date = rng.choice(live)
            assert "err=" not in delete_flight(client, date).headers["location"]
            live.remove(date)
        else:
            date = rng.choice(dates)
            transparent = {
                (rng.randrange(6), rng.randrange(4))
                for _ in range(rng.randrange(0, 10))
            }
            forest = {
                (rng.randrange(6), rng.randrange(4))
                for _ in range(rng.randrange(0, 10))
            }
            response = register(
                client, date,
                frame_bytes(6, 4, transparent=transparent),
                land_mask(6, 4, forest=forest),
            )
            assert "err=" not in response.headers["location"]
            if date not in live:
                live.append(date)
        assert actual_changes(client) == reference_changes(client), f"step {step}"


# --------------------------------------------------------------------------- #
# Pages
# --------------------------------------------------------------------------- #


def test_index_links_to_flights_page(client):
    assert 'href="/flights"' in client.get("/").text
    page = client.get("/flights")
    assert page.status_code == 200
    assert "林地变化监测" in page.text
    assert 'href="/"' in page.text


def test_flights_page_lists_registrations(client, sample):
    register(client, D1, frame_bytes(4, 4), land_mask(4, 4, forest=ALL_4X4))
    page = client.get("/flights").text
    assert D1 in page
    assert "4×4 像元" in page
