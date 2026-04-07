"""
build_db.py — SG Film Ajánló adatbázis-építő
=============================================
Beolvassa a movies_clean.csv-t (vagy movies.csv-t) és SQLite DB-t épít belőle.

Újítások az eredeti verzióhoz képest:
  - Részletes logolás (INFO/WARNING/ERROR)
  - Robusztus CSV-felismerés (eltérő oszlopnevek is működnek)
  - genres és tags normalizálva (kisbetű, pipe-szeparált, duplikátum nélkül)
  - Fallback poster: YouTube keresés helyett üres string (Flask kezeli)
  - Upsert: ha már létezik a film, frissíti az adatait
  - Összefoglaló táblázat a végén
  - Parancssori argumentum: python build_db.py --csv másik.csv --db másik.db

Használat:
    python build_db.py
    python build_db.py --csv movies.csv --db movies.db
"""

from __future__ import annotations

import argparse
import csv
import logging
import os
import sqlite3
import sys
from typing import Any, Dict, List, Optional
from urllib.parse import quote

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
    stream=sys.stdout,
)
log = logging.getLogger("build_db")

# ---------------------------------------------------------------------------
# Alapértelmezett útvonalak
# ---------------------------------------------------------------------------
BASE_DIR  = os.path.dirname(os.path.abspath(__file__))
_CSV_CLEAN = os.path.join(BASE_DIR, "movies_clean.csv")
_CSV_RAW   = os.path.join(BASE_DIR, "movies.csv")
DEFAULT_CSV = _CSV_CLEAN if os.path.exists(_CSV_CLEAN) else _CSV_RAW
DEFAULT_DB  = os.path.join(BASE_DIR, "movies.db")

# ---------------------------------------------------------------------------
# Segédfüggvények
# ---------------------------------------------------------------------------

def detect_delimiter(first_line: str) -> str:
    """Automatikusan felismeri a CSV elválasztóját (,  ;  TAB)."""
    candidates = [",", ";", "\t"]
    counts = {c: first_line.count(c) for c in candidates}
    delim = max(counts, key=counts.get) if first_line else ","
    log.info("CSV elválasztó felismerve: %r", delim)
    return delim


def to_int(value: Any, default: int = 0) -> int:
    """Biztonságos int-konverzió."""
    try:
        return int(float(str(value or "").strip()))
    except (ValueError, TypeError):
        return default


def clean(value: Any) -> str:
    """Strip + None-safe."""
    return str(value or "").strip()


def normalize_pipe_list(value: str) -> str:
    """
    Pipe-szeparált listát normalizál:
      - kisbetű
      - felesleges szóközök eltávolítása
      - duplikátumok kiszűrése
      - abc-sorrendbe rendezés
    Pl.: 'Akció | Dráma | akció' → 'akció|dráma'
    """
    parts = [p.strip().lower() for p in value.split("|") if p.strip()]
    unique = sorted(set(parts))
    return "|".join(unique)


def resolve_poster(title: str, raw: str) -> str:
    """
    Visszaadja a poster URL-t ha érvényes HTTP URL,
    egyébként üres stringet (a Flask/frontend kezeli a fallbacket).
    """
    url = clean(raw)
    if url.startswith("http://") or url.startswith("https://"):
        return url
    log.warning("Hiányzó/érvénytelen poster URL: '%s' → üres string", title)
    return ""


def resolve_trailer(title: str, raw: str) -> str:
    """
    Visszaadja a trailer URL-t ha érvényes,
    egyébként YouTube keresési linket generál.
    """
    url = clean(raw)
    if url.startswith("http://") or url.startswith("https://"):
        return url
    fallback = f"https://www.youtube.com/results?search_query={quote(title + ' trailer')}"
    log.warning("Hiányzó trailer: '%s' → fallback YouTube keresés", title)
    return fallback


# ---------------------------------------------------------------------------
# CSV-olvasó: rugalmas oszlopnév-felismerés
# ---------------------------------------------------------------------------

# Ha a CSV-ben eltérő oszlopnevek szerepelnek, ezek is felismert aliasok
_ALIASES: Dict[str, List[str]] = {
    "title":         ["title", "Title", "cím", "film"],
    "year":          ["year", "Year", "év"],
    "minutes":       ["minutes", "Minutes", "perc", "length", "runtime"],
    "genres":        ["genres", "Genres", "műfaj", "genre"],
    "tags":          ["tags", "Tags", "tag", "kulcsszó"],
    "why":           ["why", "Why", "miért", "description", "desc"],
    "certification": ["certification", "Certification", "korhatár", "rating"],
    "trailer":       ["trailer", "Trailer", "trailer_url"],
    "poster":        ["poster", "Poster", "poster_url", "image", "kép"],
}

def _resolve_field(row: Dict[str, str], field: str) -> str:
    """Megkeresi a mező értékét az ismert aliasok alapján."""
    for alias in _ALIASES.get(field, [field]):
        if alias in row and row[alias] is not None:
            return row[alias]
    return ""


def load_movies_from_csv(path: str) -> List[Dict[str, Any]]:
    """Beolvassa a CSV-t és visszaadja a filmek listáját dict formában."""
    if not os.path.exists(path):
        log.error("CSV nem található: %s", path)
        raise FileNotFoundError(f"CSV nem található: {path}")

    movies: List[Dict[str, Any]] = []
    skipped = 0

    with open(path, "r", encoding="utf-8-sig", errors="replace", newline="") as f:
        first_line = f.readline()
        f.seek(0)
        delim = detect_delimiter(first_line)
        reader = csv.DictReader(f, delimiter=delim)

        # Ellenőrizzük az oszlopneveket
        if reader.fieldnames:
            log.info("CSV oszlopok: %s", list(reader.fieldnames))
        else:
            log.warning("Nem sikerült felismerni az oszlopneveket!")

        for line_no, row in enumerate(reader, start=2):
            title = clean(_resolve_field(row, "title"))
            if not title:
                skipped += 1
                log.debug("Üres title, sor kihagyva: %d", line_no)
                continue

            year    = to_int(_resolve_field(row, "year"))
            minutes = to_int(_resolve_field(row, "minutes"))
            genres  = normalize_pipe_list(_resolve_field(row, "genres"))
            tags    = normalize_pipe_list(_resolve_field(row, "tags"))
            why     = clean(_resolve_field(row, "why"))
            cert    = clean(_resolve_field(row, "certification"))
            trailer = resolve_trailer(title, _resolve_field(row, "trailer"))
            poster  = resolve_poster(title, _resolve_field(row, "poster"))

            # Alapvető validáció
            if year < 1888 or year > 2100:
                log.warning("Furcsa év '%s': %s — megtartjuk, de ellenőrizd!", title, year)
            if minutes < 1 or minutes > 600:
                log.warning("Furcsa hossz '%s': %s perc — megtartjuk, de ellenőrizd!", title, minutes)

            movies.append({
                "title":         title,
                "year":          year,
                "minutes":       minutes,
                "genres":        genres,
                "tags":          tags,
                "why":           why,
                "certification": cert,
                "trailer":       trailer,
                "poster":        poster,
            })

    log.info("CSV beolvasva: %d film, %d sor kihagyva", len(movies), skipped)
    return movies


# ---------------------------------------------------------------------------
# DB-építő
# ---------------------------------------------------------------------------

_DDL = """
CREATE TABLE IF NOT EXISTS movies (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    title         TEXT    NOT NULL,
    year          INTEGER NOT NULL DEFAULT 0,
    minutes       INTEGER NOT NULL DEFAULT 0,
    genres        TEXT    NOT NULL DEFAULT '',
    tags          TEXT    NOT NULL DEFAULT '',
    why           TEXT    NOT NULL DEFAULT '',
    certification TEXT    NOT NULL DEFAULT '',
    trailer       TEXT    NOT NULL DEFAULT '',
    poster        TEXT    NOT NULL DEFAULT '',
    tmdb_id       TEXT    NOT NULL DEFAULT '',
    UNIQUE(title, year)
);
CREATE INDEX IF NOT EXISTS idx_title   ON movies(title);
CREATE INDEX IF NOT EXISTS idx_year    ON movies(year);
CREATE INDEX IF NOT EXISTS idx_minutes ON movies(minutes);
"""

_UPSERT = """
INSERT INTO movies (title, year, minutes, genres, tags, why, certification, trailer, poster, tmdb_id)
VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, '')
ON CONFLICT(title, year) DO UPDATE SET
    minutes       = excluded.minutes,
    genres        = excluded.genres,
    tags          = excluded.tags,
    why           = excluded.why,
    certification = excluded.certification,
    trailer       = excluded.trailer,
    poster        = CASE WHEN movies.poster LIKE 'https://image.tmdb.org/%'
                         THEN movies.poster ELSE excluded.poster END
"""


def build_db(movies: List[Dict[str, Any]], db_path: str) -> Dict[str, Any]:
    """
    Felépíti (vagy frissíti) az SQLite adatbázist.
    Upsert logikával dolgozik: ha már létezik a film (title+year),
    frissíti az adatait, nem dob hibát.
    """
    is_new = not os.path.exists(db_path)
    log.info("%s DB: %s", "Új" if is_new else "Meglévő frissítése:", db_path)

    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA journal_mode=WAL")  # párhuzamos olvasás
    conn.execute("PRAGMA foreign_keys=ON")
    cur = conn.cursor()

    # Tábla + indexek létrehozása (ha még nem léteznek)
    for stmt in _DDL.strip().split(";"):
        stmt = stmt.strip()
        if stmt:
            cur.execute(stmt)

    inserted = updated = errors = 0

    for m in movies:
        try:
            # Lekérjük az előző rowcount-ot hogy tudjuk insert vagy update lett-e
            cur.execute(
                "SELECT id FROM movies WHERE title=? AND year=?",
                (m["title"], m["year"])
            )
            exists = cur.fetchone() is not None

            cur.execute(_UPSERT, (
                m["title"], m["year"], m["minutes"],
                m["genres"], m["tags"], m["why"],
                m["certification"], m["trailer"], m["poster"],
            ))

            if exists:
                updated += 1
            else:
                inserted += 1

        except sqlite3.Error as exc:
            errors += 1
            log.error("DB hiba '%s': %s", m.get("title", "?"), exc)

    conn.commit()

    # Összefoglaló lekérdezések
    cur.execute("SELECT COUNT(*) FROM movies")
    total = cur.fetchone()[0]

    cur.execute("SELECT COUNT(*) FROM movies WHERE poster != ''")
    with_poster = cur.fetchone()[0]

    cur.execute("SELECT COUNT(*) FROM movies WHERE poster = ''")
    no_poster = cur.fetchone()[0]

    cur.execute("SELECT COUNT(*) FROM movies WHERE tmdb_id != ''")
    with_tmdb_id = cur.fetchone()[0]

    cur.execute("SELECT title, year, minutes, poster FROM movies ORDER BY title LIMIT 5")
    sample = cur.fetchall()

    conn.close()

    return {
        "inserted":    inserted,
        "updated":     updated,
        "errors":      errors,
        "total":       total,
        "with_poster": with_poster,
        "no_poster":   no_poster,
        "with_tmdb_id": with_tmdb_id,
        "sample":      sample,
    }


# ---------------------------------------------------------------------------
# Összefoglaló kiírás
# ---------------------------------------------------------------------------

def print_summary(result: Dict[str, Any], db_path: str) -> None:
    sep = "─" * 50
    print(f"\n{sep}")
    print("  ✅  KÉSZ — SG Film DB")
    print(sep)
    print(f"  DB helye      : {db_path}")
    print(f"  Összes film   : {result['total']}")
    print(f"  Új bejegyzés  : {result['inserted']}")
    print(f"  Frissített    : {result['updated']}")
    print(f"  Hibás sorok   : {result['errors']}")
    print(f"  Van posztere  : {result['with_poster']}")
    print(f"  Nincs posztere: {result['no_poster']}")
    print(f"  Van TMDB ID   : {result.get('with_tmdb_id', '?')}")
    print(sep)
    print("  Minta (első 5 film):")
    for row in result["sample"]:
        poster_ok = "✅" if row[3] else "❌"
        print(f"    {poster_ok} {row[0]} ({row[1]}) — {row[2]} perc")
    print(sep + "\n")


# ---------------------------------------------------------------------------
# Belépési pont
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="SG Film Ajánló — CSV → SQLite adatbázis-építő"
    )
    parser.add_argument(
        "--csv", default=DEFAULT_CSV,
        help=f"CSV fájl útvonala (alapértelmezett: {DEFAULT_CSV})"
    )
    parser.add_argument(
        "--db", default=DEFAULT_DB,
        help=f"SQLite DB útvonala (alapértelmezett: {DEFAULT_DB})"
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true",
        help="Részletesebb logolás (DEBUG szint)"
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    log.info("CSV: %s", args.csv)
    log.info("DB:  %s", args.db)

    try:
        movies = load_movies_from_csv(args.csv)
    except FileNotFoundError as exc:
        log.error("%s", exc)
        sys.exit(1)

    if not movies:
        log.error("Egy film sem töltődött be a CSV-ből!")
        sys.exit(1)

    result = build_db(movies, args.db)
    print_summary(result, args.db)


if __name__ == "__main__":
    main()