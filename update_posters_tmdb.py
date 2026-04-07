import os
import sqlite3
import time
import requests

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE_DIR, "movies.db")

TMDB_API_KEY = "93b61916ec60f44d3893982847e56c7e"

TMDB_SEARCH_URL = "https://api.themoviedb.org/3/search/movie"
TMDB_IMAGE_BASE = "https://image.tmdb.org/t/p/w500"


def clean_title(title: str) -> str:
    return (title or "").strip()


def get_poster_from_tmdb(title: str, year: int | None = None) -> tuple[str, str]:
    """
    Visszaad:
    - poster_url
    - tmdb_id
    """
    params = {
        "api_key": TMDB_API_KEY,
        "query": title,
        "include_adult": "false",
    }

    if year and year > 0:
        params["year"] = year

    try:
        r = requests.get(TMDB_SEARCH_URL, params=params, timeout=15)
        r.raise_for_status()
        data = r.json()
    except Exception as e:
        print(f"HIBA lekérésnél: {title} -> {e}")
        return "", ""

    results = data.get("results", [])
    if not results:
        return "", ""

    best = results[0]
    poster_path = (best.get("poster_path") or "").strip()
    tmdb_id = str(best.get("id") or "").strip()

    if not poster_path:
        return "", tmdb_id

    return TMDB_IMAGE_BASE + poster_path, tmdb_id


def main():
    if not os.path.exists(DB_PATH):
        print("Nincs movies.db:", DB_PATH)
        return

    if not TMDB_API_KEY or TMDB_API_KEY == "IDE_JON_A_KULCSOD":
        print("Írd be a TMDB API kulcsodat a script tetején.")
        return

    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()

    cur.execute("""
        SELECT id, title, year, poster, tmdb_id
        FROM movies
    """)
    rows = cur.fetchall()

    updated = 0
    skipped = 0
    failed = 0

    for row in rows:
        movie_id = row["id"]
        title = clean_title(row["title"])
        year = int(row["year"] or 0)
        current_poster = (row["poster"] or "").strip()

        # Ha már tmdb-s poszter van, skip
        if current_poster.startswith("https://image.tmdb.org/"):
            skipped += 1
            continue

        print(f"Keresés: {title} ({year})")

        poster_url, tmdb_id = get_poster_from_tmdb(title, year)

        if poster_url:
            cur.execute("""
                UPDATE movies
                SET poster = ?, tmdb_id = ?
                WHERE id = ?
            """, (poster_url, tmdb_id, movie_id))
            updated += 1
            print("  -> OK")
        else:
            failed += 1
            print("  -> NINCS POSZTER")

        # Kicsi várakozás, hogy kulturált legyen
        time.sleep(0.2)

    conn.commit()
    conn.close()

    print("\nKÉSZ ✅")
    print("Frissítve:", updated)
    print("Skip:", skipped)
    print("Sikertelen:", failed)


if __name__ == "__main__":
    main()