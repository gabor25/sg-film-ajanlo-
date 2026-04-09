"""
SG Film Ajánló — v2 (improved)
================================
A Hungarian AI movie recommender powered by Flask.
Supports offline NLU + optional OpenAI profile extraction.
 
Run:
    pip install flask openai   # openai is optional
    python sg_film_ajanlo.py
 
Config via .env:
    OPENAI_API_KEY=sk-...
    OPENAI_MODEL=gpt-4o-mini   (default)
    FLASK_SECRET=your-secret   (strongly recommended)
    FLASK_PORT=5000
    FLASK_DEBUG=false
"""
 
from __future__ import annotations
 
import csv
import json
import logging
import os
import random
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
 
from flask import Flask, jsonify, request, session
 
# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("sg_film")
 
# ---------------------------------------------------------------------------
# Minimal .env loader (no extra dependencies)
# ---------------------------------------------------------------------------
def _load_dotenv(path: str = ".env") -> None:
    if not os.path.exists(path):
        return
    try:
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, _, val = line.partition("=")
                val = val.strip().strip('"').strip("'")
                os.environ.setdefault(key.strip(), val)
    except OSError as exc:
        log.warning("Could not read .env: %s", exc)
 
_load_dotenv()
 
# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
class Config:
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "").strip()
    OPENAI_MODEL: str   = os.getenv("OPENAI_MODEL", "gpt-4o-mini").strip()
    FLASK_SECRET: str   = os.getenv("FLASK_SECRET", "").strip()
    FLASK_PORT: int     = int(os.getenv("FLASK_PORT", "5000"))
    FLASK_DEBUG: bool   = os.getenv("FLASK_DEBUG", "false").lower() == "true"
    SITE_PASSWORD: str  = os.getenv("SITE_PASSWORD", "").strip()
 
    # Scoring weights
    TIME_CLOSE_10  = 10
    TIME_CLOSE_25  =  7
    TIME_CLOSE_45  =  4
    TIME_CLOSE_80  =  1
    TIME_FAR_PENALTY = -2
    MOOD_MATCH     =  3
    BRAIN_MATCH    =  4
    BRAIN_PENALTY  = -2
    KEYWORD_MATCH  =  2
    RANDOM_BONUS   =  2   # randint(0, RANDOM_BONUS)
 
if not Config.FLASK_SECRET:
    log.warning(
        "FLASK_SECRET is not set — using an insecure fallback. "
        "Set FLASK_SECRET in your .env for production."
    )
 
# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------
@dataclass
class Movie:
    title:         str
    year:          int
    minutes:       int
    genres:        List[str]
    tags:          List[str]
    why:           str
    poster:        str = ""   # image URL
    trailer:       str = ""   # YouTube URL
    certification: str = ""   # e.g. "12", "16", "18", "PG-13"
    avg_rating:    float = 0.0  # MovieLens átlagértékelés
    rating_count:  int   = 0    # értékelések száma
    tmdb_id:       str = ""     # TMDB azonosító
 
# ---------------------------------------------------------------------------
# CSV loading
# ---------------------------------------------------------------------------
BASE_DIR  = os.path.dirname(os.path.abspath(__file__))
CSV_CLEAN = os.path.join(BASE_DIR, "movies_clean.csv")
CSV_RAW   = os.path.join(BASE_DIR, "movies.csv")
CSV_PATH  = CSV_CLEAN if os.path.exists(CSV_CLEAN) else CSV_RAW
 
_FIELD_ALIASES: Dict[str, List[str]] = {
    "title":         ["title", "Title"],
    "year":          ["year", "Year"],
    "minutes":       ["minutes", "Minutes"],
    "genres":        ["genres", "Genres"],
    "tags":          ["tags", "Tags"],
    "why":           ["why", "Why"],
    "poster":        ["poster"],
    "trailer":       ["trailer"],
    "certification": ["certification", "Certification"],
}
 
def _first(row: Dict[str, str], *keys: str) -> str:
    """Return the first non-empty value from a CSV row by key priority."""
    for k in keys:
        v = row.get(k)
        if v:
            return v.strip()
    return ""
 
def _to_int(value: Any, default: int = 0) -> int:
    try:
        s = str(value).strip()
        return int(float(s)) if s else default
    except (ValueError, TypeError):
        return default
 
def _split_pipe(value: Any) -> List[str]:
    return [p.strip().lower() for p in str(value or "").split("|") if p.strip()]
 
def _detect_delimiter(first_line: str) -> str:
    candidates = [";", ",", "\t"]
    counts = {c: first_line.count(c) for c in candidates}
    return max(counts, key=counts.get) if first_line else ","
 
def load_movies_csv(path: str) -> List[Movie]:
    movies: List[Movie] = []
    if not os.path.exists(path):
        log.error("CSV file not found: %s", path)
        return movies
 
    skipped = 0
    with open(path, encoding="utf-8-sig", errors="replace", newline="") as f:
        first_line = f.readline()
        f.seek(0)
        reader = csv.DictReader(f, delimiter=_detect_delimiter(first_line))
        for line_no, row in enumerate(reader, start=2):
            try:
                title = _first(row, "title", "Title")
                if not title:
                    skipped += 1
                    continue
                movies.append(Movie(
                    title         = title,
                    year          = _to_int(_first(row, "year", "Year")),
                    minutes       = _to_int(_first(row, "minutes", "Minutes")),
                    genres        = _split_pipe(_first(row, "genres", "Genres")),
                    tags          = _split_pipe(_first(row, "tags", "Tags")),
                    why           = _first(row, "why", "Why"),
                    poster        = _first(row, "poster").strip('"').strip("'"),
                    trailer       = _first(row, "trailer").strip('"').strip("'"),
                    certification = _first(row, "certification", "Certification"),
                ))
            except Exception as exc:
                skipped += 1
                if skipped <= 5:
                    log.warning("Skipping row %d: %s | keys=%s", line_no, exc, list(row.keys()))
 
    log.info("Loaded %d movies, skipped %d — from %s", len(movies), skipped, path)
    return movies
 
 
# ---------------------------------------------------------------------------
# DB loading (SQLite — elsődleges forrás, CSV fallback)
# ---------------------------------------------------------------------------
import sqlite3
 
DB_PATH = os.path.join(BASE_DIR, "movies.db")
 
_SELECT_SQL = """
    SELECT
        title, year, minutes,
        genres, tags, why,
        poster, trailer, certification,
        COALESCE(avg_rating, 0.0) as avg_rating,
        COALESCE(rating_count, 0) as rating_count,
        COALESCE(tmdb_id, '') as tmdb_id
    FROM movies
    ORDER BY title
"""
 
def _split_pipe_db(value) -> List[str]:
    return [p.strip().lower() for p in (value or "").split("|") if p.strip()]
 
def load_movies_db(path: str = DB_PATH) -> List[Movie]:
    movies: List[Movie] = []
    if not os.path.exists(path):
        log.warning("DB nem található: %s — CSV fallbackre váltok", path)
        return movies
    try:
        conn = sqlite3.connect(path)
        conn.row_factory = sqlite3.Row
    except sqlite3.Error as exc:
        log.error("Nem sikerült megnyitni a DB-t: %s — %s", path, exc)
        return movies
 
    skipped = 0
    try:
        with conn:
            rows = conn.execute(_SELECT_SQL).fetchall()
        for row in rows:
            try:
                movies.append(Movie(
                    title         = (row["title"]         or "").strip(),
                    year          = int(row["year"]        or 0),
                    minutes       = int(row["minutes"]     or 0),
                    genres        = _split_pipe_db(row["genres"]),
                    tags          = _split_pipe_db(row["tags"]),
                    why           = (row["why"]            or "").strip(),
                    poster        = (row["poster"]         or "").strip(),
                    trailer       = (row["trailer"]        or "").strip(),
                    certification = (row["certification"]  or "").strip(),
                    avg_rating    = float(row["avg_rating"]   or 0.0),
                    rating_count  = int(row["rating_count"]   or 0),
                    tmdb_id       = (row["tmdb_id"]           or "").strip(),
                ))
            except Exception as exc:
                skipped += 1
                log.warning("Hibás DB sor kihagyva ('%s'): %s", row["title"], exc)
    except sqlite3.Error as exc:
        log.error("DB lekérdezési hiba: %s", exc)
        movies = []
    finally:
        conn.close()
 
    log.info(
        "DB-ből betöltve: %d film%s — %s",
        len(movies),
        f", {skipped} kihagyva" if skipped else "",
        path,
    )
    return movies
 
MOVIES: List[Movie] = load_movies_db(DB_PATH)
if not MOVIES:
    log.warning("DB üres vagy nem elérhető — CSV-ből töltök be")
    MOVIES = load_movies_csv(CSV_PATH)
 
# ---------------------------------------------------------------------------
# Offline NLU
# ---------------------------------------------------------------------------
MOOD_SYNONYMS: Dict[str, List[str]] = {
    "porgos":      ["pörg", "akció", "gyors", "üldöz", "harc", "bosszú", "adrenalin", "darál"],
    "nyugis":      ["nyugis", "chill", "feel", "laza", "szívmelenget", "romi", "romant"],
    "sotet":       ["söt", "sot", "thriller", "krimi", "parás", "nyomaszt", "gyilk", "pszich"],
    "felemelo":    ["felem", "motiv", "inspir", "pozit", "kitart", "remény", "remeny"],
    "vicces":      ["vicc", "kom", "nevet", "őrült", "orult", "paród", "parod", "humor"],
    "romantic":    ["roman", "szerel", "romantic", "love", "randi", "pár", "par"],
}
 
BRAIN_SYNONYMS: Dict[str, List[str]] = {
    "konnyu":           ["könny", "konny", "laza", "agykikapcs", "nem akarok gondolkodni", "egyszerű", "egyszeru"],
    "kozepes":          ["közep", "kozep", "normál", "normal"],
    "elgondolkodtato":  ["elgondolk", "agyas", "agyal", "csavaros", "pszich", "bonyolult", "twist"],
}
 
_STOP_WORDS = {
    "legyen", "valami", "film", "néznék", "neznek", "néznek",
    "ma", "most", "kell", "akarok", "szeretnék", "szeretnek",
}
 
def extract_time(text: str) -> Optional[int]:
    t = (text or "").lower()
 
    m = re.search(r"(\d{2,3})\s*(perc|p)\b", t)
    if m:
        v = int(m.group(1))
        if 60 <= v <= 240:
            return v
 
    m = re.search(r"(\d+(?:[.,]\d+)?)\s*ó", t)
    if m:
        hours = float(m.group(1).replace(",", "."))
        v = int(round(hours * 60))
        if 60 <= v <= 240:
            return v
 
    if "másfél" in t or "masfel" in t:
        return 90
    if any(x in t for x in ("két óra", "2 óra", "2 ora", "120")):
        return 120
    if any(x in t for x in ("három óra", "3 óra", "3 ora", "180")):
        return 180
    return None
 
def _best_match(text: str, synonyms: Dict[str, List[str]]) -> Optional[str]:
    t = (text or "").lower()
    best, best_score = None, 0
    for key, keywords in synonyms.items():
        score = sum(1 for kw in keywords if kw in t)
        if score > best_score:
            best_score, best = score, key
    return best if best_score > 0 else None
 
def extract_mood(text: str) -> Optional[str]:
    return _best_match(text, MOOD_SYNONYMS)
 
def extract_brain(text: str) -> Optional[str]:
    return _best_match(text, BRAIN_SYNONYMS)
 
def extract_keywords(text: str) -> str:
    t = (text or "").lower().strip()
    tokens = [x for x in re.split(r"[^a-záéíóöőúüű0-9]+", t) if x]
    tokens = [x for x in tokens if x not in _STOP_WORDS and len(x) >= 3]
    return " ".join(tokens[:10])
 
# ---------------------------------------------------------------------------
# Optional OpenAI NLU
# ---------------------------------------------------------------------------
_NLU_SYSTEM_PROMPT = (
    "Te egy magyar NLU (szövegértelmező) vagy filmes ajánlóhoz. "
    "Feladat: a felhasználó üzenetéből kinyerni a profilt. "
    "CSAK érvényes JSON-t adj vissza, semmi mást.\n"
    "Kimenet séma:\n"
    '{"mood": "porgos|nyugis|sotet|felemelo|vicces|null", '
    '"time": 90|120|180|egyéb perc (60-240)|null, '
    '"brain": "konnyu|kozepes|elgondolkodtato|null", '
    '"q": "kulcsszavak röviden"}\n'
    "Ha nincs információ, null. A q legyen rövid (max ~8 szó)."
)
 
def openai_nlu_profile(user_msg: str) -> Optional[Dict[str, Any]]:
    if not Config.OPENAI_API_KEY:
        return None
    try:
        from openai import OpenAI  # type: ignore
    except ImportError:
        return None
 
    client = OpenAI(api_key=Config.OPENAI_API_KEY)
    try:
        resp = client.chat.completions.create(
            model=Config.OPENAI_MODEL,
            messages=[
                {"role": "system", "content": _NLU_SYSTEM_PROMPT},
                {"role": "user",   "content": user_msg},
            ],
            temperature=0.2,
            max_tokens=120,
        )
        raw = (resp.choices[0].message.content or "").strip()
        try:
            obj = json.loads(raw)
        except json.JSONDecodeError:
            start, end = raw.find("{"), raw.rfind("}")
            if start != -1 and end > start:
                obj = json.loads(raw[start:end + 1])
            else:
                return None
        return obj if isinstance(obj, dict) else None
    except Exception as exc:
        log.debug("OpenAI NLU failed: %s", exc)
        return None
 
# ---------------------------------------------------------------------------
# Recommender scoring
# ---------------------------------------------------------------------------
MOOD_KEYWORDS: Dict[str, List[str]] = {
    "porgos":    ["action", "adventure", "thriller", "crime", "war",
                  "akció", "gyors", "pörgős", "harc", "bosszú",
                  "chase", "fight", "battle", "explosion", "gun"],
    "nyugis":    ["romance", "comedy", "family", "animation", "musical",
                  "feel-good", "nyugis", "laza", "szívmelengető",
                  "love", "friendship", "heartwarming", "gentle"],
    "sotet":     ["thriller", "horror", "mystery", "crime", "noir",
                  "sötét", "csavaros", "pszichológiai", "krimi",
                  "dark", "psychological", "suspense", "murder", "death"],
    "felemelo":  ["drama", "biography", "sport", "music", "history",
                  "motiváló", "felemelő", "inspiráló", "sport",
                  "inspiring", "triumph", "overcome", "true story"],
    "vicces":    ["comedy", "animation", "family", "parody",
                  "vicces", "komédia", "humor", "funny",
                  "hilarious", "witty", "slapstick", "satire"],
    "romantic":  ["romance", "romantic", "love", "wedding", "relationship",
                  "romantikus", "szerelem", "szerelmes"],
}
 
BRAIN_KEYWORDS: Dict[str, List[str]] = {
    "konnyu":          ["könnyű", "laza", "comedy", "animation", "family",
                        "simple", "fun", "light", "feel-good"],
    "kozepes":         ["közepes", "drama", "adventure", "action"],
    "elgondolkodtato": ["elgondolkodtató", "agyas", "csavaros", "pszichológiai",
                        "psychological", "mystery", "twist", "mind", "complex",
                        "thought-provoking", "philosophical", "sci-fi"],
}
 
def score_movie(m: Movie, mood: str, time_limit: int, brain: str, extra: str) -> int:
    cfg = Config
    score = 0
    blob = f"{m.title} {' '.join(m.tags)} {' '.join(m.genres)} {m.why}".lower()
 
    # Time proximity
    if m.minutes and time_limit:
        diff = abs(m.minutes - time_limit)
        if diff <= 10:      score += cfg.TIME_CLOSE_10
        elif diff <= 25:    score += cfg.TIME_CLOSE_25
        elif diff <= 45:    score += cfg.TIME_CLOSE_45
        elif diff <= 80:    score += cfg.TIME_CLOSE_80
        else:               score += cfg.TIME_FAR_PENALTY
 
    # Mood match
    for word in MOOD_KEYWORDS.get(mood, []):
        if word in blob:
            score += cfg.MOOD_MATCH
 
    # Brain/complexity match
    heavy_words = BRAIN_KEYWORDS["elgondolkodtato"]
    if brain == "konnyu":
        if any(w in blob for w in heavy_words):
            score += cfg.BRAIN_PENALTY
        if "komédia" in blob or "vicces" in blob:
            score += 1
    elif brain == "elgondolkodtato":
        if any(w in blob for w in heavy_words):
            score += cfg.BRAIN_MATCH
 
    # Extra keyword bonus
    if extra:
        for token in [t for t in extra.lower().replace(",", " ").split() if len(t) >= 3]:
            if token in blob:
                score += cfg.KEYWORD_MATCH
 
    # Rating bonus — népszerű és jól értékelt filmek előnybe kerülnek
    if m.avg_rating >= 4.0 and m.rating_count >= 50:
        score += 3
    elif m.avg_rating >= 3.5 and m.rating_count >= 20:
        score += 1
 
    score += random.randint(0, cfg.RANDOM_BONUS)
    return score
 
# ---------------------------------------------------------------------------
# Batch AI "miért ajánljuk" generátor
# ---------------------------------------------------------------------------
_why_cache: Dict[str, str] = {}
 
def batch_generate_why(
    movies: List[Movie],
    mood: str,
    brain: str,
    extra: str,
) -> Dict[str, str]:
    """
    Egy OpenAI hívással generál egyedi 'miért?' szöveget az összes filmhez.
    Visszaad: {film_kulcs: szöveg} dict-et.
    Ha nincs API kulcs → szabályalapú fallback.
    """
    results: Dict[str, str] = {}
 
    # Cache-ből kivesszük amit már tudunk
    to_generate = []
    for m in movies:
        key = f"{m.title}|{m.year}|{mood}|{brain}|{extra[:20]}"
        if key in _why_cache:
            results[key] = _why_cache[key]
        else:
            to_generate.append((key, m))
 
    if not to_generate:
        return results
 
    # Szabályalapú fallback ha nincs API kulcs
    if not Config.OPENAI_API_KEY:
        for key, m in to_generate:
            text = _why_rules(m, mood, brain, extra)
            _why_cache[key] = text
            results[key] = text
        return results
 
    # Batch AI generálás — egy hívással az összes filmhez
    try:
        from openai import OpenAI  # type: ignore
        client = OpenAI(api_key=Config.OPENAI_API_KEY)
 
        mood_hu = {
            "porgos": "pörgős akciós",
            "nyugis": "nyugis chill",
            "sotet": "sötét feszült",
            "felemelo": "felemelő motiváló",
            "vicces": "vicces könnyed",
        }.get(mood, mood)
 
        brain_hu = {
            "konnyu": "könnyed agykikapcsolós",
            "kozepes": "közepes mélységű",
            "elgondolkodtato": "elgondolkodtató agyalós",
        }.get(brain, brain)
 
        # Film lista összeállítása a prompthoz
        film_lista = ""
        for i, (key, m) in enumerate(to_generate, 1):
            tags_str = ", ".join(m.tags[:5]) if m.tags else "-"
            genres_str = ", ".join(m.genres[:3]) if m.genres else "-"
            film_lista += (
                str(i) + ". " + m.title
                + " (" + str(m.year) + ") | mufaj: " + genres_str
                + " | tagek: " + tags_str
                + " | rating: " + str(round(m.avg_rating, 1)) + "/5\n"
            )
 
        extra_part = (", kulcsszavak: " + extra) if extra else ""
        prompt = (
            "Te egy baratsagos, emberi hangvételu magyar filmes ajánló vagy.\n"
            + "A néző hangulatkeresése: " + mood_hu
            + ", preferencia: " + brain_hu
            + extra_part + "\n\n"
            + "Minden filmhez írj egy SZEMÉLYES, EMBERI hangvételű magyar ajánlót (2-3 mondat).\n"
            + "Fontos szabályok:\n"
            + "- Minden film ajánlója TELJESEN KÜLÖNBÖZŐ legyen!\n"
            + "- Legyen KONKRÉT: utalj a film hangulatára, sztorira, vagy miért illik a nézőnek\n"
            + "- Emberi hangvétel: mintha egy barát ajánlaná, nem egy robot\n"
            + "- Illeszkedjen a néző " + mood_hu + " hangulatához\n"
            + "- Csak magyarul, ne kezdd 'Ez a film' szavakkal\n"
            + "Csak a sorszámot és a szöveget írd!\n\n"
            + "Filmek:\n"
            + film_lista
            + "\nFormátum:\n1. [személyes ajánló]\n2. [személyes ajánló]\nstb."
        )
        resp = client.chat.completions.create(
            model=Config.OPENAI_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.9,
            max_tokens=len(to_generate) * 80 + 100,
        )
 
        raw = (resp.choices[0].message.content or "").strip()
 
        # Válasz feldolgozása
        import re as _re
        lines = raw.strip().splitlines()
        parsed: Dict[int, str] = {}
        for line in lines:
            line = line.strip()
            if not line:
                continue
            m_re = _re.match(r"^(\d+)[.\)]\s*(.+)$", line)
            if m_re:
                idx_num = int(m_re.group(1))
                text = m_re.group(2).strip().strip('"').strip("'")
                if text:
                    parsed[idx_num] = text
 
        # Berakjuk a cache-be és results-ba
        for i, (key, mov) in enumerate(to_generate, 1):
            text = parsed.get(i)
            if not text or len(text) < 4:
                text = _why_rules(mov, mood, brain, extra)
            _why_cache[key] = text
            results[key] = text
 
        log.info("Batch AI why: %d film, %d generált", len(movies), len(to_generate))
 
    except Exception as exc:
        log.warning("Batch AI why failed: %s — fallback szabályokra", exc)
        for key, m in to_generate:
            text = _why_rules(m, mood, brain, extra)
            _why_cache[key] = text
            results[key] = text
 
    return results
 
 
def _why_rules(m: Movie, mood: str, brain: str, extra: str) -> str:
    """Szabályalapú fallback — hosszabb, egyedi magyar szöveg a film adatai alapján."""
    genre_hu = {
        "action": "akció", "thriller": "thriller", "comedy": "vígjáték",
        "drama": "dráma", "romance": "romantikus", "horror": "horror",
        "crime": "krimi", "animation": "animáció", "adventure": "kaland",
        "mystery": "rejtély", "sport": "sport", "biography": "életrajz",
        "family": "családi", "music": "zenés", "sci-fi": "sci-fi",
        "fantasy": "fantasy", "war": "háborús", "history": "történelmi",
        "documentary": "dokumentum",
    }
    mood_hu = {
        "porgos":   "pörgős",
        "nyugis":   "nyugis",
        "sotet":    "sötét, feszült",
        "felemelo": "felemelő",
        "vicces":   "vicces",
        "romantic": "romantikus",
    }.get(mood, "")
 
    sentences = []
    genres_hu = [genre_hu.get(g, g) for g in m.genres[:3]]
    good_tags  = [t for t in m.tags if len(t) > 3][:3]
 
    # 1. mondat — műfaj + hangulat
    if genres_hu and mood_hu:
        sentences.append(
            f"Ha {mood_hu} hangulatot keresel, ez a {', '.join(genres_hu)} film "
            f"tökéletes választás lehet."
        )
    elif genres_hu:
        sentences.append(f"Egy {', '.join(genres_hu)} alkotás {m.year}-ből.")
 
    # 2. mondat — egyedi tagek
    if good_tags:
        sentences.append(
            f"A film témái között szerepel: {', '.join(good_tags)} — "
            f"ezek garantálják az egyedi élményt."
        )
 
    # 3. mondat — értékelés + év
    if m.avg_rating >= 4.0 and m.rating_count >= 30:
        sentences.append(
            f"A nézők ★{m.avg_rating:.1f}/5 átlaggal értékelték "
            f"({m.rating_count} szavazat alapján)."
        )
    elif m.year > 0 and m.year < 1990:
        sentences.append(f"Egy {m.year}-es klasszikus, amely ma is megállja a helyét.")
 
    if not sentences:
        return f"{', '.join(genres_hu) if genres_hu else 'Film'} — ★{m.avg_rating:.1f}/5"
 
    return " ".join(sentences)
 
 
def rank_movies(mood: str, time_limit: int, brain: str, extra: str, offset: int, take: int) -> List[Movie]:
    # Minden filmhez kiszámoljuk a pontszámot
    scored = [
        (score_movie(m, mood, time_limit, brain, extra), m)
        for m in MOVIES
    ]
 
    # Rendezés pontszám szerint
    scored.sort(key=lambda x: x[0], reverse=True)
 
    # Minimum score: legalább 1 pontot kell kapni
    # Ha extra kulcsszó van, szigorúbb szűrés
    if extra and extra.strip():
        min_score = 2
    else:
        min_score = 1
 
    filtered = [m for score, m in scored if score >= min_score]
 
    # Ha túl kevés film maradt, lazítunk a szűrésen
    if len(filtered) < take * 2:
        filtered = [m for score, m in scored if score >= 0]
 
    return filtered[offset : offset + take]
 
# ---------------------------------------------------------------------------
# Session profile helpers
# ---------------------------------------------------------------------------
def default_profile() -> Dict[str, Any]:
    return {"time": None, "mood": None, "brain": None, "extra": "", "ready": False, "history": []}
 
def get_profile() -> Dict[str, Any]:
    p = session.get("profile")
    if not p:
        p = default_profile()
        session["profile"] = p
    return p
 
def missing_fields(p: Dict[str, Any]) -> List[str]:
    return [f for f in ("mood", "time", "brain") if not p.get(f)]
 
def next_question(p: Dict[str, Any]) -> Tuple[str, List[str]]:
    missing = missing_fields(p)
    if not missing:
        return (
            'Oké, megvan minden. Finomítsunk: írj 1-2 kulcsszót (pl. maffia / űr / csavaros / bosszú) vagy nyomj „Ajánlj"-t.',
            ["Ajánlj", "Újra dobás", "Sötétebb", "Viccesebb", "Rövidebb", "Reset"],
        )
    dispatch = {
        "mood":  ("Milyen hangulatot szeretnél? (pörgős / nyugis / sötét / felemelő / vicces)",
                  ["pörgős", "nyugis", "sötét", "felemelő", "vicces"]),
        "time":  ("Mennyi időd van ma filmre? (90 / 120 / 180 perc)",
                  ["90 perc", "120 perc", "180 perc"]),
        "brain": ("Mennyire legyen elgondolkodtató? (könnyű / közepes / elgondolkodtató)",
                  ["könnyű", "közepes", "elgondolkodtató"]),
    }
    return dispatch.get(missing[0], ("Oké.", []))
 
# ---------------------------------------------------------------------------
# Optional OpenAI chat reply
# ---------------------------------------------------------------------------
_CHAT_SYSTEM_PROMPT = (
    "Te egy magyar nyelvű filmes ajánló asszisztens vagy (SG Film Ajánló). "
    "Cél: barátságos, rövid, mégis 'AI-s' beszélgetés. "
    "NE ajánlj rögtön filmeket 'szia' üzenetre. Először hiányzó adatokat kérdezz ki: "
    "idő (90/120/180), hangulat, elgondolkodtatóság. "
    "Ha már mind megvan, kérj extra kulcsszót, vagy ajánlj. "
    "Stílus: sötét elegancia, laza, motiváló, 1-2 emoji max. "
    "Soha ne állítsd, hogy stream oldalak vagytok."
)
 
def openai_chat_reply(p: Dict[str, Any], user_msg: str) -> Optional[str]:
    if not Config.OPENAI_API_KEY:
        return None
    try:
        from openai import OpenAI  # type: ignore
    except ImportError:
        return None
 
    client = OpenAI(api_key=Config.OPENAI_API_KEY)
    history = p.get("history") or []
    messages = [{"role": "system", "content": _CHAT_SYSTEM_PROMPT}]
    for entry in history[-12:]:
        if entry.get("role") in ("user", "assistant") and entry.get("content"):
            messages.append({"role": entry["role"], "content": entry["content"]})
 
    profile_hint = (
        f"Jelenlegi profil: time={p.get('time')}, mood={p.get('mood')}, "
        f"brain={p.get('brain')}, extra='{p.get('extra', '')}'."
    )
    messages.append({"role": "system", "content": profile_hint})
    messages.append({"role": "user",   "content": user_msg})
 
    try:
        resp = client.chat.completions.create(
            model=Config.OPENAI_MODEL,
            messages=messages,
            temperature=0.6,
            max_tokens=120,
        )
        return (resp.choices[0].message.content or "").strip() or None
    except Exception as exc:
        log.debug("OpenAI chat failed: %s", exc)
        return None
 
# ---------------------------------------------------------------------------
# Flask application
# ---------------------------------------------------------------------------
app = Flask(__name__)
app.secret_key = Config.FLASK_SECRET or "SG_INSECURE_FALLBACK_CHANGE_ME"
 
# ---------------------------------------------------------------------------
# Jelszóvédelem (opcionális — ha SITE_PASSWORD be van állítva a .env-ben)
# ---------------------------------------------------------------------------
@app.before_request
def require_password():
    if not Config.SITE_PASSWORD:
        return  # nincs jelszó beállítva → szabad hozzáférés
 
    # Statikus erőforrások és a login végpont mindig szabad
    if request.path in ("/login", "/logout"):
        return
 
    # Ha már be van jelentkezve
    if session.get("auth") == Config.SITE_PASSWORD:
        return
 
    # API hívások → 401
    if request.path.startswith("/api/"):
        return jsonify({"error": "Unauthorized"}), 401
 
    # Böngésző → login oldal
    return _login_page()
 
@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        pw = request.form.get("password", "").strip()
        if pw == Config.SITE_PASSWORD:
            session["auth"] = pw
            return '<script>location.href="/"</script>'
        return _login_page(error=True)
    return _login_page()
 
@app.route("/logout")
def logout():
    session.pop("auth", None)
    return '<script>location.href="/login"</script>'
 
def _login_page(error: bool = False) -> str:
    err_html = '<p style="color:#c84b4b;font-size:13px;margin-top:8px">Hibás jelszó!</p>' if error else ''
    return f"""<!doctype html>
<html lang="hu">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width,initial-scale=1"/>
<title>SG Film Ajánló — Belépés</title>
<style>
  body {{background:#080a0d;display:flex;align-items:center;justify-content:center;
        min-height:100vh;font-family:system-ui,sans-serif;color:#dde4ed;margin:0}}
  .box {{border:1px solid #2a3441;border-radius:16px;padding:40px 36px;
         background:#0e1218;width:320px;text-align:center;box-shadow:0 20px 60px rgba(0,0,0,.6)}}
  h2 {{font-size:22px;color:#c8a84b;margin-bottom:8px}}
  p  {{color:#7a8a9a;font-size:13px;margin-bottom:24px}}
  input {{width:100%;padding:11px 14px;border-radius:10px;border:1px solid #2a3441;
          background:#080a0d;color:#dde4ed;font-size:14px;outline:none;
          box-sizing:border-box;margin-bottom:12px}}
  input:focus {{border-color:#c8a84b}}
  button {{width:100%;padding:11px;border-radius:10px;border:none;
           background:#c8a84b;color:#080a0d;font-weight:700;font-size:14px;
           cursor:pointer}}
  button:hover {{background:#e8c86c}}
</style>
</head>
<body>
<div class="box">
  <h2>🎬 SG Film Ajánló</h2>
  <p>Tesztelés alatt — jelszó szükséges</p>
  <form method="post" action="/login">
    <input type="password" name="password" placeholder="Jelszó" autofocus/>
    <button type="submit">Belépés</button>
    {err_html}
  </form>
</div>
</body>
</html>"""
 
_GREETINGS = {"szia", "helo", "hello", "helló", "csá", "csa", "szevasz", "sziasztok", "hi"}
_RESET_CMDS = {"reset", "uj", "új", "kezdjük újra", "restart"}
 
# ---------------------------------------------------------------------------
# API: /api/trailer — YouTube trailer keresés (első találat video ID)
# ---------------------------------------------------------------------------
_trailer_cache: Dict[str, str] = {}
 
@app.get("/api/trailer")
def api_trailer():
    import urllib.request as _ureq, json as _json
    from urllib.parse import quote as _quote
    title    = request.args.get("title", "").strip()
    year     = request.args.get("year", "").strip()
    tmdb_id  = request.args.get("tmdb_id", "").strip()
    if not title:
        return jsonify({"url": ""}), 400
 
    cache_key = f"{title}_{year}_{tmdb_id}"
    if cache_key in _trailer_cache:
        return jsonify({"url": _trailer_cache[cache_key]})
 
    # 1. TMDB API-val keressük a YouTube trailer ID-t (ha van kulcs és tmdb_id)
    tmdb_key = os.getenv("TMDB_API_KEY", "").strip()
    if tmdb_key and tmdb_id:
        try:
            api_url = (
                "https://api.themoviedb.org/3/movie/"
                + tmdb_id
                + "/videos?api_key=" + tmdb_key
                + "&language=en-US"
            )
            req = _ureq.Request(api_url, headers={"User-Agent": "Mozilla/5.0"})
            with _ureq.urlopen(req, timeout=8) as resp:
                data = _json.loads(resp.read())
            for v in (data.get("results") or []):
                if v.get("site") == "YouTube" and v.get("type") == "Trailer":
                    yt_url = "https://www.youtube.com/watch?v=" + v["key"]
                    _trailer_cache[cache_key] = yt_url
                    log.info("Trailer found via TMDB: %s", title)
                    return jsonify({"url": yt_url})
            # Ha nincs Trailer, próbáljuk a Teaser-t
            for v in (data.get("results") or []):
                if v.get("site") == "YouTube":
                    yt_url = "https://www.youtube.com/watch?v=" + v["key"]
                    _trailer_cache[cache_key] = yt_url
                    return jsonify({"url": yt_url})
        except Exception as exc:
            log.debug("TMDB trailer lookup failed for %s: %s", title, exc)
 
    # 2. Fallback: YouTube keresési link
    query = (title + " " + year + " official trailer").strip()
    yt_search = "https://www.youtube.com/results?search_query=" + _quote(query)
    _trailer_cache[cache_key] = yt_search
    return jsonify({"url": yt_search})
 
# ---------------------------------------------------------------------------
# API: /api/poster  — TMDB képek proxyzása (böngésző blokk megkerülése)
# ---------------------------------------------------------------------------
@app.get("/api/poster")
def api_poster():
    import urllib.request
    from flask import Response
    url = request.args.get("url", "").strip()
    if not url or not url.startswith("https://image.tmdb.org/"):
        return "Invalid URL", 400
    try:
        req = urllib.request.Request(url, headers={
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
            "Referer": "https://www.themoviedb.org/",
            "Accept": "image/webp,image/apng,image/*,*/*;q=0.8",
        })
        with urllib.request.urlopen(req, timeout=15) as resp:
            data = resp.read()
            content_type = resp.headers.get("Content-Type", "image/jpeg")
        return Response(data, content_type=content_type,
                        headers={"Cache-Control": "public, max-age=86400"})
    except Exception as exc:
        log.debug("Poster proxy failed for %s: %s", url, exc)
        return "Not found", 404
 
# ---------------------------------------------------------------------------
# API: /api/debug
# ---------------------------------------------------------------------------
@app.get("/api/debug")
def api_debug():
    return jsonify({
        "csv_path":         CSV_PATH,
        "csv_exists":       os.path.exists(CSV_PATH),
        "movies_loaded":    len(MOVIES),
        "sample_titles":    [m.title for m in MOVIES[:5]],
        "openai_available": bool(Config.OPENAI_API_KEY),
        "model":            Config.OPENAI_MODEL,
    })
 
# ---------------------------------------------------------------------------
# API: /api/csv_info
# ---------------------------------------------------------------------------
@app.get("/api/csv_info")
def api_csv_info():
    if not os.path.exists(CSV_PATH):
        return jsonify({"ok": False, "error": "CSV not found", "path": CSV_PATH}), 404
    try:
        with open(CSV_PATH, encoding="utf-8-sig", newline="") as f:
            first  = f.readline().strip()
            second = f.readline().strip()
        return jsonify({"ok": True, "csv_path": CSV_PATH, "header": first[:300], "sample": second[:300]})
    except OSError as exc:
        return jsonify({"ok": False, "error": str(exc)}), 500
 
# ---------------------------------------------------------------------------
# API: /api/nlu
# ---------------------------------------------------------------------------
@app.get("/api/nlu")
def api_nlu():
    text = request.args.get("text", "").strip()
    if not text:
        return jsonify({"error": "Missing 'text' parameter"}), 400
    offline = {
        "time":  extract_time(text),
        "mood":  extract_mood(text),
        "brain": extract_brain(text),
        "q":     extract_keywords(text),
    }
    return jsonify({"offline": offline, "openai": openai_nlu_profile(text)})
 
# ---------------------------------------------------------------------------
# API: /api/recs
# ---------------------------------------------------------------------------
@app.get("/api/recs")
def api_recs():
    try:
        mood  = request.args.get("mood", "porgos")
        brain = request.args.get("brain", "konnyu")
        time_limit = int(request.args.get("time", "120"))
        q      = request.args.get("q", "")
        offset = max(0, int(request.args.get("offset", "0")))
        take   = min(24, max(1, int(request.args.get("take", "12"))))
    except (ValueError, TypeError) as exc:
        return jsonify({"error": f"Invalid parameter: {exc}"}), 400
 
    items = rank_movies(mood, time_limit, brain, q, offset, take)
 
    # Batch AI miért generálás — egy hívással az összes filmhez
    why_map = batch_generate_why(items, mood, brain, q)
 
    return jsonify({
        "total":  len(MOVIES),
        "offset": offset,
        "take":   take,
        "items":  [
            {
                "title":         m.title,
                "year":          m.year,
                "minutes":       m.minutes,
                "poster":        m.poster,
                "trailer":       m.trailer,
                "certification": m.certification,
                "why":           why_map.get(
                                     f"{m.title}|{m.year}|{mood}|{brain}|{q[:20]}",
                                     _why_rules(m, mood, brain, q)
                                 ),
                "genres":        m.genres,
                "tags":          m.tags,
                "avg_rating":    m.avg_rating,
                "rating_count":  m.rating_count,
                "tmdb_id":       m.tmdb_id,
            }
            for m in items
        ],
    })
 
# ---------------------------------------------------------------------------
# API: /api/chat
# ---------------------------------------------------------------------------
@app.post("/api/chat")
def api_chat():
    body = request.get_json(force=True, silent=True) or {}
    user_msg = str(body.get("message") or "").strip()
    p = get_profile()
    low = user_msg.lower()
 
    if not user_msg:
        q, quick = next_question(p)
        return jsonify({"assistant": q, "quick_replies": quick, "profile": p})
 
    # Reset
    if low in _RESET_CMDS:
        session["profile"] = default_profile()
        p = get_profile()
        q, quick = next_question(p)
        return jsonify({"assistant": "Oké, tiszta lap. 🙂 " + q, "quick_replies": quick, "profile": p})
 
    # Közvetlen mood beállítás a gombokból
    if user_msg.startswith('__mood__'):
        mood_val = user_msg[8:].strip()
        valid_moods = {"porgos","nyugis","sotet","felemelo","vicces","romantic"}
        if mood_val in valid_moods:
            p["mood"] = mood_val
            if not p.get("ready") and not missing_fields(p):
                p["ready"] = True
            session["profile"] = p
            mood_names = {
                "porgos":   "Pörgős akció hangulat",
                "nyugis":   "Nyugis, chill hangulat",
                "sotet":    "Sötét, feszült hangulat",
                "felemelo": "Felemelő, motiváló hangulat",
                "vicces":   "Vicces, könnyed hangulat",
                "romantic": "Romantikus hangulat",
            }
            ai_text = mood_names.get(mood_val, mood_val) + " — keresem a legjobb filmeket! 🎬"
            _, quick = next_question(p) if missing_fields(p) else (
                None, ["Ajánlj","Újra dobás","Rövidebb","Reset"]
            )
            return jsonify({"assistant": ai_text, "quick_replies": quick, "profile": p})
 
    # Greeting → always reset and re-ask
    if low in _GREETINGS:
        session["profile"] = default_profile()
        p = get_profile()
        q, quick = next_question(p)
        return jsonify({"assistant": "Szia 🙂 " + q, "quick_replies": quick, "profile": p})
 
    # Quick action buttons
    _BUTTON_ACTIONS: Dict[str, Any] = {
        "sötétebb":  lambda: p.update({"mood": "sotet"}),
        "sotetebb":  lambda: p.update({"mood": "sotet"}),
        "viccesebb": lambda: p.update({"mood": "vicces"}),
        "rövidebb":  lambda: p.update({"time": 90}),
        "rovidebb":  lambda: p.update({"time": 90}),
        "ajánlj":    lambda: p.update({"ready": True}),
        "ajanlj":    lambda: p.update({"ready": True}),
        "romantikus": lambda: p.update({"mood": "romantic"}),
        # ── FIX: "újra dobás" button resets offset but keeps profile ready ──
        "újra dobás": lambda: p.update({"ready": True}),
        "ujra dobas": lambda: p.update({"ready": True}),
    }
    if low in _BUTTON_ACTIONS:
        _BUTTON_ACTIONS[low]()
 
    # Offline NLU
    time_val  = extract_time(user_msg)
    mood_val  = extract_mood(user_msg)
    brain_val = extract_brain(user_msg)
    kw        = extract_keywords(user_msg)
 
    # Merge with optional OpenAI NLU
    nlu = openai_nlu_profile(user_msg)
    if nlu:
        try:
            if nlu.get("time") is not None:
                time_val = int(nlu["time"])
        except (ValueError, TypeError):
            pass
        mood_val  = str(nlu["mood"]).strip()  if nlu.get("mood")  else mood_val
        brain_val = str(nlu["brain"]).strip() if nlu.get("brain") else brain_val
        kw        = str(nlu["q"]).strip()     if nlu.get("q")     else kw
 
    if time_val:  p["time"]  = time_val
    if mood_val:  p["mood"]  = mood_val
    if brain_val: p["brain"] = brain_val
 
    # Accumulate extra keywords
    if kw and len(kw) >= 3:
        merged = f"{p.get('extra', '')} {kw}".strip()
        p["extra"] = merged[:240]
 
    # ── FIX: auto-activate ready when all three profile fields are filled ──
    if not p.get("ready") and not missing_fields(p):
        p["ready"] = True
 
    # Build reply text
    ai_text = openai_chat_reply(p, user_msg)
    if not ai_text:
        if missing_fields(p):
            ai_text, _ = next_question(p)
        else:
            ai_text = (
                f"Oké, érzem a vibe-ot. "
                f"(hangulat: {p['mood']}, idő: {p['time']} perc, mód: {p['brain']}). "
                f"Jobbra dobom a poszteres listát — nyomj 'Tölts még'-et is. :)"
            )
 
    # Persist history (keep last 30 turns)
    hist = p.get("history") or []
    hist.extend([
        {"role": "user",      "content": user_msg},
        {"role": "assistant", "content": ai_text},
    ])
    p["history"] = hist[-30:]
    session["profile"] = p
 
    _, quick = next_question(p) if missing_fields(p) else (
        None,
        ["Ajánlj", "Újra dobás", "Sötétebb", "Viccesebb", "Rövidebb", "Reset"],
    )
 
    return jsonify({"assistant": ai_text, "quick_replies": quick, "profile": p})
 
# ---------------------------------------------------------------------------
# API: /api/local_poster — helyi posters/ mappából szolgál ki képeket
# ---------------------------------------------------------------------------
@app.get("/api/local_poster/<filename>")
def api_local_poster(filename):
    import re
    from flask import send_from_directory, Response
    # Csak .jpg/.png fájlokat engedünk
    if not re.match(r'^[a-zA-Z0-9_\-]+\.(jpg|png|jpeg)$', filename):
        return "Invalid", 400
    poster_dir = os.path.join(BASE_DIR, "posters")
    filepath = os.path.join(poster_dir, filename)
    if os.path.exists(filepath):
        return send_from_directory(poster_dir, filename,
                                   max_age=86400)
    # Ha nincs helyi fájl, próbáljuk a proxyn
    return api_poster()
 
# ---------------------------------------------------------------------------
# Frontend — single-page app
# ---------------------------------------------------------------------------
_HTML = r"""<!doctype html>
<html lang="hu">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1, maximum-scale=1"/>
<title>SG Film Ajánló</title>
<link rel="preconnect" href="https://fonts.googleapis.com"/>
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin/>
<link href="https://fonts.googleapis.com/css2?family=DM+Serif+Display:ital@0;1&family=DM+Sans:ital,opsz,wght@0,9..40,300..700;1,9..40,300..700&display=swap" rel="stylesheet"/>
<style>
:root {
  --bg:#080a0d; --surface:#0e1218; --surface2:#131820;
  --border:#1e2730; --border2:#2a3441;
  --text:#dde4ed; --muted:#7a8a9a; --faint:#3a4a5a;
  --gold:#c8a84b; --gold2:#e8c86c; --red:#c84b4b;
  --shadow:0 20px 60px rgba(0,0,0,.6);
  --radius:16px;
  --font-serif:'DM Serif Display',Georgia,serif;
  --font-sans:'DM Sans',system-ui,sans-serif;
}
*,*::before,*::after{box-sizing:border-box;margin:0;padding:0}
html{scroll-behavior:smooth;-webkit-tap-highlight-color:transparent}
body{
  background:var(--bg);color:var(--text);
  font-family:var(--font-sans);font-size:14px;line-height:1.55;
  min-height:100vh;padding:16px 12px 60px;
  background-image:
    radial-gradient(ellipse 900px 500px at 10% 0%,rgba(200,168,75,.06) 0%,transparent 70%),
    radial-gradient(ellipse 700px 400px at 90% 100%,rgba(60,40,20,.15) 0%,transparent 70%);
}
/* Film strip top - eltávolítva */
.wrap{max-width:1160px;margin:0 auto}
 
/* ── Header ── */
.header{
  display:flex;align-items:center;justify-content:space-between;
  gap:12px;flex-wrap:wrap;margin-bottom:20px;padding-bottom:16px;
  border-bottom:1px solid var(--border);
}
.brand{display:flex;align-items:center;gap:12px}
.logo{
  width:44px;height:44px;border-radius:12px;border:1px solid var(--border2);
  background:linear-gradient(145deg,#1a2230,#0a0d12);
  display:flex;align-items:center;justify-content:center;flex-shrink:0;
}
.logo span{font-family:var(--font-serif);font-size:16px;color:var(--gold);letter-spacing:1px}
.brand-text h1{font-family:var(--font-serif);font-size:24px;line-height:1.1;color:var(--text)}
.brand-text h1 em{font-style:italic;color:var(--gold)}
.brand-text p{color:var(--muted);font-size:12px;margin-top:2px}
.badges{display:flex;gap:6px;flex-wrap:wrap;align-items:center}
.badge{
  border:1px solid var(--border);background:rgba(14,18,24,.7);
  border-radius:999px;padding:5px 10px;color:var(--muted);font-size:11px;
}
.badge .hl{color:var(--gold)}
 
/* ── Layout grid ── */
.grid{display:grid;grid-template-columns:1fr 1fr;gap:14px;align-items:start}
@media(max-width:768px){.grid{grid-template-columns:1fr}}
 
/* ── Cards ── */
.card{
  border:1px solid var(--border);background:rgba(14,18,24,.8);
  border-radius:var(--radius);box-shadow:var(--shadow);overflow:hidden;
  backdrop-filter:blur(8px);
}
.card-head{
  padding:12px 16px;border-bottom:1px solid var(--border);
  display:flex;align-items:center;justify-content:space-between;gap:10px;
  background:linear-gradient(180deg,rgba(22,30,42,.9) 0%,rgba(14,18,24,.6) 100%);
}
.card-title{font-family:var(--font-serif);font-size:15px;color:var(--gold)}
.card-body{padding:14px 16px}
 
/* ── Buttons ── */
.btn{
  cursor:pointer;border:1px solid var(--border2);
  background:linear-gradient(145deg,#1c2840,#0c1020);
  color:var(--text);font-family:var(--font-sans);font-weight:600;
  font-size:13px;padding:9px 14px;border-radius:12px;
  transition:border-color .15s,background .15s,transform .1s;
  white-space:nowrap;-webkit-appearance:none;
}
.btn:hover{border-color:var(--gold);background:linear-gradient(145deg,#212e48,#0e1428)}
.btn:active{transform:scale(.97)}
 
/* ── Mood selector ── */
.mood-grid{
  display:grid;grid-template-columns:repeat(3,1fr);gap:8px;
  margin-bottom:14px;
}
.mood-btn{
  cursor:pointer;border:1px solid var(--border);background:rgba(14,18,24,.6);
  border-radius:12px;padding:10px 6px;text-align:center;
  transition:border-color .15s,background .15s,transform .1s;
  -webkit-appearance:none;
}
.mood-btn:hover{border-color:var(--gold);background:rgba(200,168,75,.08)}
.mood-btn:active{transform:scale(.96)}
.mood-btn.active{border-color:var(--gold);background:rgba(200,168,75,.12)}
.mood-btn .emoji{font-size:22px;display:block;margin-bottom:4px}
.mood-btn .label{font-size:11px;color:var(--muted);font-weight:600}
.mood-btn.active .label{color:var(--gold2)}
 
/* ── Surprise button ── */
.surprise-btn{
  width:100%;cursor:pointer;border:1px solid var(--gold);
  background:linear-gradient(145deg,rgba(200,168,75,.12),rgba(200,168,75,.05));
  color:var(--gold2);font-family:var(--font-sans);font-weight:700;
  font-size:15px;padding:14px;border-radius:14px;
  transition:background .2s,transform .1s;margin-bottom:14px;
  -webkit-appearance:none;letter-spacing:.3px;
}
.surprise-btn:hover{background:linear-gradient(145deg,rgba(200,168,75,.2),rgba(200,168,75,.08))}
.surprise-btn:active{transform:scale(.98)}
 
/* ── Chat ── */
.chat-box{
  background:rgba(8,10,14,.8);border:1px solid var(--border);
  border-radius:14px;padding:12px;height:280px;
  overflow-y:auto;scroll-behavior:smooth;
}
@media(max-width:768px){.chat-box{height:220px}}
.chat-box::-webkit-scrollbar{width:3px}
.chat-box::-webkit-scrollbar-thumb{background:var(--border2);border-radius:2px}
.msg{display:flex;gap:8px;margin:10px 0;animation:fadeUp .2s ease}
@keyframes fadeUp{from{opacity:0;transform:translateY(5px)}to{opacity:1;transform:translateY(0)}}
.avatar{
  width:28px;height:28px;border-radius:8px;border:1px solid var(--border2);
  background:linear-gradient(145deg,#1a2230,#0c1018);
  display:flex;align-items:center;justify-content:center;
  flex-shrink:0;font-size:9px;font-weight:700;color:var(--gold);
}
.msg.me .avatar{color:var(--muted)}
.bubble{
  max-width:82%;padding:9px 12px;border-radius:12px;
  border:1px solid var(--border);background:rgba(14,18,24,.9);
  line-height:1.5;white-space:pre-wrap;font-size:13px;
}
.msg.me .bubble{margin-left:auto;background:rgba(28,36,52,.8);border-color:var(--border2)}
.typing-dots span{
  display:inline-block;width:4px;height:4px;border-radius:50%;
  background:var(--gold);margin:0 2px;animation:blink 1.2s infinite;
}
.typing-dots span:nth-child(2){animation-delay:.2s}
.typing-dots span:nth-child(3){animation-delay:.4s}
@keyframes blink{0%,80%,100%{opacity:.2;transform:scale(.8)}40%{opacity:1;transform:scale(1)}}
 
/* ── Chips ── */
.chips{display:flex;gap:6px;flex-wrap:wrap;margin-top:10px}
.chip{
  cursor:pointer;border:1px solid var(--border);background:rgba(14,18,24,.7);
  color:var(--text);padding:6px 11px;border-radius:999px;
  font-size:12px;font-weight:500;transition:border-color .15s,background .15s;
  -webkit-appearance:none;
}
.chip:hover{border-color:var(--gold);background:rgba(200,168,75,.08);color:var(--gold2)}
 
/* ── Input row ── */
.input-row{display:flex;gap:8px;margin-top:12px}
.chat-input{
  flex:1;min-width:0;padding:10px 13px;border-radius:12px;
  border:1px solid var(--border2);background:rgba(8,10,14,.9);
  color:var(--text);font-family:var(--font-sans);font-size:14px;
  outline:none;transition:border-color .15s;
  -webkit-appearance:none;
}
.chat-input:focus{border-color:var(--gold)}
.chat-input::placeholder{color:var(--faint)}
 
/* ── Status ── */
.status-bar{
  margin-top:10px;display:flex;gap:8px;flex-wrap:wrap;
  justify-content:space-between;align-items:center;
}
.status-text{color:var(--muted);font-size:11px}
.footer-note{
  margin-top:12px;padding:9px 12px;border:1px solid var(--border);
  border-radius:10px;background:rgba(14,18,24,.6);color:var(--muted);font-size:11px;
}
 
/* ── Poster strip ── */
#poster-strip{
  display:grid;grid-template-columns:repeat(auto-fill,minmax(140px,1fr));
  gap:10px;padding:12px;max-height:620px;overflow-y:auto;
}
@media(max-width:768px){
  #poster-strip{grid-template-columns:repeat(auto-fill,minmax(120px,1fr));padding:8px;gap:8px}
}
#poster-strip::-webkit-scrollbar{width:3px}
#poster-strip::-webkit-scrollbar-thumb{background:var(--border2);border-radius:2px}
.poster-card{
  border:1px solid var(--border);border-radius:12px;overflow:hidden;
  background:rgba(10,14,20,.85);
  transition:transform .18s ease,border-color .18s ease,box-shadow .18s ease;
  animation:fadeUp .25s ease;
}
.poster-card:hover{transform:translateY(-3px);border-color:var(--gold);box-shadow:0 6px 24px rgba(200,168,75,.1)}
.poster-img{
  width:100%;aspect-ratio:2/3;
  background:radial-gradient(ellipse 140% 100% at 50% 0%,rgba(200,168,75,.1),transparent 60%),#0a0c10;
  display:flex;align-items:center;justify-content:center;
  overflow:hidden;position:relative;
}
.poster-img img{width:100%;height:100%;object-fit:cover;display:block;transition:transform .3s ease}
.poster-card:hover .poster-img img{transform:scale(1.04)}
.poster-fallback{
  font-family:var(--font-serif);font-size:11px;color:rgba(200,168,75,.7);
  text-align:center;padding:8px;line-height:1.3;
}
.cert-badge{
  position:absolute;top:6px;right:6px;padding:2px 6px;border-radius:5px;
  background:rgba(200,168,75,.15);border:1px solid rgba(200,168,75,.3);
  color:var(--gold);font-size:9px;font-weight:700;
}
.tmdb-badge{
  position:absolute;bottom:5px;left:5px;padding:2px 5px;border-radius:4px;
  background:rgba(1,180,228,.15);border:1px solid rgba(1,180,228,.3);
  color:#01b4e4;font-size:9px;font-weight:600;text-decoration:none;opacity:.8;
}
.watchlist-btn{
  position:absolute;top:6px;left:6px;width:26px;height:26px;
  border-radius:7px;border:1px solid var(--border2);
  background:rgba(8,10,14,.8);color:var(--muted);font-size:13px;
  display:flex;align-items:center;justify-content:center;
  cursor:pointer;transition:border-color .15s,color .15s;-webkit-appearance:none;
}
.watchlist-btn:hover{border-color:var(--gold);color:var(--gold)}
.watchlist-btn.saved{border-color:var(--gold);color:var(--gold);background:rgba(200,168,75,.15)}
.poster-info{padding:8px 8px 10px}
.poster-title{font-weight:700;font-size:12px;line-height:1.25;color:var(--text);margin-bottom:3px}
.poster-meta{color:var(--muted);font-size:10px;margin-bottom:7px}
.poster-btns{display:flex;gap:5px}
.poster-btns a,.poster-btns button{
  flex:1;text-align:center;padding:6px 4px;border-radius:8px;
  border:1px solid var(--border);background:rgba(14,18,24,.6);
  color:var(--text);text-decoration:none;font-size:10px;font-weight:600;
  transition:border-color .15s,background .15s;cursor:pointer;
  font-family:var(--font-sans);-webkit-appearance:none;
}
.poster-btns a:hover,.poster-btns button:hover{border-color:var(--gold);background:rgba(200,168,75,.08)}
.poster-btns .why-btn{color:var(--gold)}
.empty-state{grid-column:1/-1;padding:28px;text-align:center;color:var(--muted);font-size:13px}
 
/* ── Watchlist panel ── */
.watchlist-panel{
  display:none;position:fixed;inset:0;background:rgba(4,6,10,.9);
  backdrop-filter:blur(8px);z-index:300;align-items:flex-end;
  justify-content:center;padding:0;
}
.watchlist-panel.open{display:flex}
.watchlist-sheet{
  background:var(--surface2);border:1px solid var(--border2);
  border-radius:20px 20px 0 0;width:100%;max-width:600px;
  max-height:80vh;overflow-y:auto;padding:20px;
}
.watchlist-sheet h3{font-family:var(--font-serif);color:var(--gold);font-size:18px;margin-bottom:16px}
.watchlist-item{
  display:flex;align-items:center;gap:12px;padding:10px 0;
  border-bottom:1px solid var(--border);
}
.watchlist-item:last-child{border-bottom:none}
.wl-title{flex:1;font-size:13px;font-weight:600}
.wl-meta{color:var(--muted);font-size:11px}
.wl-remove{
  cursor:pointer;width:26px;height:26px;border-radius:7px;
  border:1px solid var(--border);background:transparent;
  color:var(--muted);font-size:14px;display:flex;align-items:center;
  justify-content:center;transition:border-color .15s,color .15s;
}
.wl-remove:hover{border-color:var(--red);color:var(--red)}
.wl-empty{color:var(--muted);font-size:13px;text-align:center;padding:20px 0}
 
/* ── Modal ── */
.modal-backdrop{
  display:none;position:fixed;inset:0;background:rgba(4,6,10,.85);
  backdrop-filter:blur(6px);z-index:200;align-items:center;
  justify-content:center;padding:16px;
}
.modal-backdrop.open{display:flex}
.modal{
  background:var(--surface2);border:1px solid var(--border2);
  border-radius:18px;max-width:460px;width:100%;
  box-shadow:0 32px 80px rgba(0,0,0,.7);overflow:hidden;
}
.modal-head{
  padding:16px 18px 12px;border-bottom:1px solid var(--border);
  display:flex;align-items:center;justify-content:space-between;gap:12px;
}
.modal-title{font-family:var(--font-serif);font-size:17px;color:var(--gold)}
.modal-close{
  cursor:pointer;width:28px;height:28px;border-radius:8px;
  border:1px solid var(--border);background:transparent;color:var(--muted);
  font-size:16px;display:flex;align-items:center;justify-content:center;
  transition:border-color .15s,color .15s;
}
.modal-close:hover{border-color:var(--red);color:var(--red)}
.modal-body{padding:18px;color:var(--text);font-size:13.5px;line-height:1.7}
 
/* ── TMDB footer ── */
.tmdb-footer{
  margin-top:24px;padding:14px 16px;border-top:1px solid var(--border);
  display:flex;align-items:center;gap:12px;flex-wrap:wrap;
}
.tmdb-logo-wrap{display:flex;align-items:center;gap:8px}
.tmdb-logo{height:18px;opacity:.7}
.tmdb-footer-text{color:var(--muted);font-size:11px;line-height:1.4}
.tmdb-footer-text a{color:#01b4e4;text-decoration:none}
</style>
</head>
<body>
<div class="wrap">
 
<!-- Header -->
<div class="header">
  <div class="brand">
    <div class="logo"><span>SG</span></div>
    <div class="brand-text">
      <h1>SG <em>Film Ajánló</em></h1>
      <p>AI chat + intelligens ajánlások</p>
    </div>
  </div>
  <div class="badges" style="align-items:center;gap:10px">
    <div id="pill-loaded" style="font-size:11px;color:var(--gold);font-weight:600"></div>
    <div style="font-size:11px;color:var(--muted)">🚫 Nem stream oldal</div>
    <button class="btn" id="btn-watchlist" style="padding:7px 12px;font-size:12px">🔖 Lista</button>
  </div>
</div>
 
<!-- Main grid -->
<div class="grid">
 
  <!-- Chat card -->
  <div class="card">
    <div class="card-head">
      <div class="card-title">🤖 AI Asszisztens</div>
      <button class="btn" id="btn-reset">↺ Reset</button>
    </div>
    <div class="card-body">
 
      <!-- Chat -->
      <div class="chat-box" id="chat-box"></div>
      <div class="chips" id="chips"></div>
      <div class="input-row">
        <input class="chat-input" id="inp" placeholder="pl. „feszült thriller 2 óra"" autocomplete="off"/>
        <button class="btn" id="btn-send">➤</button>
      </div>
 
      <!-- Mood selector -->
      <div class="mood-grid" id="mood-grid" style="margin-top:12px">
        <button class="mood-btn" data-mood="porgos"><span class="emoji">⚡</span><span class="label">Pörgős</span></button>
        <button class="mood-btn" data-mood="nyugis"><span class="emoji">😌</span><span class="label">Nyugis</span></button>
        <button class="mood-btn" data-mood="sotet"><span class="emoji">🌑</span><span class="label">Sötét</span></button>
        <button class="mood-btn" data-mood="felemelo"><span class="emoji">🚀</span><span class="label">Felemelő</span></button>
        <button class="mood-btn" data-mood="vicces"><span class="emoji">😂</span><span class="label">Vicces</span></button>
        <button class="mood-btn" data-mood="romantic"><span class="emoji">💕</span><span class="label">Romantikus</span></button>
      </div>
 
      <!-- Surprise me -->
      <button class="surprise-btn" id="btn-surprise" style="margin-top:10px;margin-bottom:0">🎲 Lepj meg! — random film</button>
      <div class="status-bar">
        <div class="status-text" id="status-line">Válassz hangulatot vagy írj!</div>
      </div>
      <div class="footer-note">
        Nem vagyunk stream oldal. A trailer YouTube-ra visz.
      </div>
    </div>
  </div>
 
  <!-- Poster card -->
  <div class="card">
    <div class="card-head">
      <div class="card-title">🎥 Ajánlott filmek</div>
      <button class="btn" id="btn-more">+ Tölts még</button>
    </div>
    <div class="card-body" style="padding:0">
      <div id="poster-strip"></div>
      <div style="padding:10px 14px 14px;border-top:1px solid var(--border);font-size:10px;color:var(--faint);line-height:1.6">
        © A filmek poszterein és adatain fennálló szerzői jogok a jogtulajdonosokat illetik.
        A trailer linkek YouTube-ra mutatnak. Ez az oldal kizárólag ajánló célokat szolgál,
        tartalmat nem tárol és nem közvetít.
      </div>
    </div>
  </div>
 
</div>
 
<!-- TMDB Footer -->
<div class="tmdb-footer">
  <div class="tmdb-logo-wrap">
    <img class="tmdb-logo"
      src="https://www.themoviedb.org/assets/2/v4/logos/v2/blue_short-8e7b30f73a4020692ccca9c88bafe5dcb6f8a62a4c6bc55cd9ba82bb2cd95f6c.svg"
      alt="TMDB logo"/>
  </div>
  <div class="tmdb-footer-text">
    This product uses the TMDB API but is not endorsed or certified by TMDB.<br>
    Poszter képek forrása: <a href="https://www.themoviedb.org/" target="_blank" rel="noopener">The Movie Database (TMDB)</a>
  </div>
</div>
 
</div><!-- /wrap -->
 
<!-- Watchlist panel -->
<div class="watchlist-panel" id="watchlist-panel">
  <div class="watchlist-sheet">
    <div style="display:flex;align-items:center;justify-content:space-between;margin-bottom:16px">
      <h3>🔖 Mentett filmek</h3>
      <button class="modal-close" id="watchlist-close">✕</button>
    </div>
    <div id="watchlist-list"></div>
  </div>
</div>
 
<!-- Why modal -->
<div class="modal-backdrop" id="modal">
  <div class="modal">
    <div class="modal-head">
      <div class="modal-title" id="modal-title">Miért ajánljuk?</div>
      <button class="modal-close" id="modal-close">✕</button>
    </div>
    <div class="modal-body" id="modal-body"></div>
  </div>
</div>
 
<script>
(function(){
'use strict';
 
/* ── State ── */
let state = {mood:'porgos',brain:'konnyu',time:120,q:'',offset:0,take:12,ready:false};
let watchlist = JSON.parse(localStorage.getItem('sg_watchlist') || '[]');
 
/* ── Refs ── */
const chatBox     = document.getElementById('chat-box');
const chips       = document.getElementById('chips');
const inp         = document.getElementById('inp');
const statusLine  = document.getElementById('status-line');
const posterStrip = document.getElementById('poster-strip');
const pillLoaded  = document.getElementById('pill-loaded');
const modal       = document.getElementById('modal');
const modalTitle  = document.getElementById('modal-title');
const modalBody   = document.getElementById('modal-body');
 
/* ── Helpers ── */
function esc(s){
  return String(s||'')
    .replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;')
    .replace(/"/g,'&quot;').replace(/'/g,'&#039;');
}
 
/* ── Modal ── */
function openModal(title,text){
  modalTitle.textContent = title;
  modalBody.textContent  = text||'—';
  modal.classList.add('open');
}
function closeModal(){modal.classList.remove('open')}
document.getElementById('modal-close').addEventListener('click',closeModal);
modal.addEventListener('click',e=>{if(e.target===modal)closeModal()});
document.addEventListener('keydown',e=>{if(e.key==='Escape')closeModal()});
 
/* ── Watchlist ── */
function saveWatchlist(){localStorage.setItem('sg_watchlist',JSON.stringify(watchlist))}
function isInWatchlist(title,year){return watchlist.some(m=>m.title===title&&m.year===year)}
function toggleWatchlist(movie){
  const idx = watchlist.findIndex(m=>m.title===movie.title&&m.year===movie.year);
  if(idx>=0) watchlist.splice(idx,1);
  else watchlist.push(movie);
  saveWatchlist();
  renderWatchlist();
  // Update all buttons for this movie
  document.querySelectorAll(`.watchlist-btn[data-title="${esc(movie.title)}"]`).forEach(btn=>{
    btn.classList.toggle('saved', isInWatchlist(movie.title, movie.year));
    btn.textContent = isInWatchlist(movie.title, movie.year) ? '★' : '☆';
  });
}
function renderWatchlist(){
  const list = document.getElementById('watchlist-list');
  if(!watchlist.length){
    list.innerHTML='<div class="wl-empty">Még nincs mentett film.</div>';
    return;
  }
  list.innerHTML = watchlist.map(m=>`
    <div class="watchlist-item">
      <div>
        <div class="wl-title">${esc(m.title)}</div>
        <div class="wl-meta">${esc(String(m.year))} · ${esc(m.genres?.join(', ')||'')}</div>
      </div>
      <button class="wl-remove" data-title="${esc(m.title)}" data-year="${esc(String(m.year))}">✕</button>
    </div>`).join('');
  list.querySelectorAll('.wl-remove').forEach(btn=>{
    btn.addEventListener('click',()=>{
      watchlist = watchlist.filter(m=>!(m.title===btn.dataset.title&&String(m.year)===btn.dataset.year));
      saveWatchlist();
      renderWatchlist();
      document.querySelectorAll(`.watchlist-btn[data-title="${btn.dataset.title}"]`).forEach(b=>{
        b.classList.remove('saved'); b.textContent='☆';
      });
    });
  });
}
document.getElementById('btn-watchlist').addEventListener('click',()=>{
  renderWatchlist();
  document.getElementById('watchlist-panel').classList.add('open');
});
document.getElementById('watchlist-close').addEventListener('click',()=>{
  document.getElementById('watchlist-panel').classList.remove('open');
});
document.getElementById('watchlist-panel').addEventListener('click',e=>{
  if(e.target===document.getElementById('watchlist-panel'))
    document.getElementById('watchlist-panel').classList.remove('open');
});
 
/* ── Mood buttons ── */
const moodMessages = {
  'porgos':   '⚡ Pörgős filmeket keresek neked!',
  'nyugis':   '😌 Nyugis, chill filmeket hozok!',
  'sotet':    '🌑 Sötét, feszült filmek jönnek!',
  'felemelo': '🚀 Felemelő, motiváló filmeket keresek!',
  'vicces':   '😂 Vicces, könnyed filmeket hozok!',
  'romantic': '💕 Romantikus filmeket keresek neked!',
};
document.querySelectorAll('.mood-btn').forEach(btn=>{
  btn.addEventListener('click', async ()=>{
    document.querySelectorAll('.mood-btn').forEach(b=>b.classList.remove('active'));
    btn.classList.add('active');
    const mood = btn.dataset.mood;
    // Közvetlenül beállítjuk a state-et
    state.mood   = mood;
    state.ready  = true;
    state.offset = 0;
    // Chat üzenet
    addMsg('SG', moodMessages[mood] || '🎬 Keresem a legjobb filmeket!');
    // Azonnal töltjük a filmeket — NEM a chaten keresztül
    posterStrip.innerHTML = '';
    await loadMore();
    statusLine.textContent = 'mood='+state.mood+' · time≈'+state.time+'min · mód='+state.brain;
  });
});
 
/* ── Surprise me ── */
document.getElementById('btn-surprise').addEventListener('click', async()=>{
  const btn = document.getElementById('btn-surprise');
  btn.textContent = '⏳ Keresés...';
  btn.disabled = true;
  try {
    const moods = ['porgos','nyugis','sotet','felemelo','vicces'];
    const randomMood = moods[Math.floor(Math.random()*moods.length)];
    const randomTime = [90,120,150,180][Math.floor(Math.random()*4)];
    const brains = ['konnyu','kozepes','elgondolkodtato'];
    const randomBrain = brains[Math.floor(Math.random()*brains.length)];
 
    const url = '/api/recs?mood='+randomMood+'&brain='+randomBrain
      +'&time='+randomTime+'&offset='+Math.floor(Math.random()*50)+'&take=1';
    const res  = await fetch(url);
    const data = await res.json();
    const film = data.items?.[0];
    if(film){
      posterStrip.innerHTML = '';
      state.offset = 0;
      state.mood   = randomMood;
      state.brain  = randomBrain;
      state.time   = randomTime;
      state.ready  = true;
      await loadMore();
      addMsg('SG', '🎲 Random ajánlat: ' + film.title + ' — próbáld ki!');
    }
  } catch(e){ console.error(e); }
  btn.textContent = '🎲 Lepj meg! — random film';
  btn.disabled = false;
});
 
/* ── Chat ── */
function addMsg(who,text){
  const row = document.createElement('div');
  row.className = 'msg '+(who==='Te'?'me':'ai');
  const av = document.createElement('div');
  av.className = 'avatar';
  av.textContent = who==='Te'?'Te':'SG';
  const bub = document.createElement('div');
  bub.className = 'bubble';
  bub.textContent = text||'';
  row.append(av,bub);
  chatBox.appendChild(row);
  chatBox.scrollTop = chatBox.scrollHeight;
  return row;
}
function addTyping(){
  const row = document.createElement('div');
  row.className = 'msg ai';
  row.innerHTML = '<div class="avatar">SG</div><div class="bubble"><div class="typing-dots"><span></span><span></span><span></span></div></div>';
  chatBox.appendChild(row);
  chatBox.scrollTop = chatBox.scrollHeight;
  return row;
}
 
/* ── Chips ── */
function setChips(arr){
  chips.innerHTML='';
  (arr||[]).forEach(t=>{
    const c=document.createElement('div');
    c.className='chip';c.textContent=t;
    c.addEventListener('click',async()=>{
      const low=t.toLowerCase();
      const timeMap={'90 perc':90,'120 perc':120,'150 perc':150,'180 perc':180};
      if(timeMap[low]!==undefined){state.time=timeMap[low];inp.value=t;await send();return;}
      const moodMap={'pörgős':'porgos','nyugis':'nyugis','sötét':'sotet','felemelő':'felemelo','vicces':'vicces','romantikus':'romantic'};
      if(moodMap[low]){document.querySelector('.mood-btn[data-mood="'+moodMap[low]+'"]')?.click();return;}
      if(low==='meglepj'||low==='lepj meg'){document.getElementById('btn-surprise').click();return;}
      inp.value=t;await send();
    });
    chips.appendChild(c);
  });
}
 
/* ── Posters ── */
function posterHTML(m){
  const poster  = (m.poster||'').trim();
  const trailer = (m.trailer||'').trim();
  const cert    = (m.certification||'').trim();
  const saved   = isInWatchlist(m.title,m.year);
 
  const fallbackStyle = poster?'style="display:none"':'';
  const fallback = '<div class="poster-fallback" '+fallbackStyle+'>'+esc(m.title)+'</div>';
 
  let imgBlock='';
  if(poster){
    const src = poster.startsWith('http')
      ? '/api/poster?url='+encodeURIComponent(poster)
      : esc(poster);
    imgBlock = '<img src="'+src+'" alt="'+esc(m.title)+'" loading="eager" style="width:100%;height:100%;object-fit:cover;display:block" onerror="this.style.display=\'none\';this.nextElementSibling.style.display=\'flex\'">';
  }
 
  const certBadge = cert?'<div class="cert-badge">'+esc(cert)+'+</div>':'';
  const tmdbBadge = poster?'<a class="tmdb-badge" href="https://www.themoviedb.org/" target="_blank" rel="noopener">TMDB</a>':'';
  const wlBtn = '<button class="watchlist-btn'+(saved?' saved':'')+'" data-title="'+esc(m.title)+'" data-year="'+esc(String(m.year))+'">'+(saved?'★':'☆')+'</button>';
 
  // Trailer URL — közvetlen lejátszás
  let ytUrl;
  let ytEmbed = '';
  if(trailer && trailer.includes('watch?v=')){
    const vid = trailer.split('watch?v=')[1].split('&')[0];
    ytUrl = 'https://www.youtube.com/watch?v=' + vid;
    ytEmbed = vid;
  } else if(trailer && trailer.includes('youtu.be/')){
    const vid = trailer.split('youtu.be/')[1].split('?')[0];
    ytUrl = 'https://www.youtube.com/watch?v=' + vid;
    ytEmbed = vid;
  } else {
    // YouTube keresés helyett az autoplay trükkel
    ytUrl = 'https://www.youtube.com/results?search_query='+encodeURIComponent(m.title+' '+m.year+' official trailer');
    ytEmbed = '';
  }
 
  const rating = m.avg_rating>0
    ? '<span style="color:var(--gold);margin-left:3px">★'+m.avg_rating.toFixed(1)+'</span>'
    : '';
 
  return '<div class="poster-card">'
    +'<div class="poster-img">'+imgBlock+fallback+certBadge+tmdbBadge+wlBtn+'</div>'
    +'<div class="poster-info">'
    +'<div class="poster-title">'+esc(m.title)+'</div>'
    +'<div class="poster-meta">'+esc(String(m.year))+' · '+esc(String(m.minutes))+' perc'+rating+'</div>'
    +'<div class="poster-btns">'
    +'<button class="trailer-play-btn"'+'  data-url="'+esc(ytUrl)+'"'+'  data-embed="'+esc(ytEmbed)+'"'+'  data-title="'+esc(m.title)+'"'+'  data-year="'+esc(String(m.year||''))+'"'+'  data-tmdb="'+esc(String(m.tmdb_id||''))+'"'+'>▶ Trailer</button>'
    +'<button class="why-btn" data-title="'+esc(m.title)+'" data-why="'+esc(m.why||'')+'">? Miért</button>'
    +'</div></div></div>';
}
 
function bindButtons(scope){
  scope.querySelectorAll('.why-btn').forEach(btn=>{
    btn.addEventListener('click',e=>{
      e.preventDefault();
      openModal(btn.dataset.title, btn.dataset.why||'—');
    });
  });
  scope.querySelectorAll('.trailer-play-btn').forEach(btn=>{
    btn.addEventListener('click', async ()=>{
      const title = btn.dataset.title;
      const year  = btn.dataset.year  || '';
      const tmdb  = btn.dataset.tmdb  || '';
      const orig  = btn.textContent;
 
      btn.textContent = '⏳';
      btn.disabled = true;
 
      try {
        const res = await fetch(
          '/api/trailer?title='+encodeURIComponent(title)
          +'&year='+encodeURIComponent(year)
          +'&tmdb_id='+encodeURIComponent(tmdb)
        );
        const data = await res.json();
        const url = data.url || '';
 
        if(url){
          window.location.href = url;
        } else {
          window.location.href =
            'https://www.youtube.com/results?search_query=' +
            encodeURIComponent(title+' '+year+' official trailer');
        }
      } catch(e){
        window.location.href =
          'https://www.youtube.com/results?search_query=' +
          encodeURIComponent(title+' trailer');
      } finally {
        btn.textContent = orig;
        btn.disabled = false;
      }
    });
  });
  scope.querySelectorAll('.watchlist-btn').forEach(btn=>{
    btn.addEventListener('click',()=>{
      const movie = {
        title: btn.dataset.title,
        year:  parseInt(btn.dataset.year)||0,
        genres: [],
      };
      toggleWatchlist(movie);
    });
  });
}
 
async function loadMore(){
  const url = '/api/recs?mood='+encodeURIComponent(state.mood)
    +'&brain='+encodeURIComponent(state.brain)
    +'&time='+encodeURIComponent(state.time)
    +'&q='+encodeURIComponent(state.q||'')
    +'&offset='+state.offset+'&take='+state.take;
  try{
    const res  = await fetch(url);
    const data = await res.json();
    const pl=document.getElementById('pill-loaded');if(pl)pl.textContent=(data.total||0)+' film';
    const items = data.items||[];
    if(!items.length){
      const el=document.createElement('div');
      el.className='empty-state';el.textContent='Nincs több találat.';
      posterStrip.appendChild(el);return;
    }
    const chunk=document.createElement('div');
    chunk.style.display='contents';
    chunk.innerHTML=items.map(posterHTML).join('');
    posterStrip.appendChild(chunk);
    bindButtons(chunk);
    state.offset+=items.length;
  }catch(e){console.error('loadMore:',e)}
}
 
/* ── Send ── */
async function send(){
  const msg=(inp.value||'').trim();
  if(!msg)return;
  addMsg('Te',msg);
  inp.value='';
  const typing=addTyping();
  try{
    const res=await fetch('/api/chat',{
      method:'POST',headers:{'Content-Type':'application/json'},
      body:JSON.stringify({message:msg}),
    });
    const data=await res.json();
    typing.remove();
    addMsg('SG',data.assistant||'…');
    setChips(data.quick_replies||[]);
    const prof=data.profile||{};
    if(prof.mood) state.mood=prof.mood;
    if(prof.brain)state.brain=prof.brain;
    if(prof.time) state.time=Number(prof.time)||state.time;
    state.q=prof.extra||'';
    const wasReady=state.ready;
    state.ready=!!prof.ready;
    statusLine.textContent='mood='+state.mood+' · time≈'+state.time+'min · mód='+state.brain
      +(state.q?' · '+state.q.slice(0,25):'');
    if(state.ready){
      state.offset=0;posterStrip.innerHTML='';
      await loadMore();
    }
  }catch(e){
    typing.remove();
    addMsg('SG','⚠ Hálózati hiba. Próbáld újra.');
    console.error(e);
  }
}
 
/* ── Events ── */
document.getElementById('btn-send').addEventListener('click',send);
document.getElementById('btn-reset').addEventListener('click',()=>{inp.value='Reset';send()});
document.getElementById('btn-more').addEventListener('click',loadMore);
inp.addEventListener('keydown',e=>{if(e.key==='Enter')send()});
 
/* ── Init ── */
addMsg('SG','Szia! 👋 Válassz hangulatot a gombokkal, vagy írj mit szeretnél nézni.');
setChips(['90 perc','120 perc','180 perc','meglepj','Reset']);
fetch('/api/debug').then(r=>r.json()).then(d=>{const p=document.getElementById('pill-loaded');if(p)p.textContent=(d.movies_loaded||0)+' film';}).catch(()=>{});
})();
</script>
</body>
</html>"""
 
@app.get("/")
def home():
    return _HTML
 
# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    log.info("Starting SG Film Ajánló on port %d (debug=%s)", Config.FLASK_PORT, Config.FLASK_DEBUG)
    app.run(
        host="0.0.0.0",
        port=int(os.environ.get("port",10000)),
        debug=False
    )
