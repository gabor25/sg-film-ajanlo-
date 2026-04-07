Nincs kijelölt elem 

Ugrás a tartalomra
A Gmail használata képernyőolvasóval
A(z) Gmail asztali értesítéseinek bekapcsolása.
   OK  Köszönöm, nem.
309/1.
Demo.html
Beérkező levelek

Behbud Malikov <behbud.malikov@gmail.com>
Mellékletek
14:44 (49 perccel ezelőtt)
címzett: én

 Egy melléklet
  •  A Gmail által ellenőrizve

Gábor Szecsődi
15:14 (20 perccel ezelőtt)
címzett: Behbud

import os
import csv
import json
import random
import re
from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Tuple

from flask import Flask, request, jsonify, session

# -----------------------------
# .env (minimal loader, no extra package)
# -----------------------------
def load_dotenv_local(path: str = ".env") -> None:
    if not os.path.exists(path):
        return
    try:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                k, v = line.split("=", 1)
                k = k.strip()
                v = v.strip().strip('"').strip("'")
                os.environ.setdefault(k, v)
    except Exception:
        pass

load_dotenv_local()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "").strip()
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-5.2").strip()

# -----------------------------
# Data model
# -----------------------------
@dataclass
class Movie:
    title: str
    year: int
    minutes: int
    genres: List[str]
    tags: List[str]
    why: str
    poster: str = ""          # URL
    trailer: str = ""         # YouTube URL
    certification: str = ""   # e.g. "12", "16", "18", "PG-13"


BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Prefer clean csv if exists, otherwise fallback to movies.csv
CSV_CLEAN = os.path.join(BASE_DIR, "movies_clean.csv")
CSV_RAW = os.path.join(BASE_DIR, "movies.csv")
CSV_PATH = CSV_CLEAN if os.path.exists(CSV_CLEAN) else CSV_RAW


# -----------------------------
# CSV loader
# -----------------------------
def detect_delimiter(first_line: str) -> str:
    # a leggyakoribbak: ; vagy , vagy tab
    candidates = [";", ",", "\t"]
    counts = {c: first_line.count(c) for c in candidates}
    return max(counts, key=counts.get) if first_line else ","


def load_movies_csv(path: str) -> List[Movie]:
    movies: List[Movie] = []
    if not os.path.exists(path):
        print("CSV nem található:", path)
        return movies

    bad = 0

    def to_int(x: Any, default: int = 0) -> int:
        try:
            s = str(x).strip()
            if s == "":
                return default
            return int(float(s))
        except Exception:
            return default

    def split_pipe(x: Any) -> List[str]:
        return [p.strip().lower() for p in str(x or "").split("|") if p.strip()]

    with open(path, "r", encoding="utf-8-sig", errors="replace", newline="") as f:
        first = f.readline()
        f.seek(0)

        delim = detect_delimiter(first)
        reader = csv.DictReader(f, delimiter=delim)

        for i, row in enumerate(reader, start=2):
            try:
                title = (row.get("title") or row.get("Title") or "").strip()
                if not title:
                    bad += 1
                    continue

                movies.append(
                    Movie(
                        title=title,
                        year=to_int(row.get("year") or row.get("Year") or 0),
                        minutes=to_int(row.get("minutes") or row.get("Minutes") or 0),
                        genres=split_pipe(row.get("genres") or row.get("Genres") or ""),
                        tags=split_pipe(row.get("tags") or row.get("Tags") or ""),
                        why=(row.get("why") or row.get("Why") or "").strip(),
                        poster=(row.get("poster") or "").strip().strip('"').strip("'"),
                        trailer=(row.get("trailer") or "").strip().strip('"').strip("'"),
                        certification=(row.get("certification") or row.get("Certification") or "").strip(),
                    )
                )
            except Exception as e:
                bad += 1
                if bad <= 6:
                    print(f"ROSSZ SOR {i}: {e} | keys={list(row.keys())}")
                continue

    print("Betöltve:", len(movies), "Hibás/skip:", bad, "Fájl:", path)
    return movies


MOVIES: List[Movie] = load_movies_csv(CSV_PATH)

# -----------------------------
# Mini NLU (offline)
# -----------------------------
MOOD_SYNONYMS = {
    "porgos": ["pörg", "akció", "gyors", "üldöz", "harc", "bosszú", "adrenalin", "darál"],
    "nyugis": ["nyugis", "chill", "feel", "laza", "szívmelenget", "romi", "romant"],
    "sotet":  ["söt", "sot", "thriller", "krimi", "parás", "nyomaszt", "gyilk", "pszich"],
    "felemelo": ["felem", "motiv", "inspir", "pozit", "kitart", "remény", "remeny"],
    "vicces": ["vicc", "kom", "nevet", "őrült", "orult", "paród", "parod", "humor"],
}

BRAIN_SYNONYMS = {
    "konnyu": ["könny", "konny", "laza", "agykikapcs", "nem akarok gondolkodni", "egyszerű", "egyszeru"],
    "kozepes": ["közep", "kozep", "normál", "normal"],
    "elgondolkodtato": ["elgondolk", "agyas", "agyal", "csavaros", "pszich", "bonyolult", "twist"],
}

def extract_time_smart(text: str) -> Optional[int]:
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
    if "két óra" in t or "2 óra" in t or "2 ora" in t or "120" in t:
        return 120
    if "három óra" in t or "3 óra" in t or "3 ora" in t or "180" in t:
        return 180
    return None

def normalize_mood_smart(text: str) -> Optional[str]:
    t = (text or "").lower()
    best = None
    best_score = 0
    for mood, keys in MOOD_SYNONYMS.items():
        score = sum(1 for k in keys if k in t)
        if score > best_score:
            best_score = score
            best = mood
    return best if best_score > 0 else None

def normalize_brain_smart(text: str) -> Optional[str]:
    t = (text or "").lower()
    best = None
    best_score = 0
    for brain, keys in BRAIN_SYNONYMS.items():
        score = sum(1 for k in keys if k in t)
        if score > best_score:
            best_score = score
            best = brain
    return best if best_score > 0 else None

def extract_keywords_smart(text: str) -> str:
    t = (text or "").lower().strip()
    stop = {"legyen", "valami", "film", "néznék", "neznek", "néznek", "ma", "most", "kell", "akarok", "szeretnék", "szeretnek"}
    tokens = [x for x in re.split(r"[^a-záéíóöőúüű0-9]+", t) if x]
    tokens = [x for x in tokens if x not in stop and len(x) >= 3]
    return " ".join(tokens[:10])

# -----------------------------
# OpenAI NLU profile (optional)
# -----------------------------
def openai_nlu_profile(user_msg: str) -> Optional[Dict[str, Any]]:
    if not OPENAI_API_KEY:
        return None
    try:
        from openai import OpenAI
    except Exception:
        return None

    client = OpenAI(api_key=OPENAI_API_KEY)
    system = (
        "Te egy magyar NLU (szövegértelmező) vagy filmes ajánlóhoz. "
        "Feladat: a felhasználó üzenetéből kinyerni a profilt. "
        "CSAK érvényes JSON-t adj vissza, semmi mást.\n"
        "Kimenet séma:\n"
        "{"
        "\"mood\": \"porgos|nyugis|sotet|felemelo|vicces|null\", "
        "\"time\": 90|120|180|egyéb perc (60-240)|null, "
        "\"brain\": \"konnyu|kozepes|elgondolkodtato|null\", "
        "\"q\": \"kulcsszavak röviden\""
        "}\n"
        "Ha nincs információ, null. A q legyen rövid (max ~8 szó)."
    )

    try:
        resp = client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user_msg}
            ],
            temperature=0.2,
            max_tokens=120
        )
        text = (resp.choices[0].message.content or "").strip()

        try:
            obj = json.loads(text)
            return obj if isinstance(obj, dict) else None
        except Exception:
            start = text.find("{")
            end = text.rfind("}")
            if start != -1 and end != -1 and end > start:
                try:
                    obj = json.loads(text[start:end+1])
                    return obj if isinstance(obj, dict) else None
                except Exception:
                    return None
            return None
    except Exception:
        return None

# -----------------------------
# Simple recommender scoring
# -----------------------------
MOOD_KEYWORDS = {
    "porgos": ["akció", "gyors", "pörgős", "harc", "üldözés", "bosszú"],
    "nyugis": ["feel-good", "nyugis", "laza", "szívmelengető", "élet", "család"],
    "sotet":  ["thriller", "sötét", "csavaros", "pszichológiai", "krimi", "bűn"],
    "felemelo": ["motiváló", "felemelő", "inspiráló", "kitartás", "sport", "cél"],
    "vicces": ["komédia", "vicces", "nevetős", "baráti", "őrült", "laza"],
}

BRAIN_KEYWORDS = {
    "konnyu": ["könnyű", "laza", "nem akarok gondolkodni"],
    "kozepes": ["közepes"],
    "elgondolkodtato": ["elgondolkodtató", "agyas", "agyalós", "csavaros", "pszichológiai"],
}

def score_movie(m: Movie, mood: str, time_limit: int, brain: str, extra: str) -> int:
    s = 0
    blob = f"{m.title} {' '.join(m.tags)} {' '.join(m.genres)} {m.why}".lower()

    # time closeness
    if m.minutes and time_limit:
        diff = abs(m.minutes - time_limit)
        if diff <= 10:
            s += 10
        elif diff <= 25:
            s += 7
        elif diff <= 45:
            s += 4
        elif diff <= 80:
            s += 1
        else:
            s -= 2

    # mood match
    for w in MOOD_KEYWORDS.get(mood, []):
        if w in blob:
            s += 3

    # brain match
    if brain == "konnyu":
        if any(w in blob for w in BRAIN_KEYWORDS["elgondolkodtato"]):
            s -= 2
        if "komédia" in blob or "vicces" in blob:
            s += 1
    elif brain == "elgondolkodtato":
        if any(w in blob for w in BRAIN_KEYWORDS["elgondolkodtato"]):
            s += 4

    # extra tokens
    extra = (extra or "").strip().lower()
    if extra:
        for token in [t for t in extra.replace(",", " ").split() if len(t) >= 3]:
            if token in blob:
                s += 2

    s += random.randint(0, 2)
    return s

def rank_movies(mood: str, time_limit: int, brain: str, extra: str, offset: int, take: int) -> List[Movie]:
    ranked = sorted(MOVIES, key=lambda mm: score_movie(mm, mood, time_limit, brain, extra), reverse=True)
    return ranked[offset:offset + take]


# -----------------------------
# “AI chat” profile in session
# -----------------------------
def default_profile() -> Dict[str, Any]:
    return {
        "time": None,
        "mood": None,
        "brain": None,
        "extra": "",
        "ready": False,
        "history": []
    }

def get_profile() -> Dict[str, Any]:
    p = session.get("profile")
    if not p:
        p = default_profile()
        session["profile"] = p
    return p

def needs_questions(p: Dict[str, Any]) -> List[str]:
    missing = []
    if not p.get("mood"):
        missing.append("mood")
    if not p.get("time"):
        missing.append("time")
    if not p.get("brain"):
        missing.append("brain")
    return missing

def next_question_text(p: Dict[str, Any]) -> Tuple[str, List[str]]:
    missing = needs_questions(p)
    if not missing:
        return (
            "Oké, megvan minden. Finomítsunk: írj 1-2 kulcsszót (pl. maffia / űr / csavaros / bosszú) vagy nyomj „Ajánlj”-t.",
            ["Ajánlj", "Újra dobás", "Sötétebb", "Viccesebb", "Rövidebb", "Reset"]
        )
    if missing[0] == "mood":
        return ("Milyen hangulatot szeretnél? (pörgős / nyugis / sötét / felemelő / vicces)", ["pörgős", "nyugis", "sötét", "felemelő", "vicces"])
    if missing[0] == "time":
        return ("Mennyi időd van ma filmre? (90 / 120 / 180 perc)", ["90 perc", "120 perc", "180 perc"])
    if missing[0] == "brain":
        return ("Mennyire legyen elgondolkodtató? (könnyű / közepes / elgondolkodtató)", ["könnyű", "közepes", "elgondolkodtató"])
    return ("Oké.", [])

# -----------------------------
# OPTIONAL: Real AI chat reply (kept)
# -----------------------------
def openai_chat_reply(p: Dict[str, Any], user_msg: str) -> Optional[str]:
    if not OPENAI_API_KEY:
        return None
    try:
        from openai import OpenAI
    except Exception:
        return None

    client = OpenAI(api_key=OPENAI_API_KEY)

    system = (
        "Te egy magyar nyelvű filmes ajánló asszisztens vagy (SG Film Ajánló). "
        "Cél: barátságos, rövid, mégis 'AI-s' beszélgetés. "
        "NE ajánlj rögtön filmeket 'szia' üzenetre. Először hiányzó adatokat kérdezz ki: idő (90/120/180), hangulat, elgondolkodtatóság. "
        "Ha már mind megvan, kérj extra kulcsszót, vagy ajánlj. "
        "Stílus: sötét elegancia, laza, motiváló, 1-2 emoji max. "
        "Soha ne állítsd, hogy stream oldalak vagytok."
    )

    history = p.get("history") or []
    msgs = [{"role": "system", "content": system}]
    for h in history[-12:]:
        if h.get("role") in ("user", "assistant") and h.get("content"):
            msgs.append({"role": h["role"], "content": h["content"]})

    profile_hint = (
        f"Jelenlegi profil: time={p.get('time')}, mood={p.get('mood')}, brain={p.get('brain')}, extra='{p.get('extra','')}'."
    )
    msgs.append({"role": "system", "content": profile_hint})
    msgs.append({"role": "user", "content": user_msg})

    try:
        resp = client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=msgs,
            temperature=0.6,
            max_tokens=120,
        )
        return resp.choices[0].message.content.strip()
    except Exception:
        return None

# -----------------------------
# Flask app
# -----------------------------
app = Flask(__name__)
app.secret_key = "SG_SECRET_123456"

# -----------------------------
# API: debug
# -----------------------------
@app.get("/api/debug")
def api_debug():
    return jsonify({
        "csv_exists": os.path.exists(CSV_PATH),
        "csv_path": CSV_PATH,
        "cwd": os.getcwd(),
        "movies_loaded": len(MOVIES),
        "sample_titles": [m.title for m in MOVIES[:5]],
        "openai_key_present": bool(OPENAI_API_KEY),
        "model": OPENAI_MODEL
    })

@app.get("/api/csv_info")
def api_csv_info():
    if not os.path.exists(CSV_PATH):
        return jsonify({"ok": False, "error": "CSV not found", "path": CSV_PATH})

    with open(CSV_PATH, "r", encoding="utf-8-sig", newline="") as f:
        first = f.readline().strip()
        second = f.readline().strip()

    return jsonify({
        "ok": True,
        "csv_path": CSV_PATH,
        "first_line": first[:300],
        "second_line": second[:300],
    })

@app.get("/api/nlu")
def api_nlu():
    text = request.args.get("text", "")
    offline = {
        "time": extract_time_smart(text),
        "mood": normalize_mood_smart(text),
        "brain": normalize_brain_smart(text),
        "q": extract_keywords_smart(text),
    }
    online = openai_nlu_profile(text)
    return jsonify({"offline": offline, "openai": online})

# -----------------------------
# API: recommendations (paged)
# -----------------------------
@app.get("/api/recs")
def api_recs():
    mood = request.args.get("mood", "porgos")
    brain = request.args.get("brain", "konnyu")
    time_limit = int(request.args.get("time", "120"))
    q = request.args.get("q", "")
    offset = int(request.args.get("offset", "0"))
    take = int(request.args.get("take", "12"))

    items = rank_movies(mood, time_limit, brain, q, offset, take)
    return jsonify({
        "total": len(MOVIES),
        "offset": offset,
        "take": take,
        "items": [
            {
                "title": m.title,
                "year": m.year,
                "minutes": m.minutes,
                "poster": m.poster,
                "trailer": m.trailer,
                "certification": m.certification,
                "why": m.why,
                "genres": m.genres,
                "tags": m.tags
            } for m in items
        ]
    })

# -----------------------------
# API: chat (smart flow)
# -----------------------------
@app.post("/api/chat")
def api_chat():
    data = request.get_json(force=True) or {}
    user_msg = (data.get("message") or "").strip()
    p = get_profile()
    low = user_msg.lower().strip()

    if not user_msg:
        q, quick = next_question_text(p)
        return jsonify({"assistant": q, "quick_replies": quick, "profile": p})

    GREETINGS = {"szia", "helo", "hello", "helló", "csá", "csa", "szevasz", "sziasztok", "hi"}

    # reset
    if low in {"reset", "uj", "új", "kezdjük újra", "restart"}:
        session["profile"] = default_profile()
        p = get_profile()
        q, quick = next_question_text(p)
        return jsonify({"assistant": "Oké, tiszta lap. 🙂 " + q, "quick_replies": quick, "profile": p})

    # Ha csak köszön, mindig új profil és kérdés (NEM ajánlunk filmet)
    if low in GREETINGS:
        session["profile"] = default_profile()
        p = get_profile()
        q, quick = next_question_text(p)
        return jsonify({"assistant": "Szia 🙂 " + q, "quick_replies": quick, "profile": p})

    # gombok (csak finoman)
    if low in {"sötétebb", "sotetebb"}:
        p["mood"] = "sotet"
    elif low == "viccesebb":
        p["mood"] = "vicces"
    elif low in {"rövidebb", "rovidebb"}:
        p["time"] = 90
    elif low in {"ajánlj", "ajanlj"}:
        p["ready"] = True

    # ---- NLU kitöltés (offline + opcionális OpenAI) ----
    t = extract_time_smart(user_msg)
    m = normalize_mood_smart(user_msg)
    b = normalize_brain_smart(user_msg)
    kw = extract_keywords_smart(user_msg)

    nlu = openai_nlu_profile(user_msg)
    if nlu:
        try:
            if nlu.get("time") is not None:
                t = int(nlu["time"])
        except Exception:
            pass
        if nlu.get("mood"):
            m = str(nlu["mood"]).strip()
        if nlu.get("brain"):
            b = str(nlu["brain"]).strip()
        if nlu.get("q"):
            kw = str(nlu["q"]).strip()

    if t:
        p["time"] = t
    if m:
        p["mood"] = m
    if b:
        p["brain"] = b

    # extra kulcsszavak (okosan)
    if kw and len(kw) >= 3:
        extra = (p.get("extra") or "")
        merged = (extra + " " + kw).strip()
        p["extra"] = merged[:240]

    # válasz szöveg (megtartjuk a stílust)
    ai_text = openai_chat_reply(p, user_msg)
    if not ai_text:
        missing = needs_questions(p)
        if missing:
            q, _quick = next_question_text(p)
            ai_text = q
        else:
            p["ready"] = True
            ai_text = (
                f"Oké, érzem a vibe-ot. (hangulat: {p['mood']}, idő: {p['time']} perc, mód: {p['brain']}). "
                f"Jobbra dobom a poszteres listát — nyomj „Tölts még”-et is. 🙂"
            )

    # history mentés
    hist = p.get("history") or []
    hist.append({"role": "user", "content": user_msg})
    hist.append({"role": "assistant", "content": ai_text})
    p["history"] = hist[-30:]
    session["profile"] = p

    # quick replies
    if needs_questions(p):
        _q, quick = next_question_text(p)
    else:
        quick = ["Ajánlj", "Újra dobás", "Sötétebb", "Viccesebb", "Rövidebb", "Reset"]

    return jsonify({"assistant": ai_text, "quick_replies": quick, "profile": p})

# -----------------------------
# UI (single page)
# -----------------------------
@app.get("/")
def home():
    return r"""
<!doctype html>
<html lang="hu">
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width, initial-scale=1" />
<title>SG Film Ajánló</title>
<style>
:root{
  --bg:#0b0d10;
  --panel:#10151c;
  --panel2:#0e1319;
  --line:#232b35;
  --text:#e9eef5;
  --muted:#a8b3c2;
  --gold:#d6b35a;
  --shadow: 0 12px 40px rgba(0,0,0,.45);
}
*{box-sizing:border-box}
body{
  margin:0;
  color:var(--text);
  font-family: ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, Arial;
  background:
    radial-gradient(1200px 700px at 15% 10%, #1a2230 0%, rgba(26,34,48,0) 55%),
    radial-gradient(1000px 700px at 90% 0%, #221b12 0%, rgba(34,27,18,0) 55%),
    var(--bg);
  padding:24px 14px 28px;
}
.wrap{max-width:1120px;margin:0 auto}
.top{
  display:flex;align-items:center;justify-content:space-between;gap:12px;flex-wrap:wrap;
  margin-bottom:14px;
}
.brand{display:flex;align-items:center;gap:12px;}
.badge{
  width:44px;height:44px;border-radius:14px;
  border:1px solid var(--line);
  background: linear-gradient(145deg, #141b25, #0b0f15);
  display:flex;align-items:center;justify-content:center;
  color:var(--gold);
  font-weight:900;
  letter-spacing:.8px;
}
h1{margin:0;font-size:28px;letter-spacing:.2px}
.sub{margin:6px 0 0;color:var(--muted);line-height:1.45}
.pills{display:flex;gap:8px;flex-wrap:wrap}
.pill{
  border:1px solid var(--line);
  background: rgba(16,21,28,.6);
  border-radius:999px;
  padding:8px 12px;
  color:var(--muted);
  font-size:12px;
}
.grid{
  display:grid;
  grid-template-columns: 1.05fr .95fr;
  gap:14px;
}
@media (max-width:980px){ .grid{grid-template-columns:1fr} }

.card{
  border:1px solid var(--line);
  background: rgba(16,21,28,.72);
  border-radius:18px;
  box-shadow: var(--shadow);
  overflow:hidden;
}
.cardHead{
  padding:14px 16px;
  border-bottom:1px solid var(--line);
  background: linear-gradient(180deg, rgba(18,24,34,.85), rgba(16,21,28,.4));
  display:flex;align-items:center;justify-content:space-between;gap:10px;
}
.cardTitle{font-weight:850;color:var(--gold)}
.cardBody{padding:14px 16px}

.chatBox{
  background: rgba(14,19,25,.75);
  border:1px solid var(--line);
  border-radius:16px;
  padding:12px;
  height:340px;
  overflow:auto;
}
.msg{margin:10px 0; display:flex; gap:10px}
.avatar{
  width:30px;height:30px;border-radius:12px;
  display:flex;align-items:center;justify-content:center;
  border:1px solid var(--line);
  background: linear-gradient(145deg, #141b25, #0b0f15);
  color:var(--gold);
  font-weight:900;
  flex:0 0 auto;
}
.bubble{
  max-width:82%;
  padding:10px 12px;
  border-radius:16px;
  border:1px solid var(--line);
  background: rgba(12,16,22,.8);
  line-height:1.4;
  white-space:pre-wrap;
}
.me .avatar{color:#e9eef5}
.me .bubble{ margin-left:auto; background: rgba(27,35,44,.7); }
.meta{ margin-top:10px; display:flex;gap:8px;flex-wrap:wrap;align-items:center;justify-content:space-between; }
.inputRow{ margin-top:10px; display:flex;gap:10px;flex-wrap:wrap; }
input{
  flex:1; min-width:220px;
  padding:12px 12px;
  border-radius:14px;
  border:1px solid var(--line);
  background: rgba(9,12,16,.9);
  color:var(--text);
  outline:none;
}
.btn{
  cursor:pointer;
  border:1px solid var(--line);
  background: linear-gradient(145deg, #1a2330, #0b0f15);
  color:var(--text);
  font-weight:800;
  padding:12px 14px;
  border-radius:14px;
}
.btn:hover{border-color:var(--gold)}
.chips{display:flex;gap:8px;flex-wrap:wrap;margin-top:10px}
.chip{
  cursor:pointer;
  border:1px solid var(--line);
  background: rgba(16,21,28,.65);
  color:var(--text);
  padding:8px 10px;
  border-radius:999px;
  font-weight:750;
  font-size:12px;
}
.chip:hover{border-color:var(--gold)}
.small{color:var(--muted); font-size:12px; line-height:1.45;}

#posterStrip{
  display:flex;
  gap:12px;
  overflow:auto;
  padding:12px;
  scroll-snap-type:x mandatory;
}
.posterCard{
  width:170px; flex:0 0 auto; scroll-snap-align:start;
  border:1px solid var(--line);
  border-radius:16px;
  overflow:hidden;
  background: rgba(12,16,22,.75);
  transition: transform .12s ease, border-color .12s ease;
}
.posterCard:hover{transform: translateY(-2px); border-color: var(--gold);}
.poster{
  width:100%;
  aspect-ratio: 2/3;
  background: radial-gradient(260px 200px at 50% 20%, rgba(214,179,90,.18), rgba(0,0,0,0) 55%), #0b0d10;
  display:flex;align-items:center;justify-content:center;
  overflow:hidden;
}
.poster img{width:100%;height:100%;object-fit:cover;display:block}
.posterFallback{
  font-weight:900;color:rgba(214,179,90,.85);
  letter-spacing:.4px;
  text-align:center;
  padding:0 10px;
}
.posterInfo{padding:10px 10px 12px}
.posterTitle{font-weight:900;color:var(--text);font-size:13px;line-height:1.2}
.posterMeta{margin-top:6px;color:var(--muted);font-size:12px}
.posterBtns{display:flex;gap:8px;margin-top:10px}
.posterBtns a{
  flex:1;
  text-align:center;
  padding:8px 10px;
  border-radius:12px;
  border:1px solid var(--line);
  background: rgba(16,21,28,.55);
  color:var(--text);
  text-decoration:none;
  font-weight:800;
  font-size:12px;
}
.posterBtns a:hover{border-color:var(--gold)}
.korhatar{
  display:inline-block;
  padding:4px 8px;
  border-radius:999px;
  border:1px solid var(--line);
  background: rgba(214,179,90,.12);
  color: var(--gold);
  font-weight:900;
  font-size:11px;
  margin-left:6px;
}
.footerNote{
  margin-top:12px;
  padding:10px 12px;
  border:1px solid var(--line);
  border-radius:14px;
  background: rgba(16,21,28,.55);
  color: var(--muted);
  font-size:12px;
}
.gold{color:var(--gold);font-weight:850}
</style>
</head>
<body>
<div class="wrap">
  <div class="top">
    <div class="brand">
      <div class="badge">SG</div>
      <div>
        <h1>SG Film Ajánló</h1>
        <div class="sub">AI chat + poszteres ajánlások — olyan élmény, amiért visszajössz.</div>
      </div>
    </div>
    <div class="pills">
      <div class="pill">Offline demo • local: <span class="gold">127.0.0.1</span></div>
      <div class="pill" id="pillLoaded">Filmek: …</div>
      <div class="pill">Nem stream / lejátszó oldal</div>
    </div>
  </div>

  <div class="grid">
    <div class="card">
      <div class="cardHead">
        <div class="cardTitle">AI chat</div>
        <button class="btn" id="btnReset">Reset</button>
      </div>
      <div class="cardBody">
        <div class="chatBox" id="chatBox"></div>
        <div class="chips" id="chips"></div>

        <div class="inputRow">
          <input id="inp" placeholder="Írj ide… (pl: 'csavaros thriller 2 óra')" />
          <button class="btn" id="btnSend">Küldés</button>
        </div>

        <div class="meta">
          <div class="small" id="statusLine">Kezdés: írj valamit — nem dobok rögtön filmeket “szia”-ra.</div>
          <div class="small">Tipp: <span class="gold">Ajánlj</span> / <span class="gold">Sötétebb</span> / <span class="gold">Rövidebb</span></div>
        </div>

        <div class="footerNote">
          <b>Fontos:</b> Nem vagyunk stream/filmlejátszó oldal. A trailer gomb külső (YouTube) oldalra visz.
          A poszterek/trailerek jogai a jogtulajdonosokat illetik.
        </div>
      </div>
    </div>

    <div class="card">
      <div class="cardHead">
        <div class="cardTitle">Ajánlott filmek (poszterek)</div>
        <button class="btn" id="btnMore">Tölts még</button>
      </div>
      <div class="cardBody" style="padding:0">
        <div id="posterStrip"></div>
      </div>
    </div>
  </div>
</div>

<script>
(function(){
  let state = {
    mood: "porgos",
    brain: "konnyu",
    time: 120,
    q: "",
    offset: 0,
    take: 12,
    ready: false
  };

  const chatBox = document.getElementById("chatBox");
  const chips = document.getElementById("chips");
  const inp = document.getElementById("inp");
  const statusLine = document.getElementById("statusLine");
  const posterStrip = document.getElementById("posterStrip");
  const pillLoaded = document.getElementById("pillLoaded");

  const btnSend = document.getElementById("btnSend");
  const btnReset = document.getElementById("btnReset");
  const btnMore = document.getElementById("btnMore");

  function escapeHtml(s){
    return (s||"")
      .replaceAll("&","&amp;")
      .replaceAll("<","&lt;")
      .replaceAll(">","&gt;")
      .replaceAll('"',"&quot;")
      .replaceAll("'","&#039;");
  }

  function addMsg(who, text){
    const row = document.createElement("div");
    row.className = "msg " + (who==="Te" ? "me" : "ai");
    const av = document.createElement("div");
    av.className = "avatar";
    av.textContent = who==="Te" ? "Te" : "SG";
    const b = document.createElement("div");
    b.className = "bubble";
    b.textContent = text || "";
    row.appendChild(av);
    row.appendChild(b);
    chatBox.appendChild(row);
    chatBox.scrollTop = chatBox.scrollHeight;
  }

  function setChips(arr){
    chips.innerHTML = "";
    (arr||[]).forEach(t=>{
      const c = document.createElement("div");
      c.className = "chip";
      c.textContent = t;
      c.onclick = ()=>sendChip(t);
      chips.appendChild(c);
    });
  }

  async function sendChip(t){
    inp.value = t;
    await send();
  }

  function posterCardHTML(m){
    const poster = (m.poster||"").trim();
    const trailer = (m.trailer||"").trim();
    const cert = (m.certification||"").trim();
    const certHtml = cert ? `<span class="korhatar">${escapeHtml(cert)}+</span>` : "";

    const posterHtml = `
  <img src="${escapeHtml(poster)}"
       alt="${escapeHtml(m.title)}"
       loading="lazy"
       onerror="this.style.display='none'; this.parentElement.querySelector('.posterFallback').style.display='flex';">
  <div class="posterFallback" style="display:none">${escapeHtml(m.title)}</div>
`;

    const ytBtn = trailer
      ? `<a href="${escapeHtml(trailer)}" target="_blank" rel="noopener">Trailer</a>`
      : `<a href="https://www.youtube.com/results?search_query=${encodeURIComponent(m.title + " trailer")}" target="_blank" rel="noopener">Trailer</a>`;

    return `
      <div class="posterCard">
        <div class="poster">${posterHtml}</div>
        <div class="posterInfo">
          <div class="posterTitle">${escapeHtml(m.title)} ${certHtml}</div>
          <div class="posterMeta">${escapeHtml(String(m.year))} • ${escapeHtml(String(m.minutes))} perc</div>
          <div class="posterBtns">
            ${ytBtn}
            <a href="#" data-title="${escapeHtml(m.title)}" data-why="${escapeHtml(m.why||"")}" class="whyBtn">Miért?</a>
          </div>
        </div>
      </div>`;
  }

  function bindWhyButtons(scope){
    const buttons = scope.querySelectorAll(".whyBtn");
    buttons.forEach(b=>{
      b.addEventListener("click", (ev)=>{
        ev.preventDefault();
        const title = b.getAttribute("data-title") || "";
        const why = b.getAttribute("data-why") || "—";
        alert(title + "\n\n" + why);
      });
    });
  }

  async function loadMore(){
    const url = `/api/recs?mood=${encodeURIComponent(state.mood)}&brain=${encodeURIComponent(state.brain)}&time=${encodeURIComponent(state.time)}&q=${encodeURIComponent(state.q||"")}&offset=${state.offset}&take=${state.take}`;
    const res = await fetch(url);
    const data = await res.json();

    pillLoaded.textContent = `Filmek: ${data.total || 0}`;
    const items = data.items || [];

    if(items.length === 0){
      const d = document.createElement("div");
      d.style.padding = "14px";
      d.style.color = "#a8b3c2";
      d.textContent = "Nincs több találat (vagy üres a CSV).";
      posterStrip.appendChild(d);
      return;
    }

    const chunk = document.createElement("div");
    chunk.style.display = "contents";
    chunk.innerHTML = items.map(posterCardHTML).join("");
    posterStrip.appendChild(chunk);
    bindWhyButtons(chunk);

    state.offset += items.length;
  }

  async function send(){
    const msg = (inp.value||"").trim();
    if(!msg) return;

    addMsg("Te", msg);
    inp.value = "";

    const typing = document.createElement("div");
    typing.className = "msg ai";
    typing.innerHTML = `<div class="avatar">SG</div><div class="bubble">…</div>`;
    chatBox.appendChild(typing);
    chatBox.scrollTop = chatBox.scrollHeight;

    const res = await fetch("/api/chat", {
      method:"POST",
      headers: {"Content-Type":"application/json"},
      body: JSON.stringify({message: msg})
    });

    const data = await res.json();
    typing.remove();

    addMsg("SG", data.assistant || "…");
    setChips(data.quick_replies || []);

    if(data.profile){
      state.mood = data.profile.mood || state.mood;
      state.brain = data.profile.brain || state.brain;
      state.time = data.profile.time || state.time;
      state.q = data.profile.extra || "";
      state.ready = !!(data.profile && data.profile.ready) || !!data.ready;
    }

    statusLine.textContent =
      `Beállítás: mood=${state.mood} • time≈${state.time} • mód=${state.brain} • kulcsszó=${(state.q||"").slice(0,30)}`;

    if(state.ready){
      state.offset = 0;
      posterStrip.innerHTML = "";
      await loadMore();
    }
  }

  // events
  btnSend.addEventListener("click", send);
  btnReset.addEventListener("click", ()=>sendChip("Reset"));
  btnMore.addEventListener("click", loadMore);
  inp.addEventListener("keydown", (e)=>{ if(e.key==="Enter") send(); });

  // init
  addMsg("SG", "Szia. Milyen filmet néznél ma? Írj példát: „csavaros thriller 2 óra”.");
  setChips(["90 perc","120 perc","180 perc","sötét","vicces","Ajánlj","Reset"]);

  fetch("/api/debug").then(r=>r.json()).then(d=>{
    pillLoaded.textContent = `Filmek: ${d.movies_loaded || 0}`;
  }).catch(()=>{});
})();
</script>
</body>
</html>
"""

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)

Behbud Malikov <behbud.malikov@gmail.com> ezt írta (időpont: 2026. febr. 28., Szo, 14:44):

Behbud Malikov
Mellékletek
15:27 (7 perccel ezelőtt)
címzett: én

Úgy tűnik, hogy ez az üzenet angol nyelvű
 Egy melléklet
  •  A Gmail által ellenőrizve

Behbud Malikov
Mellékletek
15:34 (0 perccel ezelőtt)
címzett: én

Úgy tűnik, hogy ez az üzenet angol nyelvű
 Egy melléklet
  •  A Gmail által ellenőrizve
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

MOVIES: List[Movie] = load_movies_csv(CSV_PATH)

# ---------------------------------------------------------------------------
# Offline NLU
# ---------------------------------------------------------------------------
MOOD_SYNONYMS: Dict[str, List[str]] = {
    "porgos":      ["pörg", "akció", "gyors", "üldöz", "harc", "bosszú", "adrenalin", "darál"],
    "nyugis":      ["nyugis", "chill", "feel", "laza", "szívmelenget", "romi", "romant"],
    "sotet":       ["söt", "sot", "thriller", "krimi", "parás", "nyomaszt", "gyilk", "pszich"],
    "felemelo":    ["felem", "motiv", "inspir", "pozit", "kitart", "remény", "remeny"],
    "vicces":      ["vicc", "kom", "nevet", "őrült", "orult", "paród", "parod", "humor"],
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
    "porgos":    ["akció", "gyors", "pörgős", "harc", "üldözés", "bosszú"],
    "nyugis":    ["feel-good", "nyugis", "laza", "szívmelengető", "élet", "család"],
    "sotet":     ["thriller", "sötét", "csavaros", "pszichológiai", "krimi", "bűn"],
    "felemelo":  ["motiváló", "felemelő", "inspiráló", "kitartás", "sport", "cél"],
    "vicces":    ["komédia", "vicces", "nevetős", "baráti", "őrült", "laza"],
}

BRAIN_KEYWORDS: Dict[str, List[str]] = {
    "konnyu":          ["könnyű", "laza", "nem akarok gondolkodni"],
    "kozepes":         ["közepes"],
    "elgondolkodtato": ["elgondolkodtató", "agyas", "agyalós", "csavaros", "pszichológiai"],
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

    score += random.randint(0, cfg.RANDOM_BONUS)
    return score

def rank_movies(mood: str, time_limit: int, brain: str, extra: str, offset: int, take: int) -> List[Movie]:
    ranked = sorted(
        MOVIES,
        key=lambda m: score_movie(m, mood, time_limit, brain, extra),
        reverse=True,
    )
    return ranked[offset : offset + take]

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
            'Oké, megvan minden. Finomítsunk: írj 1-2 kulcsszót (pl. maffia / űr / csavaros / bosszú) vagy nyomj \u201eAjánlj\u201d-t.',
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

_GREETINGS = {"szia", "helo", "hello", "helló", "csá", "csa", "szevasz", "sziasztok", "hi"}
_RESET_CMDS = {"reset", "uj", "új", "kezdjük újra", "restart"}

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
                "why":           m.why,
                "genres":        m.genres,
                "tags":          m.tags,
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

    # Build reply text
    ai_text = openai_chat_reply(p, user_msg)
    if not ai_text:
        if missing_fields(p):
            ai_text, _ = next_question(p)
        else:
            p["ready"] = True
            ai_text = (
                f"Oké, érzem a vibe-ot. "
                f"(hangulat: {p['mood']}, idő: {p['time']} perc, mód: {p['brain']}). "
                f"Jobbra dobom a poszteres list\u00e1t \u2014 nyomj 'T\u00f6lts m\u00e9g'-et is. :)"
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
# Frontend — single-page app
# ---------------------------------------------------------------------------
_HTML = r"""<!doctype html>
<html lang="hu">
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width, initial-scale=1" />
<title>SG Film Ajánló</title>
<link rel="preconnect" href="https://fonts.googleapis.com" />
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin />
<link href="https://fonts.googleapis.com/css2?family=DM+Serif+Display:ital@0;1&family=DM+Sans:ital,opsz,wght@0,9..40,300..700;1,9..40,300..700&display=swap" rel="stylesheet" />
<style>
/* ─────────────── Variables ─────────────── */
:root {
  --bg:       #080a0d;
  --surface:  #0e1218;
  --surface2: #131820;
  --border:   #1e2730;
  --border2:  #2a3441;
  --text:     #dde4ed;
  --muted:    #7a8a9a;
  --faint:    #3a4a5a;
  --gold:     #c8a84b;
  --gold2:    #e8c86c;
  --red:      #c84b4b;
  --shadow:   0 20px 60px rgba(0,0,0,.6);
  --radius:   16px;
  --font-serif: 'DM Serif Display', Georgia, serif;
  --font-sans:  'DM Sans', system-ui, sans-serif;
}

/* ─────────────── Reset ─────────────── */
*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
html { scroll-behavior: smooth; }
body {
  background: var(--bg);
  color: var(--text);
  font-family: var(--font-sans);
  font-size: 14px;
  line-height: 1.55;
  min-height: 100vh;
  padding: 28px 16px 48px;
  background-image:
    radial-gradient(ellipse 900px 500px at 10% 0%, rgba(200,168,75,.06) 0%, transparent 70%),
    radial-gradient(ellipse 700px 400px at 90% 100%, rgba(60,40,20,.15) 0%, transparent 70%);
}

/* ─────────────── Film strip decoration ─────────────── */
body::before {
  content: '';
  position: fixed;
  top: 0; left: 0; right: 0;
  height: 4px;
  background: repeating-linear-gradient(
    90deg,
    var(--gold) 0px, var(--gold) 14px,
    transparent 14px, transparent 22px
  );
  opacity: .35;
  pointer-events: none;
  z-index: 100;
}

/* ─────────────── Layout ─────────────── */
.wrap { max-width: 1160px; margin: 0 auto; }

/* ─────────────── Header ─────────────── */
.header {
  display: flex;
  align-items: flex-start;
  justify-content: space-between;
  gap: 16px;
  flex-wrap: wrap;
  margin-bottom: 24px;
  padding-bottom: 20px;
  border-bottom: 1px solid var(--border);
}
.brand { display: flex; align-items: center; gap: 14px; }
.logo {
  width: 52px; height: 52px;
  border-radius: 14px;
  border: 1px solid var(--border2);
  background: linear-gradient(145deg, #1a2230, #0a0d12);
  display: flex; align-items: center; justify-content: center;
  flex-shrink: 0;
  box-shadow: inset 0 1px 0 rgba(255,255,255,.04), 0 4px 16px rgba(0,0,0,.4);
}
.logo span {
  font-family: var(--font-serif);
  font-size: 18px;
  color: var(--gold);
  letter-spacing: 1px;
}
.brand-text h1 {
  font-family: var(--font-serif);
  font-size: 30px;
  letter-spacing: .3px;
  line-height: 1.1;
  color: var(--text);
}
.brand-text h1 em {
  font-style: italic;
  color: var(--gold);
}
.brand-text p {
  color: var(--muted);
  font-size: 13px;
  margin-top: 3px;
}
.badges {
  display: flex;
  gap: 8px;
  flex-wrap: wrap;
  align-items: center;
  margin-top: 6px;
}
.badge {
  border: 1px solid var(--border);
  background: rgba(14,18,24,.7);
  border-radius: 999px;
  padding: 6px 12px;
  color: var(--muted);
  font-size: 11px;
  letter-spacing: .3px;
}
.badge .hl { color: var(--gold); }

/* ─────────────── Main grid ─────────────── */
.grid {
  display: grid;
  grid-template-columns: 1.1fr 0.9fr;
  gap: 16px;
  align-items: start;
}
@media (max-width: 900px) { .grid { grid-template-columns: 1fr; } }

/* ─────────────── Cards ─────────────── */
.card {
  border: 1px solid var(--border);
  background: rgba(14,18,24,.8);
  border-radius: var(--radius);
  box-shadow: var(--shadow);
  overflow: hidden;
  backdrop-filter: blur(8px);
}
.card-head {
  padding: 14px 18px;
  border-bottom: 1px solid var(--border);
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 10px;
  background: linear-gradient(180deg, rgba(22,30,42,.9) 0%, rgba(14,18,24,.6) 100%);
}
.card-title {
  font-family: var(--font-serif);
  font-size: 16px;
  color: var(--gold);
  letter-spacing: .2px;
}
.card-body { padding: 16px 18px; }

/* ─────────────── Buttons ─────────────── */
.btn {
  cursor: pointer;
  border: 1px solid var(--border2);
  background: linear-gradient(145deg, #1c2840, #0c1020);
  color: var(--text);
  font-family: var(--font-sans);
  font-weight: 600;
  font-size: 13px;
  padding: 10px 16px;
  border-radius: 12px;
  transition: border-color .15s, background .15s, transform .1s;
  white-space: nowrap;
}
.btn:hover {
  border-color: var(--gold);
  background: linear-gradient(145deg, #212e48, #0e1428);
}
.btn:active { transform: scale(.97); }

/* ─────────────── Chat ─────────────── */
.chat-box {
  background: rgba(8,10,14,.8);
  border: 1px solid var(--border);
  border-radius: 14px;
  padding: 14px;
  height: 360px;
  overflow-y: auto;
  scroll-behavior: smooth;
}
.chat-box::-webkit-scrollbar { width: 4px; }
.chat-box::-webkit-scrollbar-thumb { background: var(--border2); border-radius: 2px; }

.msg { display: flex; gap: 10px; margin: 12px 0; animation: fadeUp .2s ease; }
@keyframes fadeUp {
  from { opacity: 0; transform: translateY(6px); }
  to   { opacity: 1; transform: translateY(0); }
}
.avatar {
  width: 32px; height: 32px; border-radius: 10px;
  border: 1px solid var(--border2);
  background: linear-gradient(145deg, #1a2230, #0c1018);
  display: flex; align-items: center; justify-content: center;
  flex-shrink: 0;
  font-size: 10px;
  font-weight: 700;
  letter-spacing: .5px;
  color: var(--gold);
}
.msg.me .avatar { color: var(--muted); }
.bubble {
  max-width: 80%;
  padding: 10px 14px;
  border-radius: 14px;
  border: 1px solid var(--border);
  background: rgba(14,18,24,.9);
  line-height: 1.5;
  white-space: pre-wrap;
  font-size: 13.5px;
}
.msg.me .bubble {
  margin-left: auto;
  background: rgba(28,36,52,.8);
  border-color: var(--border2);
}

/* Typing indicator */
.typing-dots span {
  display: inline-block;
  width: 5px; height: 5px;
  border-radius: 50%;
  background: var(--gold);
  margin: 0 2px;
  animation: blink 1.2s infinite;
}
.typing-dots span:nth-child(2) { animation-delay: .2s; }
.typing-dots span:nth-child(3) { animation-delay: .4s; }
@keyframes blink {
  0%, 80%, 100% { opacity: .2; transform: scale(.8); }
  40% { opacity: 1; transform: scale(1); }
}

/* Chips */
.chips {
  display: flex;
  gap: 8px;
  flex-wrap: wrap;
  margin-top: 12px;
}
.chip {
  cursor: pointer;
  border: 1px solid var(--border);
  background: rgba(14,18,24,.7);
  color: var(--text);
  padding: 7px 12px;
  border-radius: 999px;
  font-size: 12px;
  font-weight: 500;
  transition: border-color .15s, background .15s;
}
.chip:hover {
  border-color: var(--gold);
  background: rgba(200,168,75,.08);
  color: var(--gold2);
}

/* Input row */
.input-row {
  display: flex;
  gap: 10px;
  margin-top: 14px;
}
.chat-input {
  flex: 1;
  min-width: 0;
  padding: 11px 14px;
  border-radius: 12px;
  border: 1px solid var(--border2);
  background: rgba(8,10,14,.9);
  color: var(--text);
  font-family: var(--font-sans);
  font-size: 13.5px;
  outline: none;
  transition: border-color .15s;
}
.chat-input:focus { border-color: var(--gold); }
.chat-input::placeholder { color: var(--faint); }

/* Status */
.status-bar {
  margin-top: 12px;
  display: flex;
  gap: 8px;
  flex-wrap: wrap;
  justify-content: space-between;
  align-items: center;
}
.status-text { color: var(--muted); font-size: 11px; }
.status-hint { color: var(--muted); font-size: 11px; }
.status-hint .hl { color: var(--gold); }

.footer-note {
  margin-top: 14px;
  padding: 10px 14px;
  border: 1px solid var(--border);
  border-radius: 12px;
  background: rgba(14,18,24,.6);
  color: var(--muted);
  font-size: 11px;
}

/* ─────────────── Poster strip ─────────────── */
#poster-strip {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(148px, 1fr));
  gap: 12px;
  padding: 14px;
  max-height: 600px;
  overflow-y: auto;
}
#poster-strip::-webkit-scrollbar { width: 4px; }
#poster-strip::-webkit-scrollbar-thumb { background: var(--border2); border-radius: 2px; }

.poster-card {
  border: 1px solid var(--border);
  border-radius: 14px;
  overflow: hidden;
  background: rgba(10,14,20,.85);
  transition: transform .18s ease, border-color .18s ease, box-shadow .18s ease;
  animation: fadeUp .25s ease;
}
.poster-card:hover {
  transform: translateY(-4px);
  border-color: var(--gold);
  box-shadow: 0 8px 32px rgba(200,168,75,.1);
}

.poster-img {
  width: 100%;
  aspect-ratio: 2/3;
  background:
    radial-gradient(ellipse 140% 100% at 50% 0%, rgba(200,168,75,.1), transparent 60%),
    #0a0c10;
  display: flex;
  align-items: center;
  justify-content: center;
  overflow: hidden;
  position: relative;
}
.poster-img img {
  width: 100%; height: 100%;
  object-fit: cover;
  display: block;
  transition: transform .3s ease;
}
.poster-card:hover .poster-img img { transform: scale(1.04); }
.poster-fallback {
  font-family: var(--font-serif);
  font-size: 12px;
  color: rgba(200,168,75,.7);
  text-align: center;
  padding: 10px;
  line-height: 1.3;
}

.cert-badge {
  position: absolute;
  top: 8px; right: 8px;
  padding: 3px 7px;
  border-radius: 6px;
  background: rgba(200,168,75,.15);
  border: 1px solid rgba(200,168,75,.3);
  color: var(--gold);
  font-size: 10px;
  font-weight: 700;
}

.poster-info { padding: 10px 10px 12px; }
.poster-title {
  font-weight: 700;
  font-size: 12.5px;
  line-height: 1.25;
  color: var(--text);
  margin-bottom: 4px;
}
.poster-meta { color: var(--muted); font-size: 11px; margin-bottom: 8px; }
.poster-btns { display: flex; gap: 6px; }
.poster-btns a {
  flex: 1;
  text-align: center;
  padding: 7px 6px;
  border-radius: 10px;
  border: 1px solid var(--border);
  background: rgba(14,18,24,.6);
  color: var(--text);
  text-decoration: none;
  font-size: 11px;
  font-weight: 600;
  transition: border-color .15s, background .15s;
}
.poster-btns a:hover { border-color: var(--gold); background: rgba(200,168,75,.08); }
.poster-btns a.why-btn { color: var(--gold); }

/* Empty state */
.empty-state {
  grid-column: 1/-1;
  padding: 32px;
  text-align: center;
  color: var(--muted);
  font-size: 13px;
}

/* ─────────────── Modal ─────────────── */
.modal-backdrop {
  display: none;
  position: fixed;
  inset: 0;
  background: rgba(4,6,10,.85);
  backdrop-filter: blur(6px);
  z-index: 200;
  align-items: center;
  justify-content: center;
  padding: 24px;
  animation: fadeIn .15s ease;
}
.modal-backdrop.open { display: flex; }
@keyframes fadeIn { from { opacity: 0; } to { opacity: 1; } }

.modal {
  background: var(--surface2);
  border: 1px solid var(--border2);
  border-radius: 20px;
  max-width: 480px;
  width: 100%;
  box-shadow: 0 32px 80px rgba(0,0,0,.7);
  overflow: hidden;
  animation: slideUp .18s ease;
}
@keyframes slideUp {
  from { transform: translateY(20px); opacity: 0; }
  to   { transform: translateY(0);    opacity: 1; }
}
.modal-head {
  padding: 18px 20px 14px;
  border-bottom: 1px solid var(--border);
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 12px;
}
.modal-title {
  font-family: var(--font-serif);
  font-size: 18px;
  color: var(--gold);
}
.modal-close {
  cursor: pointer;
  width: 28px; height: 28px;
  border-radius: 8px;
  border: 1px solid var(--border);
  background: transparent;
  color: var(--muted);
  font-size: 16px;
  display: flex; align-items: center; justify-content: center;
  transition: border-color .15s, color .15s;
}
.modal-close:hover { border-color: var(--red); color: var(--red); }
.modal-body {
  padding: 20px;
  color: var(--text);
  font-size: 14px;
  line-height: 1.65;
}
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
        <p>AI chat + intelligens poszteres ajánlások</p>
      </div>
    </div>
    <div class="badges">
      <div class="badge">Local: <span class="hl">127.0.0.1:5000</span></div>
      <div class="badge" id="pill-loaded">Filmek: …</div>
      <div class="badge">Nem stream oldal</div>
    </div>
  </div>

  <!-- Main grid -->
  <div class="grid">

    <!-- Chat card -->
    <div class="card">
      <div class="card-head">
        <div class="card-title">AI Asszisztens</div>
        <button class="btn" id="btn-reset">↺ Reset</button>
      </div>
      <div class="card-body">
        <div class="chat-box" id="chat-box"></div>
        <div class="chips" id="chips"></div>
        <div class="input-row">
          <input class="chat-input" id="inp"
            placeholder="pl. „csavaros thriller 2 óra"" autocomplete="off" />
          <button class="btn" id="btn-send">Küldés</button>
        </div>
        <div class="status-bar">
          <div class="status-text" id="status-line">Kezdj el írni — nem dobok rögtön filmeket üdvözlésre.</div>
          <div class="status-hint">Tipp: <span class="hl">Ajánlj</span> · <span class="hl">Sötétebb</span> · <span class="hl">Rövidebb</span></div>
        </div>
        <div class="footer-note">
          <b>Fontos:</b> Nem vagyunk stream vagy lejátszó oldal.
          A trailer gomb YouTube-ra visz. A poszterek/trailerek jogai a jogtulajdonosokat illetik.
        </div>
      </div>
    </div>

    <!-- Poster card -->
    <div class="card">
      <div class="card-head">
        <div class="card-title">Ajánlott filmek</div>
        <button class="btn" id="btn-more">+ Tölts még</button>
      </div>
      <div class="card-body" style="padding:0">
        <div id="poster-strip"></div>
      </div>
    </div>

  </div>
</div>

<!-- Why? Modal -->
<div class="modal-backdrop" id="modal">
  <div class="modal">
    <div class="modal-head">
      <div class="modal-title" id="modal-title">Miért ajánlott?</div>
      <button class="modal-close" id="modal-close">✕</button>
    </div>
    <div class="modal-body" id="modal-body"></div>
  </div>
</div>

<script>
(function () {
  'use strict';

  /* ─── state ─── */
  let state = { mood: 'porgos', brain: 'konnyu', time: 120, q: '', offset: 0, take: 12, ready: false };

  /* ─── refs ─── */
  const chatBox   = document.getElementById('chat-box');
  const chips     = document.getElementById('chips');
  const inp       = document.getElementById('inp');
  const statusLine = document.getElementById('status-line');
  const posterStrip = document.getElementById('poster-strip');
  const pillLoaded  = document.getElementById('pill-loaded');
  const modal       = document.getElementById('modal');
  const modalTitle  = document.getElementById('modal-title');
  const modalBody   = document.getElementById('modal-body');

  /* ─── helpers ─── */
  function esc(s) {
    return String(s || '')
      .replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;')
      .replace(/"/g,'&quot;').replace(/'/g,'&#039;');
  }

  /* ─── Modal ─── */
  function openModal(title, text) {
    modalTitle.textContent = title;
    modalBody.textContent  = text || '—';
    modal.classList.add('open');
  }
  function closeModal() { modal.classList.remove('open'); }

  document.getElementById('modal-close').addEventListener('click', closeModal);
  modal.addEventListener('click', e => { if (e.target === modal) closeModal(); });
  document.addEventListener('keydown', e => { if (e.key === 'Escape') closeModal(); });

  /* ─── Chat ─── */
  function addMsg(who, html, isHtml = false) {
    const row = document.createElement('div');
    row.className = 'msg ' + (who === 'Te' ? 'me' : 'ai');

    const av = document.createElement('div');
    av.className = 'avatar';
    av.textContent = who === 'Te' ? 'Te' : 'SG';

    const bub = document.createElement('div');
    bub.className = 'bubble';
    if (isHtml) bub.innerHTML = html;
    else bub.textContent = html || '';

    row.append(av, bub);
    chatBox.appendChild(row);
    chatBox.scrollTop = chatBox.scrollHeight;
    return row;
  }

  function addTyping() {
    const row = document.createElement('div');
    row.className = 'msg ai';
    row.innerHTML = `<div class="avatar">SG</div>
      <div class="bubble"><div class="typing-dots">
        <span></span><span></span><span></span>
      </div></div>`;
    chatBox.appendChild(row);
    chatBox.scrollTop = chatBox.scrollHeight;
    return row;
  }

  /* ─── Chips ─── */
  function setChips(arr) {
    chips.innerHTML = '';
    (arr || []).forEach(t => {
      const c = document.createElement('div');
      c.className = 'chip';
      c.textContent = t;
      c.addEventListener('click', () => sendChip(t));
      chips.appendChild(c);
    });
  }

  async function sendChip(t) { inp.value = t; await send(); }

  /* ─── Posters ─── */
  function posterHTML(m) {
    const poster  = (m.poster  || '').trim();
    const trailer = (m.trailer || '').trim();
    const cert    = (m.certification || '').trim();

    const imgSrc = poster
      ? `<img src="${esc(poster)}" alt="${esc(m.title)}" loading="lazy"
           onerror="this.style.display='none';this.nextElementSibling.style.display='flex'">`
      : '';
    const fallback = `<div class="poster-fallback" ${poster ? 'style="display:none"' : ''}>${esc(m.title)}</div>`;
    const certBadge = cert ? `<div class="cert-badge">${esc(cert)}+</div>` : '';

    const ytUrl = trailer
      ? esc(trailer)
      : `https://www.youtube.com/results?search_query=${encodeURIComponent(m.title + ' trailer')}`;

    return `
    <div class="poster-card">
      <div class="poster-img">
        ${imgSrc}${fallback}${certBadge}
      </div>
      <div class="poster-info">
        <div class="poster-title">${esc(m.title)}</div>
        <div class="poster-meta">${esc(String(m.year))} · ${esc(String(m.minutes))} perc</div>
        <div class="poster-btns">
          <a href="${ytUrl}" target="_blank" rel="noopener">▶ Trailer</a>
          <a href="#" class="why-btn"
             data-title="${esc(m.title)}"
             data-why="${esc(m.why || '')}">? Miért</a>
        </div>
      </div>
    </div>`;
  }

  function bindWhyButtons(scope) {
    scope.querySelectorAll('.why-btn').forEach(btn => {
      btn.addEventListener('click', e => {
        e.preventDefault();
        openModal(btn.dataset.title, btn.dataset.why || '—');
      });
    });
  }

  async function loadMore() {
    const url = `/api/recs?mood=${encodeURIComponent(state.mood)}`
      + `&brain=${encodeURIComponent(state.brain)}`
      + `&time=${encodeURIComponent(state.time)}`
      + `&q=${encodeURIComponent(state.q || '')}`
      + `&offset=${state.offset}&take=${state.take}`;

    const res  = await fetch(url);
    const data = await res.json();

    pillLoaded.textContent = `Filmek: ${data.total || 0}`;
    const items = data.items || [];

    if (!items.length) {
      const el = document.createElement('div');
      el.className = 'empty-state';
      el.textContent = 'Nincs több találat (vagy üres a CSV).';
      posterStrip.appendChild(el);
      return;
    }

    const chunk = document.createElement('div');
    chunk.style.display = 'contents';
    chunk.innerHTML = items.map(posterHTML).join('');
    posterStrip.appendChild(chunk);
    bindWhyButtons(chunk);
    state.offset += items.length;
  }

  /* ─── Send ─── */
  async function send() {
    const msg = (inp.value || '').trim();
    if (!msg) return;

    addMsg('Te', msg);
    inp.value = '';

    const typing = addTyping();

    try {
      const res  = await fetch('/api/chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ message: msg }),
      });
      const data = await res.json();
      typing.remove();

      addMsg('SG', data.assistant || '…');
      setChips(data.quick_replies || []);

      const prof = data.profile || {};
      if (prof.mood)  state.mood  = prof.mood;
      if (prof.brain) state.brain = prof.brain;
      if (prof.time)  state.time  = prof.time;
      state.q     = prof.extra || '';
      state.ready = !!(prof.ready || data.ready);

      statusLine.textContent =
        `mood=${state.mood} · time≈${state.time}min · mód=${state.brain}`
        + (state.q ? ` · kulcsszó: ${state.q.slice(0, 30)}` : '');

      if (state.ready) {
        state.offset = 0;
        posterStrip.innerHTML = '';
        await loadMore();
      }
    } catch (err) {
      typing.remove();
      addMsg('SG', '⚠ Hálózati hiba. Próbáld újra.');
      console.error(err);
    }
  }

  /* ─── Events ─── */
  document.getElementById('btn-send').addEventListener('click', send);
  document.getElementById('btn-reset').addEventListener('click', () => sendChip('Reset'));
  document.getElementById('btn-more').addEventListener('click', loadMore);
  inp.addEventListener('keydown', e => { if (e.key === 'Enter') send(); });

  /* ─── Init ─── */
  addMsg('SG', 'Szia. Milyen filmet néznél ma? Írj példát: „csavaros thriller 2 óra".');
  setChips(['90 perc', '120 perc', '180 perc', 'sötét', 'vicces', 'Ajánlj', 'Reset']);

  fetch('/api/debug')
    .then(r => r.json())
    .then(d => { pillLoaded.textContent = `Filmek: ${d.movies_loaded || 0}`; })
    .catch(() => {});
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
        host="127.0.0.1",
        port=Config.FLASK_PORT,
        debug=Config.FLASK_DEBUG,
    )