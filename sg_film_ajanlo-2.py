"""SG Film Ajánló — tiszta újraírás"""
from __future__ import annotations
import csv, json, logging, os, random, re, sqlite3, secrets
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
from flask import Flask, jsonify, request, session, redirect, Response
import urllib.request as _ureq
import urllib.parse as _uparse
 
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("sg")
 
# --- .env betöltés ---
def _load_env(path=".env"):
    if not os.path.exists(path): return
    for line in open(path, encoding="utf-8"):
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line: continue
        k, _, v = line.partition("=")
        os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))
_load_env()
 
TMDB_KEY    = os.getenv("TMDB_API_KEY","")
G_CLIENT_ID = os.getenv("GOOGLE_CLIENT_ID","")
G_SECRET    = os.getenv("GOOGLE_CLIENT_SECRET","")
G_REDIRECT  = os.getenv("GOOGLE_REDIRECT_URI","https://sg-film-ajanlo.onrender.com/auth/google/callback")
FLASK_SECRET= os.getenv("FLASK_SECRET","SG_DEV_SECRET_CHANGE_ME")
 
# --- Adatmodell ---
@dataclass
class Movie:
    title: str; year: int; minutes: int
    genres: List[str]; tags: List[str]
    poster: str = ""; trailer: str = ""
    avg_rating: float = 0.0; rating_count: int = 0
    tmdb_id: str = ""
 
# --- DB betöltés ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH  = os.path.join(BASE_DIR, "movies.db")
 
def load_movies() -> List[Movie]:
    if not os.path.exists(DB_PATH):
        log.warning("movies.db nem található!")
        return []
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    rows = conn.execute("""
        SELECT title,year,minutes,genres,tags,poster,trailer,
               COALESCE(avg_rating,0) avg_rating,
               COALESCE(rating_count,0) rating_count,
               COALESCE(tmdb_id,'') tmdb_id
        FROM movies ORDER BY title
    """).fetchall()
    conn.close()
    def sp(v): return [x.strip().lower() for x in (v or "").split("|") if x.strip()]
    return [Movie(
        title=r["title"], year=int(r["year"] or 0), minutes=int(r["minutes"] or 0),
        genres=sp(r["genres"]), tags=sp(r["tags"]),
        poster=r["poster"] or "", trailer=r["trailer"] or "",
        avg_rating=float(r["avg_rating"]), rating_count=int(r["rating_count"]),
        tmdb_id=r["tmdb_id"]
    ) for r in rows]
 
MOVIES = load_movies()
log.info("Betöltve: %d film", len(MOVIES))
 
# --- Scoring ---
MOOD_KW = {
    "porgos":   ["action","adventure","thriller","crime","war","fight","chase","gun","battle"],
    "nyugis":   ["comedy","family","animation","musical","feel-good","friendship","gentle","light"],
    "sotet":    ["thriller","horror","mystery","crime","noir","dark","psychological","suspense","murder"],
    "felemelo": ["drama","biography","sport","music","history","inspiring","triumph","overcome"],
    "vicces":   ["comedy","animation","family","parody","humor","funny","witty","slapstick"],
    "romantic": ["romance","romantic","love","wedding","relationship","szerelem","szerelmes"],
}
BRAIN_KW = {
    "konnyu":          ["comedy","animation","family","simple","fun","light","feel-good"],
    "kozepes":         ["drama","adventure","action"],
    "elgondolkodtato": ["psychological","mystery","twist","mind","complex","philosophical","sci-fi","thought-provoking"],
}
ERA_FILTER = {
    "recent":  lambda m: m.year >= 2015,
    "new":     lambda m: 2000 <= m.year < 2015,
    "classic": lambda m: 1980 <= m.year < 2000,
    "old":     lambda m: 0 < m.year < 1980,
    "all":     lambda m: True,
}
EXTRA_KW = {
    # intenzitás
    "enyhén sötét, nem ijeszt":          "mystery thriller mild",
    "közepes — feszült de nem horror":   "thriller suspense",
    "igazán durva és nyomasztó":         "horror dark disturbing brutal",
    "tiszta akció, minimális vér":       "action adventure clean",
    "közepes — van némi brutalitás":     "action crime thriller",
    "brutális és nyers":                 "brutal raw violence gritty",
    "ártatlan, mindenki nevet":          "family comedy fun",
    "fekete humor, csípős poénok":       "dark comedy satire",
    "abszurd és őrült":                  "absurd comedy quirky",
    "kicsit megható, de nem sírok":      "drama inspiring uplifting",
    "erősen érzelmes, könnyeket csal":   "emotional tearjerker drama",
    "inkább inspiráló mint érzelmes":    "inspiring motivating triumph",
    "vidám és szórakoztató":             "comedy fun entertainment light",
    "szívmelengető és meghitt":          "heartwarming feel-good friendship",
    "kalandos de könnyed":               "adventure light fun family",
    "édes és boldog szerelem":           "romance happy love sweet",
    "drámai és tragikus románc":         "tragic romance drama",
    "bonyolult kapcsolat sok fordulattal": "complex relationship drama twist",
    "enyhe és könnyű":                   "feel-good light comedy",
    "közepes intenzitású":               "drama thriller",
    "nagyon intenzív és erős":           "intense powerful drama",
}
 
def score(m: Movie, mood: str, time_limit: int, brain: str, extra: str) -> int:
    s = 0
    blob = f"{m.title} {' '.join(m.tags)} {' '.join(m.genres)}".lower()
    # idő
    if m.minutes and time_limit:
        diff = abs(m.minutes - time_limit)
        if diff<=10: s+=10
        elif diff<=25: s+=7
        elif diff<=45: s+=4
        elif diff<=80: s+=1
        else: s-=2
    # hangulat
    for w in MOOD_KW.get(mood, []):
        if w in blob: s+=3
    # brain
    heavy = BRAIN_KW["elgondolkodtato"]
    if brain=="konnyu":
        if any(w in blob for w in heavy): s-=2
    elif brain=="elgondolkodtato":
        if any(w in blob for w in heavy): s+=4
    # extra
    if extra:
        matched=0
        for tok in extra.lower().split():
            if len(tok)<3: continue
            if tok in blob: s+=4; matched+=1
            elif any(tok in t for t in m.tags): s+=2; matched+=1
        if matched>=3: s+=5
        elif matched>=2: s+=2
    # rating bónusz
    if m.avg_rating>=4.0 and m.rating_count>=50: s+=3
    elif m.avg_rating>=3.5 and m.rating_count>=20: s+=1
    s += random.randint(0,2)
    return s
 
def rank(mood,time_limit,brain,extra,offset,take,era="all"):
    fn = ERA_FILTER.get(era, ERA_FILTER["all"])
    pool = [m for m in MOVIES if fn(m)]
    scored = sorted([(score(m,mood,time_limit,brain,extra),m) for m in pool], key=lambda x:x[0], reverse=True)
    min_s = 6 if extra else 3
    filtered = [m for s,m in scored if s>=min_s]
    if len(filtered)<take: filtered = [m for s,m in scored if s>=1]
    if len(filtered)<take: filtered = [m for s,m in scored]
    return filtered[offset:offset+take]
 
def why_text(m: Movie, mood: str) -> str:
    mood_hu = {"porgos":"pörgős","nyugis":"nyugis","sotet":"sötét","felemelo":"felemelő","vicces":"vicces","romantic":"romantikus"}.get(mood,"")
    genre_hu = {"action":"akció","thriller":"thriller","comedy":"vígjáték","drama":"dráma","romance":"romantikus","horror":"horror","crime":"krimi","animation":"animáció","adventure":"kaland","mystery":"rejtély","sport":"sport","biography":"életrajz","family":"családi","sci-fi":"sci-fi","fantasy":"fantasy","war":"háborús","history":"történelmi"}
    genres = [genre_hu.get(g,g) for g in m.genres[:2]]
    tags   = [t for t in m.tags if len(t)>3][:2]
    parts  = []
    if genres and mood_hu:
        parts.append(f"Ha {mood_hu} hangulatot keresel, ez a {', '.join(genres)} film tökéletes lehet.")
    if tags:
        parts.append(f"Témái: {', '.join(tags)}.")
    if m.avg_rating>=4.0 and m.rating_count>=30:
        parts.append(f"★{m.avg_rating:.1f}/5 ({m.rating_count} szavazat).")
    return " ".join(parts) if parts else f"{', '.join(genres)} — ★{m.avg_rating:.1f}/5"
 
# --- Session profil ---
def empty_profile():
    return {"mood":None,"time":None,"brain":None,"extra":"","era":"all","era_asked":False,"emotion_asked":False,"ready":False,"history":[]}
 
def get_p():
    p = session.get("profile")
    if not p: p=empty_profile(); session["profile"]=p
    return p
 
def missing(p):
    return [f for f in ("mood","time","brain") if not p.get(f)]
 
def next_q(p) -> Tuple[str, List[str]]:
    m = missing(p)
    if "mood" in m:
        return ("Milyen hangulatban vagy ma este?",
                ["Pörgős (akció, feszültség)","Nyugis (laza, chill)","Sötét (thriller, horror)","Felemelő (inspiráló, dráma)","Vicces (komédia, humor)","Romantikus (szerelem, érzelem)"])
    if "time" in m:
        return ("Mennyi időd van ma filmre?",
                ["90 perc (rövid)","120 perc (közepes)","150 perc (hosszabb)","180 perc (epikus)"])
    if "brain" in m:
        return ("Mennyire legyen elgondolkodtató?",
                ["Könnyű (kikapcsolódás)","Közepes (kis csavar)","Elgondolkodtató (mély, összetett)"])
    if not p.get("era_asked"):
        p["era_asked"]=True
        return ("Melyik korszakból szeretnél filmet nézni?",
                ["Friss (2015-2025)","Modern (2000-2015)","Klasszikus (1980-2000)","Régi (1980 előtt)","Mindegy"])
    if not p.get("emotion_asked"):
        p["emotion_asked"]=True
        mood=p.get("mood","")
        qs = {
            "sotet":    ("Mennyire legyen durva és intenzív?",["Enyhén sötét, nem ijeszt","Közepes — feszült de nem horror","Igazán durva és nyomasztó","Mindegy"]),
            "porgos":   ("Mennyire legyen brutális az akció?",["Tiszta akció, minimális vér","Közepes — van némi brutalitás","Brutális és nyers","Mindegy"]),
            "vicces":   ("Milyen humor illik most hozzád?",["Ártatlan, mindenki nevet","Fekete humor, csípős poénok","Abszurd és őrült","Mindegy"]),
            "felemelo": ("Mennyire legyen érzelmes?",["Kicsit megható, de nem sírok","Erősen érzelmes, könnyeket csal","Inkább inspiráló mint érzelmes","Mindegy"]),
            "nyugis":   ("Milyen hangulatú legyen?",["Vidám és szórakoztató","Szívmelengető és meghitt","Kalandos de könnyed","Mindegy"]),
            "romantic": ("Milyen románcot szeretnél?",["Édes és boldog szerelem","Drámai és tragikus románc","Bonyolult kapcsolat sok fordulattal","Mindegy"]),
        }
        return qs.get(mood,("Mennyire legyen intenzív?",["Enyhe és könnyű","Közepes intenzitású","Nagyon intenzív és erős","Mindegy"]))
    # MIND AZ 5 MEGVAN → FILMEK
    p["ready"]=True
    return ("Tökéletes, minden megvan. Jönnek a filmek!",[])
 
# --- Flask ---
app = Flask(__name__)
app.secret_key = FLASK_SECRET
 
# --- Google OAuth ---
@app.get("/auth/google")
def auth_google():
    if not G_CLIENT_ID: return "OAuth nincs beállítva",500
    state=secrets.token_hex(16); session["oauth_state"]=state
    params={"client_id":G_CLIENT_ID,"redirect_uri":G_REDIRECT,"response_type":"code","scope":"openid email profile","state":state}
    return redirect("https://accounts.google.com/o/oauth2/v2/auth?"+_uparse.urlencode(params))
 
@app.get("/auth/google/callback")
def auth_callback():
    if request.args.get("state","")!=session.get("oauth_state",""): return redirect("/?error=state")
    code=request.args.get("code","")
    if not code: return redirect("/?error=no_code")
    try:
        data=_uparse.urlencode({"code":code,"client_id":G_CLIENT_ID,"client_secret":G_SECRET,"redirect_uri":G_REDIRECT,"grant_type":"authorization_code"}).encode()
        req=_ureq.Request("https://oauth2.googleapis.com/token",data=data,headers={"Content-Type":"application/x-www-form-urlencoded"})
        with _ureq.urlopen(req,timeout=10) as r: tokens=json.loads(r.read())
        access=tokens.get("access_token")
        req2=_ureq.Request("https://www.googleapis.com/oauth2/v3/userinfo",headers={"Authorization":"Bearer "+access})
        with _ureq.urlopen(req2,timeout=10) as r: info=json.loads(r.read())
        session["user"]={"name":info.get("name",""),"email":info.get("email",""),"picture":info.get("picture","")}
    except Exception as e:
        log.warning("Google auth failed: %s",e)
    return redirect("/")
 
@app.get("/auth/logout")
def auth_logout():
    session.pop("user",None); return redirect("/")
 
@app.get("/api/me")
def api_me():
    u=session.get("user"); return jsonify({"user":u,"logged_in":bool(u)})
 
# --- API: poster proxy ---
@app.get("/api/poster")
def api_poster():
    url=request.args.get("url","")
    if not url.startswith("https://image.tmdb.org/"): return "Invalid",400
    try:
        req=_ureq.Request(url,headers={"User-Agent":"Mozilla/5.0","Referer":"https://www.themoviedb.org/"})
        with _ureq.urlopen(req,timeout=15) as r:
            return Response(r.read(),content_type=r.headers.get("Content-Type","image/jpeg"),headers={"Cache-Control":"public,max-age=86400"})
    except: return "Not found",404
 
# --- API: trailer ---
_trailer_cache={}
@app.get("/api/trailer")
def api_trailer():
    title=request.args.get("title",""); year=request.args.get("year",""); tmdb_id=request.args.get("tmdb_id","")
    key=f"{title}_{year}_{tmdb_id}"
    if key in _trailer_cache: return jsonify({"url":_trailer_cache[key]})
    if TMDB_KEY and tmdb_id:
        try:
            req=_ureq.Request(f"https://api.themoviedb.org/3/movie/{tmdb_id}/videos?api_key={TMDB_KEY}&language=en-US",headers={"User-Agent":"Mozilla/5.0"})
            with _ureq.urlopen(req,timeout=8) as r: data=json.loads(r.read())
            for v in data.get("results",[]):
                if v.get("site")=="YouTube" and v.get("type")=="Trailer":
                    url="https://www.youtube.com/watch?v="+v["key"]; _trailer_cache[key]=url; return jsonify({"url":url})
        except Exception as e: log.debug("Trailer lookup failed: %s",e)
    url="https://www.youtube.com/results?search_query="+_uparse.quote(f"{title} {year} official trailer")
    _trailer_cache[key]=url; return jsonify({"url":url})
 
# --- API: JustWatch ---
@app.get("/api/justwatch")
def api_justwatch():
    title=request.args.get("title","")
    return jsonify({"url":"https://www.justwatch.com/hu/kereses?q="+_uparse.quote(title)})
 
# --- API: debug ---
@app.get("/api/debug")
def api_debug():
    return jsonify({"movies_loaded":len(MOVIES),"tmdb":bool(TMDB_KEY),"google":bool(G_CLIENT_ID)})
 
# --- API: recs ---
@app.get("/api/recs")
def api_recs():
    mood  = request.args.get("mood","porgos")
    brain = request.args.get("brain","konnyu")
    time_limit = int(request.args.get("time","120"))
    q     = request.args.get("q","")
    offset= max(0,int(request.args.get("offset","0")))
    take  = min(24,max(1,int(request.args.get("take","6"))))
    era   = request.args.get("era","all")
    items = rank(mood,time_limit,brain,q,offset,take,era)
    return jsonify({
        "total":len(MOVIES),"offset":offset,"take":take,
        "items":[{"title":m.title,"year":m.year,"minutes":m.minutes,
                  "poster":m.poster,"trailer":m.trailer,"tmdb_id":m.tmdb_id,
                  "genres":m.genres,"tags":m.tags,"avg_rating":m.avg_rating,
                  "rating_count":m.rating_count,"why":why_text(m,mood)} for m in items]
    })
 
# --- API: chat ---
_GREETINGS = {"szia","hello","helló","hi","helo","csá","szevasz"}
_RESETS    = {"reset","új","uj","újra","restart"}
 
@app.post("/api/chat")
def api_chat():
    try:
        body = request.get_json(force=True, silent=True) or {}
        msg  = str(body.get("message","")).strip()
        p    = get_p()
        low  = msg.lower().strip()
 
        # Reset
        if low in _RESETS:
            session["profile"]=empty_profile(); p=get_p()
            q,chips=next_q(p); session["profile"]=p
            return jsonify({"assistant":"Oké, tiszta lap! "+q,"quick_replies":chips,"profile":p})
 
        # Greeting
        if low in _GREETINGS:
            session["profile"]=empty_profile(); p=get_p()
            q,chips=next_q(p); session["profile"]=p
            return jsonify({"assistant":"Szia! "+q,"quick_replies":chips,"profile":p})
 
        # Üres
        if not msg:
            q,chips=next_q(p); session["profile"]=p
            return jsonify({"assistant":q,"quick_replies":chips,"profile":p})
 
        def add_extra(w):
            p["extra"]=(p.get("extra","")+(" "+w if p.get("extra") else w))[:240]
 
        # Gomb akciók
        ACTIONS={
            # Hangulat
            "pörgős (akció, feszültség)": lambda: p.update({"mood":"porgos"}),
            "nyugis (laza, chill)":        lambda: p.update({"mood":"nyugis"}),
            "sötét (thriller, horror)":    lambda: p.update({"mood":"sotet"}),
            "felemelő (inspiráló, dráma)": lambda: p.update({"mood":"felemelo"}),
            "vicces (komédia, humor)":      lambda: p.update({"mood":"vicces"}),
            "romantikus (szerelem, érzelem)": lambda: p.update({"mood":"romantic"}),
            "pörgős": lambda: p.update({"mood":"porgos"}),
            "nyugis":  lambda: p.update({"mood":"nyugis"}),
            "sötét":   lambda: p.update({"mood":"sotet"}),
            "felemelő":lambda: p.update({"mood":"felemelo"}),
            "vicces":  lambda: p.update({"mood":"vicces"}),
            "romantikus": lambda: p.update({"mood":"romantic"}),
            # Idő
            "90 perc (rövid)":    lambda: p.update({"time":90}),
            "120 perc (közepes)": lambda: p.update({"time":120}),
            "150 perc (hosszabb)":lambda: p.update({"time":150}),
            "180 perc (epikus)":  lambda: p.update({"time":180}),
            "90 perc": lambda: p.update({"time":90}),
            "120 perc":lambda: p.update({"time":120}),
            "150 perc":lambda: p.update({"time":150}),
            "180 perc":lambda: p.update({"time":180}),
            # Brain
            "könnyű (kikapcsolódás)":          lambda: p.update({"brain":"konnyu"}),
            "közepes (kis csavar)":            lambda: p.update({"brain":"kozepes"}),
            "elgondolkodtató (mély, összetett)":lambda: p.update({"brain":"elgondolkodtato"}),
            "könnyű":         lambda: p.update({"brain":"konnyu"}),
            "közepes":        lambda: p.update({"brain":"kozepes"}),
            "elgondolkodtató":lambda: p.update({"brain":"elgondolkodtato"}),
            # Korszak
            "friss (2015-2025)":      lambda: p.update({"era":"recent","era_asked":True}),
            "modern (2000-2015)":     lambda: p.update({"era":"new",   "era_asked":True}),
            "klasszikus (1980-2000)": lambda: p.update({"era":"classic","era_asked":True}),
            "régi (1980 előtt)":      lambda: p.update({"era":"old",   "era_asked":True}),
            "mindegy":                lambda: p.update({"era":"all",   "era_asked":True}),
            # Ajánlj
            "ajánlj most": lambda: p.update({"ready":True}),
            "ajánlj":      lambda: p.update({"ready":True}),
        }
        # Extra kulcsszavak hozzáadása
        for k,v in EXTRA_KW.items():
            ACTIONS[k] = (lambda v=v: add_extra(v))
 
        if low in ACTIONS:
            ACTIONS[low]()
        else:
            # Szöveg értelmezés
            t=msg.lower()
            if re.search(r"(\d{2,3})\s*(perc|p)\b",t):
                m2=re.search(r"(\d{2,3})\s*(perc|p)\b",t)
                v=int(m2.group(1))
                if 60<=v<=240: p["time"]=v
            if any(w in t for w in ["pörg","akció","harc"]): p["mood"]="porgos"
            elif any(w in t for w in ["nyugi","chill","laza"]): p["mood"]="nyugis"
            elif any(w in t for w in ["sötét","thriller","krimi","horror"]): p["mood"]="sotet"
            elif any(w in t for w in ["felem","motiv","inspir"]): p["mood"]="felemelo"
            elif any(w in t for w in ["vicc","kom","humor"]): p["mood"]="vicces"
            elif any(w in t for w in ["roman","szerel"]): p["mood"]="romantic"
            if any(w in t for w in ["könny","egyszerű","laza film"]): p["brain"]="konnyu"
            elif any(w in t for w in ["elgondolk","csavar","bonyol"]): p["brain"]="elgondolkodtato"
            # Extra kulcsszó
            kw=" ".join([x for x in re.split(r"[^a-záéíóöőúüű0-9]+",t) if len(x)>=3 and x not in {"legyen","valami","film","nézek","kérek","szeretnék"}])
            if kw: add_extra(kw[:60])
 
        # Következő kérdés
        q_text, chips = next_q(p)
 
        hist=p.get("history",[]); hist.extend([{"role":"user","content":msg},{"role":"assistant","content":q_text}]); p["history"]=hist[-20:]
        session["profile"]=p; session.modified=True
 
        return jsonify({"assistant":q_text,"quick_replies":chips,"profile":p})
 
    except Exception as e:
        log.exception("Chat hiba: %s",e)
        return jsonify({"assistant":"Hiba történt, próbáld újra.","quick_replies":["Reset"],"profile":empty_profile()}),500
 
 
# --- HTML Frontend ---
_HTML = """<!doctype html>
<html lang="hu">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width,initial-scale=1,maximum-scale=1"/>
<title>SG Film Ajánló</title>
<link href="https://fonts.googleapis.com/css2?family=DM+Serif+Display:ital@0;1&family=DM+Sans:wght@300..700&display=swap" rel="stylesheet"/>
<style>
:root{--bg:#080a0d;--surface:#0e1218;--border:#1e2730;--border2:#2a3441;--text:#dde4ed;--muted:#7a8a9a;--faint:#3a4a5a;--gold:#c8a84b;--gold2:#e8c86c;--red:#c84b4b;--radius:16px;--font-serif:'DM Serif Display',Georgia,serif;--font-sans:'DM Sans',system-ui,sans-serif}
*{box-sizing:border-box;margin:0;padding:0}
html{-webkit-tap-highlight-color:transparent}
body{background:var(--bg);color:var(--text);font-family:var(--font-sans);font-size:14px;line-height:1.55;min-height:100vh;padding:16px 12px 60px;background-image:radial-gradient(ellipse 900px 500px at 10% 0%,rgba(200,168,75,.06),transparent 70%)}
.wrap{max-width:1160px;margin:0 auto}
/* Header */
.header{display:flex;align-items:center;justify-content:space-between;gap:12px;flex-wrap:wrap;margin-bottom:20px;padding-bottom:16px;border-bottom:1px solid var(--border)}
.brand{display:flex;align-items:center;gap:12px}
.logo{width:44px;height:44px;border-radius:12px;border:1px solid var(--border2);background:linear-gradient(145deg,#1a2230,#0a0d12);display:flex;align-items:center;justify-content:center;flex-shrink:0}
.logo span{font-family:var(--font-serif);font-size:16px;color:var(--gold);letter-spacing:1px}
.brand-text h1{font-family:var(--font-serif);font-size:24px;line-height:1.1}
.brand-text h1 em{font-style:italic;color:var(--gold)}
.brand-text p{color:var(--muted);font-size:12px;margin-top:2px}
.badges{display:flex;align-items:center;gap:10px;flex-wrap:wrap}
/* Buttons */
.btn{cursor:pointer;border:1px solid var(--border2);background:linear-gradient(145deg,#1c2840,#0c1020);color:var(--text);font-family:var(--font-sans);font-weight:600;font-size:13px;padding:9px 14px;border-radius:12px;transition:border-color .15s,background .15s,transform .1s;white-space:nowrap;-webkit-appearance:none}
.btn:hover{border-color:var(--gold);background:linear-gradient(145deg,#212e48,#0e1428)}
.btn:active{transform:scale(.97)}
/* Grid */
.grid{display:grid;grid-template-columns:1fr 1fr;gap:14px;align-items:start}
@media(max-width:768px){.grid{grid-template-columns:1fr}}
/* Card */
.card{border:1px solid var(--border);background:rgba(14,18,24,.8);border-radius:var(--radius);overflow:hidden;box-shadow:0 20px 60px rgba(0,0,0,.6)}
.card-head{padding:12px 16px;border-bottom:1px solid var(--border);display:flex;align-items:center;justify-content:space-between;gap:10px;background:linear-gradient(180deg,rgba(22,30,42,.9),rgba(14,18,24,.6))}
.card-title{font-family:var(--font-serif);font-size:15px;color:var(--gold)}
.card-body{padding:14px 16px}
/* Chat */
.chat-box{background:rgba(4,6,10,.5);border-radius:12px;padding:12px;height:300px;overflow-y:auto;scroll-behavior:smooth}
@media(max-width:768px){.chat-box{height:220px}}
.chat-box::-webkit-scrollbar{width:3px}
.chat-box::-webkit-scrollbar-thumb{background:var(--border2);border-radius:2px}
.msg{display:flex;gap:8px;margin:10px 0;animation:fadeUp .2s ease}
@keyframes fadeUp{from{opacity:0;transform:translateY(5px)}to{opacity:1;transform:translateY(0)}}
.av{width:26px;height:26px;border-radius:8px;background:rgba(200,168,75,.12);display:flex;align-items:center;justify-content:center;flex-shrink:0;font-size:9px;font-weight:700;color:var(--gold)}
.msg.me .av{background:rgba(42,52,65,.5);color:var(--muted)}
.bub{max-width:80%;padding:10px 14px;border-radius:14px;background:rgba(22,30,44,.9);line-height:1.6;font-size:13px;box-shadow:0 1px 4px rgba(0,0,0,.3)}
.msg.me .bub{margin-left:auto;background:rgba(200,168,75,.1);border:1px solid rgba(200,168,75,.2)}
.dots span{display:inline-block;width:4px;height:4px;border-radius:50%;background:var(--gold);margin:0 2px;animation:blink 1.2s infinite}
.dots span:nth-child(2){animation-delay:.2s}
.dots span:nth-child(3){animation-delay:.4s}
@keyframes blink{0%,80%,100%{opacity:.2;transform:scale(.8)}40%{opacity:1;transform:scale(1)}}
/* Chips */
.chips{display:flex;gap:6px;flex-wrap:wrap;margin-top:10px}
.chip{cursor:pointer;border:1px solid rgba(42,52,65,.5);background:rgba(14,18,24,.5);color:var(--muted);padding:6px 12px;border-radius:999px;font-size:12px;font-weight:500;transition:all .15s;-webkit-appearance:none}
.chip:hover{border-color:var(--gold);color:var(--gold2);background:rgba(200,168,75,.08)}
/* Input */
.inp-row{display:flex;gap:8px;margin-top:12px}
.chat-inp{flex:1;min-width:0;padding:11px 16px;border-radius:14px;border:1px solid rgba(42,52,65,.6);background:rgba(8,10,14,.7);color:var(--text);font-family:var(--font-sans);font-size:14px;outline:none;transition:border-color .2s;-webkit-appearance:none}
.chat-inp:focus{border-color:rgba(200,168,75,.5)}
.chat-inp::placeholder{color:rgba(90,110,130,.6)}
/* Mood grid */
.mood-grid{display:grid;grid-template-columns:repeat(3,1fr);gap:8px;margin-top:14px}
.mood-btn{cursor:pointer;border:1px solid rgba(30,39,48,.6);background:rgba(10,14,20,.4);border-radius:14px;padding:10px 6px;text-align:center;transition:all .2s;-webkit-appearance:none}
.mood-btn:hover{border-color:rgba(200,168,75,.4);background:rgba(200,168,75,.06);transform:translateY(-1px)}
.mood-btn.active{border-color:var(--gold);background:rgba(200,168,75,.1)}
.mood-btn .emoji{font-size:20px;display:block;margin-bottom:4px}
.mood-btn .label{font-size:11px;color:var(--muted);font-weight:600}
.mood-btn.active .label{color:var(--gold2)}
/* Surprise */
.surprise-btn{width:100%;cursor:pointer;border:1px solid var(--gold);background:linear-gradient(145deg,rgba(200,168,75,.1),rgba(200,168,75,.04));color:var(--gold2);font-family:var(--font-sans);font-weight:700;font-size:14px;padding:13px;border-radius:14px;transition:all .2s;margin-top:10px;-webkit-appearance:none}
.surprise-btn:hover{background:linear-gradient(145deg,rgba(200,168,75,.2),rgba(200,168,75,.08))}
/* Poster strip */
#poster-strip{display:grid;grid-template-columns:repeat(auto-fill,minmax(140px,1fr));gap:10px;padding:12px;max-height:640px;overflow-y:auto}
@media(max-width:768px){#poster-strip{grid-template-columns:repeat(auto-fill,minmax(120px,1fr));padding:8px;gap:8px}}
#poster-strip::-webkit-scrollbar{width:3px}
#poster-strip::-webkit-scrollbar-thumb{background:var(--border2);border-radius:2px}
/* Poster card */
.pc{border:1px solid var(--border);border-radius:12px;overflow:hidden;background:rgba(10,14,20,.85);transition:transform .18s,border-color .18s,box-shadow .18s;animation:fadeUp .25s ease}
.pc:hover{transform:translateY(-3px);border-color:var(--gold);box-shadow:0 6px 24px rgba(200,168,75,.1)}
.pimg{width:100%;aspect-ratio:2/3;background:#0a0c10;display:flex;align-items:center;justify-content:center;overflow:hidden;position:relative}
.pimg img{width:100%;height:100%;object-fit:cover;display:block;transition:transform .3s}
.pc:hover .pimg img{transform:scale(1.04)}
.pfallback{font-family:var(--font-serif);font-size:10px;color:rgba(200,168,75,.6);text-align:center;padding:8px;line-height:1.3}
.wl-btn{position:absolute;top:6px;left:6px;width:26px;height:26px;border-radius:7px;border:1px solid var(--border2);background:rgba(8,10,14,.8);color:var(--muted);font-size:13px;display:flex;align-items:center;justify-content:center;cursor:pointer;-webkit-appearance:none}
.wl-btn:hover{border-color:var(--gold);color:var(--gold)}
.wl-btn.saved{border-color:var(--gold);color:var(--gold);background:rgba(200,168,75,.15)}
.cert{position:absolute;top:6px;right:6px;padding:2px 5px;border-radius:4px;background:rgba(200,168,75,.15);border:1px solid rgba(200,168,75,.3);color:var(--gold);font-size:9px;font-weight:700}
.tmdb-b{position:absolute;bottom:5px;left:5px;padding:2px 5px;border-radius:4px;background:rgba(1,180,228,.12);border:1px solid rgba(1,180,228,.25);color:#01b4e4;font-size:9px;font-weight:600;text-decoration:none;opacity:.8}
.pinfo{padding:8px 8px 10px}
.ptitle{font-weight:700;font-size:12px;line-height:1.25;color:var(--text);margin-bottom:3px}
.pmeta{color:var(--muted);font-size:10px;margin-bottom:7px}
.pbtns{display:flex;gap:4px}
.pbtns button{flex:1;text-align:center;padding:5px 3px;border-radius:8px;border:1px solid var(--border);background:rgba(14,18,24,.6);color:var(--text);font-size:9.5px;font-weight:600;cursor:pointer;font-family:var(--font-sans);-webkit-appearance:none;transition:border-color .15s}
.pbtns button:hover{border-color:var(--gold);background:rgba(200,168,75,.08)}
.why-btn{color:var(--gold) !important}
.jw-btn{color:#00d8ff !important;border-color:rgba(0,216,255,.2) !important;background:rgba(0,216,255,.04) !important}
.empty{grid-column:1/-1;padding:28px;text-align:center;color:var(--muted);font-size:13px}
/* Watchlist */
.wl-panel{display:none;position:fixed;inset:0;background:rgba(4,6,10,.9);backdrop-filter:blur(8px);z-index:300;align-items:flex-end;justify-content:center}
.wl-panel.open{display:flex}
.wl-sheet{background:#131820;border:1px solid var(--border2);border-radius:20px 20px 0 0;width:100%;max-width:600px;max-height:80vh;overflow-y:auto;padding:20px}
.wl-sheet h3{font-family:var(--font-serif);color:var(--gold);font-size:18px;margin-bottom:16px}
.wl-item{display:flex;align-items:center;gap:12px;padding:10px 0;border-bottom:1px solid var(--border)}
.wl-item:last-child{border-bottom:none}
.wl-rm{cursor:pointer;width:26px;height:26px;border-radius:7px;border:1px solid var(--border);background:transparent;color:var(--muted);font-size:14px;display:flex;align-items:center;justify-content:center;-webkit-appearance:none}
.wl-rm:hover{border-color:var(--red);color:var(--red)}
/* Modal */
.modal-bg{display:none;position:fixed;inset:0;background:rgba(4,6,10,.85);backdrop-filter:blur(6px);z-index:200;align-items:center;justify-content:center;padding:16px}
.modal-bg.open{display:flex}
.modal{background:#131820;border:1px solid var(--border2);border-radius:18px;max-width:460px;width:100%;overflow:hidden;box-shadow:0 32px 80px rgba(0,0,0,.7)}
.modal-head{padding:16px 18px 12px;border-bottom:1px solid var(--border);display:flex;align-items:center;justify-content:space-between}
.modal-ttl{font-family:var(--font-serif);font-size:17px;color:var(--gold)}
.modal-x{cursor:pointer;width:28px;height:28px;border-radius:8px;border:1px solid var(--border);background:transparent;color:var(--muted);font-size:16px;display:flex;align-items:center;justify-content:center;-webkit-appearance:none}
.modal-x:hover{border-color:var(--red);color:var(--red)}
.modal-body{padding:18px;color:var(--text);font-size:13.5px;line-height:1.7}
/* Footer */
.tmdb-footer{margin-top:24px;padding:14px 16px;border-top:1px solid var(--border);display:flex;align-items:center;gap:12px;flex-wrap:wrap}
.tmdb-footer img{height:18px;opacity:.7}
.tmdb-footer p{color:var(--muted);font-size:11px;line-height:1.4}
.tmdb-footer a{color:#01b4e4;text-decoration:none}
</style>
</head>
<body>
<div class="wrap">
<div class="header">
  <div class="brand">
    <div class="logo"><span>SG</span></div>
    <div class="brand-text">
      <h1>SG <em>Film Ajánló</em></h1>
      <p>AI chat + intelligens ajánlások</p>
    </div>
  </div>
  <div class="badges">
    <span id="pill" style="font-size:11px;color:var(--gold);font-weight:600"></span>
    <span style="font-size:11px;color:var(--muted)">🚫 Nem stream oldal</span>
    <button class="btn" id="btn-wl" style="padding:7px 12px;font-size:12px">🔖 Lista</button>
    <div id="user-area"></div>
  </div>
</div>
<div class="grid">
  <div class="card">
    <div class="card-head" style="border-bottom:1px solid rgba(30,39,48,.4)">
      <div class="card-title" style="font-size:14px;opacity:.9">Asszisztens</div>
      <button class="btn" id="btn-reset" style="padding:5px 10px;font-size:11px;opacity:.5;border:none;background:transparent;color:var(--muted)">↺ Reset</button>
    </div>
    <div class="card-body" style="padding:12px 14px">
      <div class="chat-box" id="chat-box"></div>
      <div class="chips" id="chips"></div>
      <div class="inp-row" style="margin-top:10px">
        <input class="chat-inp" id="inp" placeholder="Írj valamit..." autocomplete="off"/>
        <button class="btn" id="btn-send" style="padding:9px 16px">➤</button>
      </div>
      <div class="mood-grid">
        <button class="mood-btn" data-mood="porgos"><span class="emoji">⚡</span><span class="label">Pörgős</span></button>
        <button class="mood-btn" data-mood="nyugis"><span class="emoji">😌</span><span class="label">Nyugis</span></button>
        <button class="mood-btn" data-mood="sotet"><span class="emoji">🌑</span><span class="label">Sötét</span></button>
        <button class="mood-btn" data-mood="felemelo"><span class="emoji">🚀</span><span class="label">Felemelő</span></button>
        <button class="mood-btn" data-mood="vicces"><span class="emoji">😂</span><span class="label">Vicces</span></button>
        <button class="mood-btn" data-mood="romantic"><span class="emoji">💕</span><span class="label">Romantikus</span></button>
      </div>
      <button class="surprise-btn" id="btn-surprise">🎲 Lepj meg — random film</button>
    </div>
  </div>
  <div class="card">
    <div class="card-head">
      <div class="card-title">🎥 Ajánlott filmek</div>
      <button class="btn" id="btn-more" style="padding:7px 12px;font-size:12px">+ Több</button>
    </div>
    <div class="card-body" style="padding:0">
      <div id="poster-strip"></div>
      <div style="padding:10px 14px 14px;border-top:1px solid var(--border);font-size:10px;color:var(--faint);line-height:1.6">
        © A filmek poszterein és adatain fennálló szerzői jogok a jogtulajdonosokat illetik. A trailer linkek YouTube-ra mutatnak. Ez az oldal kizárólag ajánló célokat szolgál.
      </div>
    </div>
  </div>
</div>
<div class="tmdb-footer">
  <img src="https://www.themoviedb.org/assets/2/v4/logos/v2/blue_short-8e7b30f73a4020692ccca9c88bafe5dcb6f8a62a4c6bc55cd9ba82bb2cd95f6c.svg" alt="TMDB"/>
  <p>This product uses the TMDB API but is not endorsed or certified by TMDB.<br>Poszter képek forrása: <a href="https://www.themoviedb.org/" target="_blank" rel="noopener">The Movie Database (TMDB)</a></p>
</div>
</div>
 
<div class="wl-panel" id="wl-panel">
  <div class="wl-sheet">
    <div style="display:flex;align-items:center;justify-content:space-between;margin-bottom:16px">
      <h3>🔖 Mentett filmek</h3>
      <button class="modal-x" id="wl-close">✕</button>
    </div>
    <div id="wl-list"></div>
  </div>
</div>
<div class="modal-bg" id="modal">
  <div class="modal">
    <div class="modal-head">
      <div class="modal-ttl" id="modal-ttl">Miért ajánljuk?</div>
      <button class="modal-x" id="modal-x">✕</button>
    </div>
    <div class="modal-body" id="modal-body"></div>
  </div>
</div>
 
<script>
(function(){
'use strict';
const $ = id => document.getElementById(id);
const chatBox=$('chat-box'), chips=$('chips'), inp=$('inp');
const posterStrip=$('poster-strip'), pill=$('pill');
const modal=$('modal'), modalTtl=$('modal-ttl'), modalBody=$('modal-body');
let state={mood:'porgos',brain:'konnyu',time:120,q:'',offset:0,take:6,era:'all',ready:false};
let watchlist=JSON.parse(localStorage.getItem('sg_wl')||'[]');
 
function esc(s){return String(s||'').replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;').replace(/"/g,'&quot;').replace(/'/g,'&#039;')}
 
// Modal
function openModal(t,b){modalTtl.textContent=t;modalBody.textContent=b||'—';modal.classList.add('open')}
$('modal-x').onclick=()=>modal.classList.remove('open');
modal.onclick=e=>{if(e.target===modal)modal.classList.remove('open')};
document.addEventListener('keydown',e=>{if(e.key==='Escape')modal.classList.remove('open')});
 
// Watchlist
function saveWL(){localStorage.setItem('sg_wl',JSON.stringify(watchlist))}
function inWL(t,y){return watchlist.some(m=>m.title===t&&m.year===y)}
function toggleWL(movie){
  const i=watchlist.findIndex(m=>m.title===movie.title&&m.year===movie.year);
  if(i>=0)watchlist.splice(i,1); else watchlist.push(movie);
  saveWL(); renderWL();
  document.querySelectorAll('.wl-btn[data-title="'+esc(movie.title)+'"]').forEach(b=>{
    b.classList.toggle('saved',inWL(movie.title,movie.year));
    b.textContent=inWL(movie.title,movie.year)?'★':'☆';
  });
}
function renderWL(){
  const list=$('wl-list');
  if(!watchlist.length){list.innerHTML='<div style="color:var(--muted);text-align:center;padding:20px">Még nincs mentett film.</div>';return}
  list.innerHTML=watchlist.map(m=>`<div class="wl-item"><div style="flex:1"><div style="font-size:13px;font-weight:600">${esc(m.title)}</div><div style="color:var(--muted);font-size:11px">${esc(String(m.year))}</div></div><button class="wl-rm" data-t="${esc(m.title)}" data-y="${esc(String(m.year))}">✕</button></div>`).join('');
  list.querySelectorAll('.wl-rm').forEach(b=>b.onclick=()=>{watchlist=watchlist.filter(m=>!(m.title===b.dataset.t&&String(m.year)===b.dataset.y));saveWL();renderWL();document.querySelectorAll('.wl-btn[data-title="'+b.dataset.t+'"]').forEach(x=>{x.classList.remove('saved');x.textContent='☆'})});
}
$('btn-wl').onclick=()=>{renderWL();$('wl-panel').classList.add('open')};
$('wl-close').onclick=()=>$('wl-panel').classList.remove('open');
$('wl-panel').onclick=e=>{if(e.target===$('wl-panel'))$('wl-panel').classList.remove('open')};
 
// Chat
function addMsg(who,text){
  const row=document.createElement('div');row.className='msg '+(who==='Te'?'me':'ai');
  const av=document.createElement('div');av.className='av';av.textContent=who==='Te'?'Te':'SG';
  const bub=document.createElement('div');bub.className='bub';bub.textContent=text||'';
  row.append(av,bub);chatBox.appendChild(row);chatBox.scrollTop=chatBox.scrollHeight;
  return row;
}
function addTyping(){
  const row=document.createElement('div');row.className='msg ai';
  row.innerHTML='<div class="av">SG</div><div class="bub"><div class="dots"><span></span><span></span><span></span></div></div>';
  chatBox.appendChild(row);chatBox.scrollTop=chatBox.scrollHeight;return row;
}
 
// Chips
function setChips(arr){
  chips.innerHTML='';
  (arr||[]).forEach(t=>{
    const c=document.createElement('div');c.className='chip';c.textContent=t;
    c.onclick=()=>{inp.value=t;send()};
    chips.appendChild(c);
  });
}
 
// Poster HTML
function posterHTML(m){
  const saved=inWL(m.title,m.year);
  const src=m.poster?'/api/poster?url='+encodeURIComponent(m.poster):'';
  const rating=m.avg_rating>0?'<span style="color:var(--gold);margin-left:3px">★'+m.avg_rating.toFixed(1)+'</span>':'';
  return '<div class="pc">'
    +'<div class="pimg">'
    +(src?'<img src="'+src+'" alt="'+esc(m.title)+'" loading="eager" onerror="this.style.display=\'none\'">':'')
    +'<div class="pfallback"'+( src?' style="display:none"':'')+'">'+esc(m.title)+'</div>'
    +(m.certification?'<div class="cert">'+esc(m.certification)+'</div>':'')
    +(m.poster?'<a class="tmdb-b" href="https://www.themoviedb.org/" target="_blank" rel="noopener">TMDB</a>':'')
    +'<button class="wl-btn'+(saved?' saved':'')+'" data-title="'+esc(m.title)+'" data-year="'+esc(String(m.year))+'">'+(saved?'★':'☆')+'</button>'
    +'</div>'
    +'<div class="pinfo">'
    +'<div class="ptitle">'+esc(m.title)+'</div>'
    +'<div class="pmeta">'+esc(String(m.year))+' · '+esc(String(m.minutes))+'p'+rating+'</div>'
    +'<div class="pbtns">'
    +'<button class="trailer-btn" data-title="'+esc(m.title)+'" data-year="'+esc(String(m.year))+'" data-tmdb="'+esc(m.tmdb_id||'')+'">▶ Trailer</button>'
    +'<button class="why-btn" data-title="'+esc(m.title)+'" data-why="'+esc(m.why||'')+'">? Miért</button>'
    +'<button class="jw-btn" data-title="'+esc(m.title)+'" data-year="'+esc(String(m.year))+'">📺 Hol?</button>'
    +'</div></div></div>';
}
 
function bindCards(scope){
  scope.querySelectorAll('.why-btn').forEach(b=>b.onclick=()=>openModal(b.dataset.title,b.dataset.why||'—'));
  scope.querySelectorAll('.wl-btn').forEach(b=>b.onclick=()=>toggleWL({title:b.dataset.title,year:parseInt(b.dataset.year)||0}));
  scope.querySelectorAll('.trailer-btn').forEach(b=>b.addEventListener('click',async()=>{
    const orig=b.textContent;b.textContent='⏳';b.disabled=true;
    try{
      const r=await fetch('/api/trailer?title='+encodeURIComponent(b.dataset.title)+'&year='+encodeURIComponent(b.dataset.year)+'&tmdb_id='+encodeURIComponent(b.dataset.tmdb||''));
      const d=await r.json();
      window.open(d.url||('https://www.youtube.com/results?search_query='+encodeURIComponent(b.dataset.title+' trailer')),'_blank','noopener');
    }catch(e){window.open('https://www.youtube.com/results?search_query='+encodeURIComponent(b.dataset.title+' trailer'),'_blank','noopener')}
    finally{b.textContent=orig;b.disabled=false}
  }));
  scope.querySelectorAll('.jw-btn').forEach(b=>b.onclick=async()=>{
    const r=await fetch('/api/justwatch?title='+encodeURIComponent(b.dataset.title));
    const d=await r.json();window.open(d.url,'_blank','noopener');
  });
}
 
async function loadMore(){
  const url='/api/recs?mood='+encodeURIComponent(state.mood)+'&brain='+encodeURIComponent(state.brain)+'&time='+encodeURIComponent(state.time)+'&q='+encodeURIComponent(state.q||'')+'&era='+encodeURIComponent(state.era||'all')+'&offset='+state.offset+'&take='+state.take;
  try{
    const r=await fetch(url);const d=await r.json();
    if(pill)pill.textContent=(d.total||0)+' film';
    const items=d.items||[];
    if(!items.length){const el=document.createElement('div');el.className='empty';el.textContent='Nincs több találat.';posterStrip.appendChild(el);return}
    const chunk=document.createElement('div');chunk.style.display='contents';
    chunk.innerHTML=items.map(posterHTML).join('');posterStrip.appendChild(chunk);
    bindCards(chunk);state.offset+=items.length;
  }catch(e){console.error('loadMore:',e)}
}
 
// Mood buttons
document.querySelectorAll('.mood-btn').forEach(btn=>{
  btn.onclick=async()=>{
    document.querySelectorAll('.mood-btn').forEach(b=>b.classList.remove('active'));
    btn.classList.add('active');
    inp.value=btn.querySelector('.label').textContent;
    await send();
  }
});
 
// Surprise
$('btn-surprise').addEventListener('click',async()=>{
  const btn=$('btn-surprise');btn.textContent='⏳';btn.disabled=true;
  try{
    const moods=['porgos','nyugis','sotet','felemelo','vicces'];
    state.mood=moods[Math.floor(Math.random()*moods.length)];
    state.brain=['konnyu','kozepes','elgondolkodtato'][Math.floor(Math.random()*3)];
    state.time=[90,120,150,180][Math.floor(Math.random()*4)];
    state.ready=true;state.offset=0;posterStrip.innerHTML='';
    await loadMore();addMsg('SG','🎲 Random filmek — jó szórakozást!');
  }catch(e){console.error(e)}
  btn.textContent='🎲 Lepj meg — random film';btn.disabled=false;
});
 
// Send
async function send(){
  const msg=(inp.value||'').trim();if(!msg)return;
  addMsg('Te',msg);inp.value='';
  const typing=addTyping();
  try{
    const res=await fetch('/api/chat',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({message:msg})});
    const data=await res.json();typing.remove();
    addMsg('SG',data.assistant||'…');setChips(data.quick_replies||[]);
    const p=data.profile||{};
    if(p.mood) state.mood=p.mood;
    if(p.brain)state.brain=p.brain;
    if(p.time) state.time=Number(p.time)||state.time;
    if(p.era)  state.era=p.era;
    state.q=p.extra||'';
    // FILMDOBÁS: ha a backend ready=true-t küld
    if(p.ready && !state.ready){
      state.ready=true;state.offset=0;posterStrip.innerHTML='';
      await loadMore();
    }
  }catch(e){typing.remove();console.error(e)}
}
 
$('btn-send').onclick=send;
$('btn-reset').onclick=()=>{inp.value='Reset';send()};
$('btn-more').onclick=loadMore;
inp.addEventListener('keydown',e=>{if(e.key==='Enter')send()});
 
// User area
async function loadUser(){
  try{
    const r=await fetch('/api/me');const d=await r.json();
    const area=$('user-area');if(!area)return;
    if(d.logged_in&&d.user){
      const name=(d.user.name||'').split(' ')[0];
      const div=document.createElement('div');div.style.cssText='display:flex;align-items:center;gap:8px';
      if(d.user.picture){const img=document.createElement('img');img.src=d.user.picture;img.style.cssText='width:28px;height:28px;border-radius:50%;border:1px solid #2a3441';div.appendChild(img)}
      const span=document.createElement('span');span.style.cssText='font-size:12px;color:#dde4ed';span.textContent=name;div.appendChild(span);
      const a=document.createElement('a');a.href='/auth/logout';a.style.cssText='font-size:11px;color:#7a8a9a;text-decoration:none;border:1px solid #1e2730;padding:4px 8px;border-radius:8px';a.textContent='Kilépés';div.appendChild(a);
      area.innerHTML='';area.appendChild(div);
    }else{
      const a=document.createElement('a');a.href='/auth/google';a.style.cssText='display:flex;align-items:center;gap:6px;padding:7px 12px;border-radius:12px;border:1px solid #2a3441;background:linear-gradient(145deg,#1c2840,#0c1020);color:#dde4ed;text-decoration:none;font-size:12px;font-weight:600';a.textContent='Google bejelentkezés';
      area.innerHTML='';area.appendChild(a);
    }
  }catch(e){console.error(e)}
}
 
// Init
loadUser();
addMsg('SG','Szia! Válassz hangulatot a gombokkal, vagy írj mit szeretnél nézni.');
fetch('/api/debug').then(r=>r.json()).then(d=>{if(pill)pill.textContent=(d.movies_loaded||0)+' film'}).catch(()=>{});
})();
</script>
</body>
</html>"""
 
@app.get("/")
def home(): return _HTML
 
if __name__=="__main__":
    port=int(os.environ.get("PORT",5000))
    log.info("Indítás porton: %d",port)
    app.run(host="0.0.0.0",port=port,debug=False)
