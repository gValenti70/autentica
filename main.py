import os
import json
import time
from typing import Optional, Tuple, Any, Dict, List

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from difflib import SequenceMatcher
from openai import AzureOpenAI
import logging
from PIL import Image
import pillow_avif  # noqa: F401
import base64
import io
import re
import socket

from pymongo import MongoClient, ReturnDocument
from pymongo.errors import DuplicateKeyError

# ======================================================
# LOGGING
# ======================================================
logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

# ======================================================
# CONFIG OPENAI
# ======================================================
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT", "https://autenticagpt.openai.azure.com/")
AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY", "")
AZURE_OPENAI_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION", "2024-12-01-preview")
DEPLOYMENT_GPT = os.getenv("DEPLOYMENT_GPT", "gpt-5.1-chat")

client_gpt = AzureOpenAI(
    azure_endpoint=AZURE_OPENAI_ENDPOINT,
    api_key=AZURE_OPENAI_API_KEY,
    api_version=AZURE_OPENAI_API_VERSION,
)

MAX_FOTO = int(os.getenv("MAX_FOTO", "7"))

# ======================================================
# MONGO CONFIG
# ======================================================
MONGO_URI = os.getenv("MONGO_URI", "")
MONGO_DB_NAME = os.getenv("MONGO_DB_NAME", "autentica")

# Collezioni (nomenclatura chiara)
COL_ANALISI = "analisi"
COL_STEPS = "analisi_steps"
COL_PROMPTS = "prompt_versions"
COL_USERS = "users"
COL_COUNTERS = "counters"

_mongo_client: Optional[MongoClient] = None

def get_mongo_client() -> MongoClient:
    global _mongo_client
    if _mongo_client is None:
        if not MONGO_URI:
            raise RuntimeError("MONGO_URI missing (set it in App Service env vars)")
        # DocumentDB Mongo compat: TLS obbligatorio, SRV ok
        _mongo_client = MongoClient(
            MONGO_URI,
            serverSelectionTimeoutMS=8000,
            connectTimeoutMS=8000,
            socketTimeoutMS=20000,
            retryWrites=False
        )
    return _mongo_client

def get_db():
    client = get_mongo_client()
    return client[MONGO_DB_NAME]

def ensure_indexes():
    db = get_db()

    # Unique su (id_analisi, step)
    db[COL_STEPS].create_index([("id_analisi", 1), ("step", 1)], unique=True)

    # Prompt: velocizziamo query (name + user + active)
    db[COL_PROMPTS].create_index([("prompt_name", 1), ("user_id", 1), ("is_active", 1)])

    # Users
    db[COL_USERS].create_index([("_id", 1)], unique=True)

    # Counter
    db[COL_COUNTERS].create_index([("_id", 1)], unique=True)

def get_next_analisi_id() -> int:
    """
    Genera un id_analisi INT atomico stile MySQL autoincrement.
    """
    db = get_db()
    doc = db[COL_COUNTERS].find_one_and_update(
        {"_id": "analisi"},
        {"$inc": {"seq": 1}},
        upsert=True,
        return_document=ReturnDocument.AFTER
    )
    # Se era appena creato e non aveva seq, sistemiamo
    seq = doc.get("seq", 1)
    if not isinstance(seq, int):
        try:
            seq = int(seq)
        except:
            seq = 1
    return seq

# ======================================================
# FASTAPI
# ======================================================
app = FastAPI(title="Autentica V2 Backend", version="6.0-mongo")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)

@app.on_event("startup")
def startup():
    try:
        # ping + indici
        client = get_mongo_client()
        client.admin.command("ping")
        ensure_indexes()
        logger.info("[MONGO] Connessione OK + indici OK")
    except Exception as e:
        logger.error(f"[MONGO] Startup error: {e}")

# ======================================================
# INPUT MODEL
# ======================================================
class InputAnalisi(BaseModel):
    tipologia: Optional[str] = "borsa"
    modello: Optional[str] = "generico"
    foto: str
    id_analisi: Optional[int]
    user_id: Optional[str] = "default"

# ======================================================
# UTILS
# ======================================================
def similarity(a, b):
    return SequenceMatcher(None, a, b).ratio()

def normalize(t):
    if not t:
        return ""
    t = t.lower()
    for sep in [" ", "-", "_"]:
        t = t.replace(sep, "")
    return t

def normalize_model_name(t):
    if not t:
        return ""
    t = t.lower()
    for rm in ["bag", "borsa", "handbag", "chanel", "prada", "gucci", "lv"]:
        t = t.replace(rm, "")
    return t.strip()

# ======================================================
# AVIF -> JPEG
# ======================================================
def convert_avif_to_jpeg(base64_data: str) -> str:
    image_bytes = base64.b64decode(base64_data)
    img = Image.open(io.BytesIO(image_bytes))
    img = img.convert("RGB")
    buffer = io.BytesIO()
    img.save(buffer, format="JPEG", quality=95)
    jpeg_base64 = base64.b64encode(buffer.getvalue()).decode("utf-8")
    return jpeg_base64

# ======================================================
# ANALISI (Mongo)
# ======================================================
def crea_nuova_analisi(user_id: str) -> int:
    db = get_db()
    new_id = get_next_analisi_id()

    db[COL_ANALISI].insert_one({
        "_id": new_id,  # <-- id_analisi INT
        "user_id": user_id,
        "stato": "in_corso",
        "step_corrente": 1,
        "marca_stimata": None,
        "modello_stimato": None,
        "percentuale_contraffazione": None,
        "giudizio_finale": None,
        "created_at": time.time()
    })

    # opzionale: auto-crea user se non esiste
    try:
        db[COL_USERS].update_one(
            {"_id": user_id},
            {"$setOnInsert": {"created_at": time.time(), "role": "user"}},
            upsert=True
        )
    except:
        pass

    return new_id

def salva_foto(id_analisi: int, foto: str) -> int:
    """
    - Converte AVIF -> JPEG base64
    - Calcola step = count + 1
    - Inserisce in analisi_steps (unique id_analisi+step)
    """
    db = get_db()

    # 1) detect + normalize base64
    base64_pura = foto.split(",")[-1]
    raw = base64.b64decode(base64_pura)

    is_avif = raw[4:8] == b'ftyp' and b'avif' in raw[:32]
    if is_avif:
        logger.info("⚠️ Rilevata immagine AVIF → conversione in JPEG...")
        foto_b64 = convert_avif_to_jpeg(base64_pura)
    else:
        foto_b64 = base64_pura

    # 2) step = count+1
    step = db[COL_STEPS].count_documents({"id_analisi": id_analisi}) + 1

    # 3) insert step doc (se collisione, ritenta incrementando)
    while True:
        try:
            db[COL_STEPS].insert_one({
                "id_analisi": id_analisi,
                "step": step,
                "foto_base64": foto_b64,
                "json_response": None,
                "created_at": time.time()
            })
            break
        except DuplicateKeyError:
            step += 1

    return step

def recupera_foto(id_analisi: int) -> List[str]:
    db = get_db()
    rows = list(db[COL_STEPS].find(
        {"id_analisi": id_analisi},
        {"_id": 0, "foto_base64": 1, "step": 1}
    ).sort("step", 1))
    return [r["foto_base64"] for r in rows]

# ======================================================
# PROMPT SYSTEM (Mongo)
# ======================================================
def load_prompt_from_db(name: str, user_id: str = "default") -> Tuple[str, Dict[str, Any]]:
    db = get_db()

    # personalizzato
    row = db[COL_PROMPTS].find_one(
        {"prompt_name": name, "user_id": user_id, "is_active": True},
        {"content": 1, "version": 1}
    )

    # fallback default
    if not row:
        row = db[COL_PROMPTS].find_one(
            {"prompt_name": name, "user_id": "default", "is_active": True},
            {"content": 1, "version": 1}
        )

    if not row:
        raise ValueError(f"Prompt '{name}' non trovato!")

    return row["content"], {"prompt_used": name, "version": row.get("version", 1)}

def load_guardrail(user_id: str):
    try:
        return load_prompt_from_db("vision_guardrail", user_id)
    except:
        return "Non parlare mai di persone o privacy.", {}

def build_prompt(tipologia: str, modello: str, num_foto: int, vademecum: str, user_id: str):
    JSON_RULE = "Rispondi SOLO con JSON valido."

    if num_foto == 1:
        base, meta = load_prompt_from_db("step1_identificazione", user_id)
    elif num_foto < 4:
        base, meta = load_prompt_from_db("step2_intermedio", user_id)
    else:
        base, meta = load_prompt_from_db("step3_finale", user_id)

    guardrail, _ = load_guardrail(user_id)

    final = (
        base.replace("{{GUARDRAIL}}", guardrail)
            .replace("{{TIPOLOGIA}}", tipologia)
            .replace("{{MODELLO}}", modello)
            .replace("{{NUM_FOTO}}", str(num_foto))
            .replace("{{VADEMECUM}}", vademecum)
            .replace("{{JSON_RULE}}", JSON_RULE)
    )

    return final, meta

# ======================================================
# VADEMECUM (FILE SYSTEM) — invariato
# ======================================================
def vademecum_dir() -> str:
    try:
        base = os.path.dirname(os.path.abspath(__file__))
    except:
        base = os.getcwd()
    path = os.path.join(base, "autentica_vademecum")
    logger.info(f"[VADEMECUM] Base directory: {path}")
    return path

def find_brand_folder(brand):
    base = vademecum_dir()
    brand_norm = normalize(brand)

    if not os.path.exists(base):
        return None

    dirs = [d for d in os.listdir(base) if os.path.isdir(os.path.join(base, d))]

    best = None
    best_score = 0
    for d in dirs:
        score = similarity(brand_norm, normalize(d))
        if score > best_score:
            best = d
            best_score = score

    return best if best_score >= 0.55 else None

def load_vademecum(model, brand):
    base = vademecum_dir()
    meta = {
        "model_requested": model,
        "model_norm": normalize(model),
        "brand": brand,
        "file": None,
        "path": None,
        "source": None,
        "match_type": None,
        "fuzzy_score": None
    }

    model_norm = normalize(model)

    folder = find_brand_folder(brand) if brand else None
    if folder:
        folder_path = os.path.join(base, folder)

        # Exact match
        for f in os.listdir(folder_path):
            if f.lower().endswith(".txt"):
                if normalize(os.path.splitext(f)[0]) == model_norm:
                    path = os.path.join(folder_path, f)
                    meta.update({"file": f, "path": path, "source": "model_exact"})
                    return open(path, "r", encoding="utf-8").read(), meta

        # Fuzzy
        best_file = None
        best_score = 0
        for f in os.listdir(folder_path):
            if f.lower().endswith(".txt"):
                score = similarity(model_norm, normalize(os.path.splitext(f)[0]))
                if score > best_score:
                    best_file = f
                    best_score = score

        if best_file and best_score >= 0.60:
            path = os.path.join(folder_path, best_file)
            meta.update({"file": best_file, "path": path, "source": "model_fuzzy", "fuzzy_score": best_score})
            return open(path, "r", encoding="utf-8").read(), meta

        # Brand generic
        generic_path = os.path.join(folder_path, "Generico.txt")
        if os.path.exists(generic_path):
            meta.update({"file": "Generico.txt", "path": generic_path, "source": "brand_generic"})
            return open(generic_path, "r", encoding="utf-8").read(), meta

    # Generic fallback
    general = os.path.join(base, "Generale.txt")
    if os.path.exists(general):
        meta.update({"file": "Generale.txt", "path": general, "source": "fallback_general"})
        return open(general, "r", encoding="utf-8").read(), meta

    meta.update({"source": "fallback_hardcoded"})
    return "Controllare logo, cuciture, hardware, materiali, simmetria e seriale.", meta

# ======================================================
# ENDPOINTS
# ======================================================
@app.post("/analizza-oggetto")
async def analizza_oggetto(input: InputAnalisi):
    db = get_db()

    tipologia = input.tipologia or "borsa"
    user_id = input.user_id or "default"

    # ID analisi
    id_analisi = input.id_analisi
    if not id_analisi:
        id_analisi = crea_nuova_analisi(user_id)

    # Salva foto -> step
    step_corrente = salva_foto(id_analisi, input.foto)

    # Recupera tutte le foto
    immagini = recupera_foto(id_analisi)
    num_foto = len(immagini)

    # Marca/modello da DB (analisi)
    analisi_doc = db[COL_ANALISI].find_one({"_id": id_analisi}, {"marca_stimata": 1, "modello_stimato": 1})

    # 🔥 LOGICA CORRETTA DI IDENTIFICAZIONE:
    if step_corrente == 1:
        modello_finale = normalize_model_name(input.modello)
        marca_precedente = None
    else:
        modello_finale = (analisi_doc or {}).get("modello_stimato")
        marca_precedente = (analisi_doc or {}).get("marca_stimata")

    # VADEMECUM — SOLO basato sulla marca stimata allo step 1 (o precedente)
    t_v_start = time.time()
    vademecum_text, vmeta = load_vademecum(modello_finale, marca_precedente)
    tempo_vademecum = round((time.time() - t_v_start) * 1000, 2)

    # PROMPT
    prompt, meta_prompt = build_prompt(tipologia, modello_finale, num_foto, vademecum_text, user_id)

    # GPT images
    gpt_images = [{"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img}"}} for img in immagini]
    messages = [{
        "role": "user",
        "content": [{"type": "text", "text": prompt}] + gpt_images
    }]

    # GPT call
    t_gpt = time.time()
    resp = client_gpt.chat.completions.create(
        model=DEPLOYMENT_GPT,
        messages=messages
    )
    tempo_gpt = round((time.time() - t_gpt) * 1000, 2)

    raw = resp.choices[0].message.content.strip()

    # Normalizza ```json
    if raw.startswith("```"):
        try:
            raw = raw.split("```")[1]
            raw = raw.replace("json", "").strip()
        except:
            pass

    # Parser JSON robusto
    data = None
    try:
        data = json.loads(raw)
    except:
        match = re.search(r"\{.*\}", raw, re.S)
        if match:
            try:
                data = json.loads(match.group(0))
            except:
                pass

    if data is None:
        data = {
            "percentuale": 100,
            "motivazione": "Errore interno: risposta non valida dal modello AI.",
            "marca_stimata": None,
            "modello_stimato": None,
            "richiedi_altra_foto": False,
            "errore_raw": raw
        }

    # 🔥 BLOCCO MARCA/MODELLO DOPO STEP 1
    if step_corrente == 1:
        brand_gpt = data.get("marca_stimata")
        modello_gpt = data.get("modello_stimato")

        def is_unknown(val):
            if not val:
                return True
            val = str(val).strip().lower()
            return val in ["incerto", "non determinabile", "non determinabile.", "n/d", "?", ""]

        brand_unknown = is_unknown(brand_gpt)

        if brand_unknown:
            data["percentuale"] = 100
            data["motivazione"] = (
                "La foto caricata non permette di identificare marca o modello. "
                "L'analisi viene chiusa con esito di NON AUTENTICITÀ."
            )
            data["richiedi_altra_foto"] = False

            db[COL_ANALISI].update_one(
                {"_id": id_analisi},
                {"$set": {
                    "stato": "completata",
                    "step_corrente": step_corrente,
                    "percentuale_contraffazione": 100,
                    "giudizio_finale": data["motivazione"]
                }}
            )
            # salva anche json nello step
            db[COL_STEPS].update_one(
                {"id_analisi": id_analisi, "step": step_corrente},
                {"$set": {"json_response": data}}
            )
            return data

        # salva marca/modello su analisi
        db[COL_ANALISI].update_one(
            {"_id": id_analisi},
            {"$set": {"marca_stimata": brand_gpt, "modello_stimato": modello_gpt}}
        )

        # ricarica vademecum definitivo
        vademecum_text, vmeta = load_vademecum(modello_gpt or "", brand_gpt)

    else:
        data["marca_stimata"] = marca_precedente
        data["modello_stimato"] = modello_finale

    # STOP LOGIC
    val = str(data.get("richiedi_altra_foto")).lower()
    need_more = not (val in ["false", "0", "no", "n"])
    if num_foto >= MAX_FOTO:
        need_more = False
    data["richiedi_altra_foto"] = need_more

    # ARRICCHIMENTO
    data.update({
        "id_analisi": id_analisi,
        "step": step_corrente,
        "tot_foto": num_foto,
        "prompt_info": {
            "prompt_name": meta_prompt.get("prompt_used"),
            "prompt_version": meta_prompt.get("version"),
            "prompt_char_len": len(prompt)
        },
        "vademecum_info": {
            **vmeta,
            "length_chars": len(vademecum_text)
        },
        "timing": {
            "tempo_chat_gpt_ms": tempo_gpt,
            "tempo_vademecum_ms": tempo_vademecum
        }
    })

    # SALVATAGGIO JSON COMPLETO nello step
    try:
        db[COL_STEPS].update_one(
            {"id_analisi": id_analisi, "step": step_corrente},
            {"$set": {"json_response": data}}
        )
    except Exception as e:
        logger.error(f"[MONGO] ERRORE salvataggio JSON step: {e}")

    # Aggiorna stato finale analisi se stop
    if not need_more:
        db[COL_ANALISI].update_one(
            {"_id": id_analisi},
            {"$set": {
                "stato": "completata",
                "step_corrente": step_corrente,
                "percentuale_contraffazione": data.get("percentuale"),
                "giudizio_finale": data.get("motivazione")
            }}
        )
    else:
        db[COL_ANALISI].update_one(
            {"_id": id_analisi},
            {"$set": {"step_corrente": step_corrente}}
        )

    return data

@app.get("/stato-analisi/{id_analisi}")
def stato_analisi(id_analisi: int):
    db = get_db()

    analisi = db[COL_ANALISI].find_one({"_id": id_analisi})
    if analisi:
        analisi["id"] = analisi.pop("_id")  # per compatibilità visuale con MySQL ("id")

    foto = list(db[COL_STEPS].find(
        {"id_analisi": id_analisi},
        {"_id": 0, "step": 1, "foto_base64": 1, "json_response": 1}
    ).sort("step", 1))

    immagini_base64 = [f.get("foto_base64") for f in foto]

    ultimo_json = None
    if foto:
        ultimo_json = foto[-1].get("json_response")

    return {
        "analisi": analisi,
        "foto": foto,
        "immagini_base64": immagini_base64,
        "ultimo_json": ultimo_json if ultimo_json else None
    }

@app.get("/")
def root():
    return {"status": "ok", "msg": "Autentica backend V2 (Mongo) attivo"}

# ======================================================
# DEBUG ENDPOINTS (utile ora)
# ======================================================
@app.get("/test-mongo")
def test_mongo():
    uri = os.getenv("MONGO_URI")
    if not uri:
        return {"status": "error", "error": "MONGO_URI missing"}
    try:
        client = MongoClient(uri, serverSelectionTimeoutMS=5000, retryWrites=False)
        client.admin.command("ping")
        return {"status": "ok", "host": client.address[0], "port": client.address[1], "db": MONGO_DB_NAME}
    except Exception as e:
        return {"status": "error", "error": str(e)}

@app.get("/debug-dns")
def debug_dns():
    host = "autentica.global.mongocluster.cosmos.azure.com"
    try:
        return {"host": host, "resolved": socket.getaddrinfo(host, None)}
    except Exception as e:
        return {"host": host, "error": str(e)}
