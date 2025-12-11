import os
import json
import time
from typing import Optional
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import mysql.connector
from mysql.connector.constants import ClientFlag
from difflib import SequenceMatcher
from openai import AzureOpenAI
import logging
import uvicorn
from PIL import Image
import pillow_avif
import base64
import io
# LOGGING
logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

# ======================================================
# CONFIG OPENAI
# ======================================================
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT", "https://autenticagpt.openai.azure.com/")
AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY", "AzJeZeCGrzTToeBSYBjvQFiVbk3YW2xLI00YhAZFM5wxaEx6yH3xJQQJ99BKACfhMk5XJ3w3AAABACOGY4tA")
AZURE_OPENAI_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION", "2024-12-01-preview")
DEPLOYMENT_GPT = os.getenv("DEPLOYMENT_GPT", "gpt-5.1-chat")

client_gpt = AzureOpenAI(
    azure_endpoint=AZURE_OPENAI_ENDPOINT,
    api_key=AZURE_OPENAI_API_KEY,
    api_version=AZURE_OPENAI_API_VERSION,
)

MAX_FOTO = 7  # limite max

# ============================================
# FASTAPI
# ============================================
app = FastAPI(title="Autentica V2 Backend", version="6.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)


MYSQL_HOST = os.getenv("MYSQL_HOST", "autenticamysql.mysql.database.azure.com")
MYSQL_USER = os.getenv("MYSQL_USER", "autentica_admin@autenticamysql")
MYSQL_PASSWORD = os.getenv("MYSQL_PASSWORD", "autentica@Admin")
MYSQL_DATABASE = os.getenv("MYSQL_DATABASE", "autentica")
# ============================================
# MYSQL
# ============================================



BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SSL_CA = os.path.join(BASE_DIR, "certs", "BaltimoreCyberTrustRoot.crt.pem")

def get_mysql_connection():
    return mysql.connector.connect(
        host=MYSQL_HOST,
        user=MYSQL_USER,
        password=MYSQL_PASSWORD,
        database=MYSQL_DATABASE,
        ssl_ca=SSL_CA,
        ssl_verify_cert=True,
        client_flags=[ClientFlag.SSL],
        ssl_disabled=False,
        ssl_verify_identity=False   # <-- CHIAVE DELLA SOLUZIONE
    )


# ============================================
# INPUT MODEL
# ============================================
class InputAnalisi(BaseModel):
    tipologia: Optional[str] = "borsa"
    modello: Optional[str] = "generico"
    foto: str
    id_analisi: Optional[int]
    user_id: Optional[str] = "default"

# ============================================
# UTILS
# ============================================
def similarity(a, b):
    return SequenceMatcher(None, a, b).ratio()

def normalize(t):
    if not t: return ""
    t = t.lower()
    for sep in [" ", "-", "_"]:
        t = t.replace(sep, "")
    return t

def normalize_model_name(t):
    if not t: return ""
    t = t.lower()
    for rm in ["bag", "borsa", "handbag", "chanel", "prada", "gucci", "lv"]:
        t = t.replace(rm, "")
    return t.strip()

# ============================================
# ANALISI DB
# ============================================
def crea_nuova_analisi(user_id):
    cnx = get_mysql_connection()
    cur = cnx.cursor()
    cur.execute("""
        INSERT INTO analisi (user_id, stato, step_corrente)
        VALUES (%s, 'in_corso', 1)
    """, (user_id,))
    cnx.commit()
    new_id = cur.lastrowid
    cur.close()
    cnx.close()
    return new_id


def salva_foto(id_analisi, foto):
    # -------------------------------------
    # 1) Detect AVIF + conversione a JPEG
    # -------------------------------------
    base64_pura = foto.split(",")[-1]  # rimuove eventuale header data:
    raw = base64.b64decode(base64_pura)

    # AVIF detection (super affidabile)
    is_avif = raw[4:8] == b'ftyp' and b'avif' in raw[:32]

    if is_avif:
        print("⚠️ Rilevata immagine AVIF → conversione in JPEG...")
        foto = convert_avif_to_jpeg(base64_pura)   # restituisce SOLO base64
    else:
        # garantiamo che foto sia solo base64 senza header
        foto = base64_pura

    # -------------------------------------
    # 2) SALVATAGGIO NEL DB
    # -------------------------------------
    cnx = get_mysql_connection()
    cur = cnx.cursor()

    cur.execute("SELECT COUNT(*) FROM analisi_foto WHERE id_analisi=%s", (id_analisi,))
    step = cur.fetchone()[0] + 1

    cur.execute("""
        INSERT INTO analisi_foto (id_analisi, step, foto_base64)
        VALUES (%s, %s, %s)
    """, (id_analisi, step, foto))

    cnx.commit()
    cur.close()
    cnx.close()

    return step

def convert_avif_to_jpeg(base64_data: str) -> str:
    # decode base64 → bytes
    image_bytes = base64.b64decode(base64_data)
    
    # open AVIF in memory
    img = Image.open(io.BytesIO(image_bytes))
    
    # convert to RGB (JPEG non supporta alpha)
    img = img.convert("RGB")
    
    # save as JPEG
    buffer = io.BytesIO()
    img.save(buffer, format="JPEG", quality=95)
    
    # re-encode to base64
    jpeg_base64 = base64.b64encode(buffer.getvalue()).decode("utf-8")
    
    return jpeg_base64


def recupera_foto(id_analisi):
    cnx = get_mysql_connection()
    cur = cnx.cursor()
    cur.execute("""
        SELECT foto_base64 FROM analisi_foto
        WHERE id_analisi=%s ORDER BY step ASC
    """, (id_analisi,))
    rows = cur.fetchall()
    cur.close()
    cnx.close()
    return [r[0] for r in rows]

# ============================================
# PROMPT SYSTEM
# ============================================
def load_prompt_from_db(name, user_id="default"):
    cnx = get_mysql_connection()
    cur = cnx.cursor(dictionary=True)

    # personalizzato
    cur.execute("""
        SELECT content, version
        FROM prompt_versions
        WHERE prompt_name=%s AND user_id=%s AND is_active=1
        LIMIT 1
    """, (name, user_id))
    row = cur.fetchone()

    # fallback default
    if not row:
        cur.execute("""
            SELECT content, version
            FROM prompt_versions
            WHERE prompt_name=%s AND user_id='default' AND is_active=1
            LIMIT 1
        """, (name,))
        row = cur.fetchone()

    cur.close()
    cnx.close()

    if not row:
        raise ValueError(f"Prompt '{name}' non trovato!")

    return row["content"], {"prompt_used": name, "version": row["version"]}

def load_guardrail(user_id):
    try:
        return load_prompt_from_db("vision_guardrail", user_id)
    except:
        return "Non parlare mai di persone o privacy.", {}

def build_prompt(tipologia, modello, num_foto, vademecum, user_id):
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

# ============================================
# VADEMECUM – LOG COMPLETE
# ============================================
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

    # Brand folder
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






@app.post("/analizza-oggetto")
async def analizza_oggetto(input: InputAnalisi):

    tipologia = input.tipologia or "borsa"
    user_id = input.user_id or "default"

    # ID analisi
    id_analisi = input.id_analisi
    if not id_analisi:
        id_analisi = crea_nuova_analisi(user_id)

    # Salva foto
    step_corrente = salva_foto(id_analisi, input.foto)

    # Recupera tutte le foto
    immagini = recupera_foto(id_analisi)
    num_foto = len(immagini)

    # Marca/modello da DB
    cnx = get_mysql_connection()
    cur = cnx.cursor(dictionary=True)
    cur.execute("SELECT marca_stimata, modello_stimato FROM analisi WHERE id=%s", (id_analisi,))
    row = cur.fetchone()
    cur.close()
    cnx.close()

    # 🔥 LOGICA CORRETTA DI IDENTIFICAZIONE:
    # SOLO AL PRIMO STEP GPT può proporre marca/modello
    if step_corrente == 1:
        modello_finale = normalize_model_name(input.modello)
        marca_precedente = None
    else:
        modello_finale = row["modello_stimato"]   # fisso dal DB
        marca_precedente = row["marca_stimata"]   # fisso dal DB

    # VADEMECUM — SOLO basato sulla marca stimata ALLO STEP 1
    t_v_start = time.time()
    vademecum_text, vmeta = load_vademecum(modello_finale, marca_precedente)
    tempo_vademecum = round((time.time() - t_v_start) * 1000, 2)

    # PROMPT
    prompt, meta_prompt = build_prompt(tipologia, modello_finale, num_foto, vademecum_text, user_id)

    # GPT
    gpt_images = [{"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img}"}} for img in immagini]

    messages = [{
        "role": "user",
        "content": [{"type": "text", "text": prompt}] + gpt_images
    }]

    t_gpt = time.time()
    resp = client_gpt.chat.completions.create(
        model=DEPLOYMENT_GPT,
        messages=messages
    )
    tempo_gpt = round((time.time() - t_gpt) * 1000, 2)

    raw = resp.choices[0].message.content.strip()
    
    # Normalizza eventuali blocchi ```json
    if raw.startswith("```"):
        try:
            raw = raw.split("```")[1]
            raw = raw.replace("json", "").strip()
        except:
            pass
    
    # PARSER JSON ROBUSTO
    data = None
    try:
        data = json.loads(raw)
    except:
        # PROVA A RECUPERARE SOLO LA PARTE JSON TRA { }
        import re
        match = re.search(r"\{.*\}", raw, re.S)
        if match:
            try:
                data = json.loads(match.group(0))
            except:
                pass
    
    # SE ANCORA NON È PARSABILE → FALLBACK DI SICUREZZA
    if data is None:
        data = {
            "percentuale": 100,
            "motivazione": "Errore interno: risposta non valida dal modello AI.",
            "marca_stimata": None,
            "modello_stimato": None,
            "richiedi_altra_foto": False,
            "errore_raw": raw
        }


    # 🔥 CORREZIONE DEFINITIVA:
    # GPT NON DEVE CAMBIARE MARCA/MODELLO DOPO LO STEP 1
    if step_corrente == 1:
    
        brand_gpt = data.get("marca_stimata")
        modello_gpt = data.get("modello_stimato")
    
        # Normalizziamo i casi "non identificazione"
        def is_unknown(val):
            if not val:
                return True
            val = str(val).strip().lower()
            return val in ["incerto", "non determinabile", "non determinabile.", "n/d", "?", ""]
    
        brand_unknown = is_unknown(brand_gpt)
        modello_unknown = is_unknown(modello_gpt)
    
        # 🔥 SE NON ABBIAMO RICONOSCIUTO LA MARCA O IL MODELLO → CHIUSURA IMMEDIATA
        if brand_unknown: # or modello_unknown:
    
            data["percentuale"] = 100
            data["motivazione"] = (
                "La foto caricata non permette di identificare marca o modello. "
                "L'analisi viene chiusa con esito di NON AUTENTICITÀ."
            )
            data["richiedi_altra_foto"] = False
    
            # Aggiorna analisi nel DB come fallita
            cnx = get_mysql_connection()
            cur = cnx.cursor()
            cur.execute("""
                UPDATE analisi
                SET stato='completata',
                    step_corrente=%s,
                    percentuale_contraffazione=100,
                    giudizio_finale=%s
                WHERE id=%s
            """, (
                step_corrente,
                data["motivazione"],
                id_analisi
            ))
            cnx.commit()
            cur.close()
            cnx.close()
    
            return data
    
        # Se marca/modello validi → salviamo
        cnx = get_mysql_connection()
        cur = cnx.cursor()
        cur.execute("""
            UPDATE analisi
            SET marca_stimata=%s, modello_stimato=%s
            WHERE id=%s
        """, (brand_gpt, modello_gpt, id_analisi))
        cnx.commit()
        cur.close()
        cnx.close()
    
        # Carichiamo vademecum definitivo
        vademecum_text, vmeta = load_vademecum(modello_gpt or "", brand_gpt)
    
    else:
        # 🔒 Step successivi: NON modificare mai marca/modello
        data["marca_stimata"] = marca_precedente
        data["modello_stimato"] = modello_finale



    # STOP LOGIC
    val = str(data.get("richiedi_altra_foto")).lower()
    need_more = not (val in ["false", "0", "no", "n"])

    if num_foto >= MAX_FOTO:
        need_more = False

    data["richiedi_altra_foto"] = need_more

    # ARRICCHIMENTO FINALE JSON
    data.update({
        "id_analisi": id_analisi,
        "step": step_corrente,
        "tot_foto": num_foto,
        "prompt_info": {
            "prompt_name": meta_prompt["prompt_used"],
            "prompt_version": meta_prompt["version"],
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

    # SALVATAGGIO JSON COMPLETO
    try:
        cnx = get_mysql_connection()
        cur = cnx.cursor()
        cur.execute("""
            UPDATE analisi_foto
            SET json_response=%s
            WHERE id_analisi=%s AND step=%s
        """, (json.dumps(data, ensure_ascii=False), id_analisi, step_corrente))
        cnx.commit()
        cur.close()
        cnx.close()
    except Exception as e:
        print("[DB] ERRORE salvataggio JSON:", e)

    # Aggiorna stato finale analisi
    if not need_more:
        cnx = get_mysql_connection()
        cur = cnx.cursor()
        cur.execute("""
            UPDATE analisi 
            SET stato='completata',
                step_corrente=%s,
                percentuale_contraffazione=%s,
                giudizio_finale=%s
            WHERE id=%s
        """, (
            step_corrente,
            data.get("percentuale"),
            data.get("motivazione"),
            id_analisi
        ))
        cnx.commit()
        cur.close()
        cnx.close()

    return data


# ============================================
# STATO ANALISI
# ============================================

@app.get("/stato-analisi/{id_analisi}")
def stato_analisi(id_analisi: int):

    cnx = get_mysql_connection()
    cur = cnx.cursor(dictionary=True)

    cur.execute("SELECT * FROM analisi WHERE id=%s", (id_analisi,))
    analisi = cur.fetchone()

    # FOTO E JSON
    cur.execute("""
        SELECT step, foto_base64, json_response
        FROM analisi_foto
        WHERE id_analisi=%s
        ORDER BY step ASC
    """, (id_analisi,))
    foto = cur.fetchall()

    cur.close()
    cnx.close()

    immagini_base64 = [f["foto_base64"] for f in foto]

    ultimo_json = None
    if foto:
        ultimo_json = foto[-1]["json_response"]

    return {
        "analisi": analisi,
        "foto": foto,
        "immagini_base64": immagini_base64,
        "ultimo_json": json.loads(ultimo_json) if ultimo_json else None
    }


# ============================================
# HEALTHCHECK
# ============================================

@app.get("/test-mysql")
def test_mysql():
    try:
        cnx = mysql.connector.connect(
            host=MYSQL_HOST,
            user=MYSQL_USER,
            password=MYSQL_PASSWORD,
            database=MYSQL_DATABASE,
            ssl_ca=SSL_CA,
            ssl_verify_cert=True,
            ssl_verify_identity=False
        )
        cnx.close()
        return {"status": "ok"}
    except Exception as e:
        return {
            "status": "error",
            "error": str(e),
            "user": MYSQL_USER,
            "ssl_ca": SSL_CA
        }


@app.get("/debug/fs")
def debug_fs():
    import os

    base_path = "/home/site/wwwroot"
    cert_path = "/home/site/wwwroot/certs/BaltimoreCyberTrustRoot.crt.pem"

    return {
        "exists_wwwroot": os.path.exists(base_path),
        "exists_certs_dir": os.path.exists(os.path.join(base_path, "certs")),
        "exists_cert_file": os.path.exists(cert_path),
        "list_wwwroot": os.listdir(base_path) if os.path.exists(base_path) else "missing",
        "list_certs": os.listdir(os.path.join(base_path, "certs")) if os.path.exists(os.path.join(base_path, "certs")) else "missing",
        "working_dir": os.getcwd()
    }

@app.get("/whereami")
def whereami():
    import os
    return {
        "cwd": os.getcwd(),
        "files": os.listdir("."),
        "absolute_path": os.path.abspath("."),
    }

@app.get("/__routes__")
def list_routes():
    return [{"path": r.path, "name": r.name} for r in app.routes]
    
@app.get("/ssl-info")
def ssl_info():
    import ssl, socket, traceback
    
    host = MYSQL_HOST
    port = 3306

    try:
        context = ssl.create_default_context()

        conn = context.wrap_socket(
            socket.socket(socket.AF_INET),
            server_hostname=host
        )
        conn.settimeout(5)
        conn.connect((host, port))

        cert = conn.getpeercert()
        conn.close()

        return {
            "status": "ok",
            "cert": cert
        }

    except Exception as e:
        return {
            "status": "error",
            "error": str(e),
            "trace": traceback.format_exc()
        }







