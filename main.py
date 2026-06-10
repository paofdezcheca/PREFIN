# main.py — Backend FastAPI para conectar con TrueLayer (Open Banking)

import os
import sqlite3
from datetime import date, datetime
from urllib.parse import urlencode

import requests
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse

# ---------------------------------------------------------------------------
# Configuración
# ---------------------------------------------------------------------------
load_dotenv()

CLIENT_ID = os.getenv("TRUELAYER_CLIENT_ID")
CLIENT_SECRET = os.getenv("TRUELAYER_CLIENT_SECRET")
REDIRECT_URI = os.getenv("TRUELAYER_REDIRECT_URI", "http://localhost:8000/callback")

AUTH_BASE = "https://auth.truelayer-sandbox.com"
TOKEN_URL = "https://auth.truelayer-sandbox.com/connect/token"
DATA_BASE = "https://api.truelayer-sandbox.com/data/v1"

if not CLIENT_ID or not CLIENT_SECRET:
    print("⚠️  Faltan variables TRUELAYER_CLIENT_ID / TRUELAYER_CLIENT_SECRET en .env")
    print("    El backend arrancará pero los endpoints de TrueLayer no funcionarán.")

# ---------------------------------------------------------------------------
# Inicializar app (crear ANTES de añadir middleware)
# ---------------------------------------------------------------------------
app = FastAPI(title="PREFIN — Backend Open Banking")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8050", "http://127.0.0.1:8050"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

DB_PATH = "tokens.db"


# ---------------------------------------------------------------------------
# Base de datos
# ---------------------------------------------------------------------------
def init_db():
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS tokens (
                id INTEGER PRIMARY KEY,
                state TEXT,
                access_token TEXT,
                created_at TEXT
            )
        """)
        conn.commit()


def insert_token(access_token: str):
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute(
            "INSERT INTO tokens (state, access_token, created_at) VALUES (?, ?, ?)",
            ("authlink", access_token, datetime.utcnow().isoformat()),
        )
        conn.commit()


def get_token():
    with sqlite3.connect(DB_PATH) as conn:
        row = conn.execute(
            "SELECT access_token FROM tokens WHERE access_token IS NOT NULL "
            "ORDER BY id DESC LIMIT 1"
        ).fetchone()
    return row[0] if row else None


init_db()


# ---------------------------------------------------------------------------
# Utilidades
# ---------------------------------------------------------------------------
def month_range(yyyy_mm: str):
    y, m = map(int, yyyy_mm.split("-"))
    start = date(y, m, 1)
    end = date(y + 1, 1, 1) if m == 12 else date(y, m + 1, 1)
    return start.isoformat(), end.isoformat()


def tl_get(path: str, token: str, params: dict = None):
    headers = {"Authorization": f"Bearer {token}"}
    r = requests.get(DATA_BASE + path, headers=headers, params=params)
    if r.status_code >= 400:
        raise HTTPException(status_code=500, detail=r.text)
    return r.json()


# ---------------------------------------------------------------------------
# Rutas
# ---------------------------------------------------------------------------
@app.get("/")
def home():
    return {"ok": True, "next": "Ve a /connect para iniciar OAuth"}


AUTH_LINK = (
    "https://auth.truelayer-sandbox.com/?response_type=code"
    "&client_id=sandbox-tfgopenbanking-b0cddd"
    "&scope=info%20accounts%20balance%20cards%20transactions"
    "%20direct_debits%20standing_orders%20offline_access"
    "&redirect_uri=http://localhost:8000/callback"
    "&providers=uk-cs-mock%20uk-ob-all%20uk-oauth-all"
)


@app.get("/connect")
def connect():
    return RedirectResponse(AUTH_LINK)


@app.get("/callback")
def callback(code: str = Query(...)):
    if not CLIENT_ID or not CLIENT_SECRET:
        raise HTTPException(status_code=500,
                            detail="Faltan credenciales TrueLayer en .env")

    data = {
        "grant_type":    "authorization_code",
        "client_id":     CLIENT_ID,
        "client_secret": CLIENT_SECRET,
        "redirect_uri":  REDIRECT_URI,
        "code":          code,
    }
    r = requests.post(TOKEN_URL, data=data)
    token = r.json().get("access_token")

    if not token:
        raise HTTPException(status_code=500,
                            detail=f"No se obtuvo access_token: {r.text}")

    insert_token(token)
    return {
        "ok": True,
        "message": "Token guardado. Vuelve a la app de Dash y pulsa 'Cargar transacciones'.",
    }


@app.get("/statement")
def statement(month: str = Query(None)):
    token = get_token()
    if not token:
        raise HTTPException(status_code=400,
                            detail="No hay token activo. Ve a /connect primero.")

    accounts = tl_get("/accounts", token).get("results", [])
    if not accounts:
        raise HTTPException(status_code=404, detail="No se encontraron cuentas.")
    account_id = accounts[0]["account_id"]

    params = {}
    if month:
        start, end = month_range(month)
        params = {"from": start, "to": end}

    transactions = tl_get(
        f"/accounts/{account_id}/transactions", token, params
    ).get("results", [])

    balance = tl_get(
        f"/accounts/{account_id}/balance", token
    ).get("results", [])

    simplified = [
        {
            "date":        t.get("timestamp"),
            "description": t.get("description"),
            "amount":      t.get("amount"),
            "currency":    t.get("currency"),
        }
        for t in transactions
    ]

    return {
        "month":        month,
        "transactions": simplified,
        "balance":      balance,
    }


@app.get("/debug-token")
def debug_token():
    with sqlite3.connect(DB_PATH) as conn:
        rows = conn.execute("SELECT COUNT(*) FROM tokens").fetchone()[0]
        last = conn.execute(
            "SELECT access_token FROM tokens ORDER BY id DESC LIMIT 1"
        ).fetchone()
    return {
        "db_path": DB_PATH,
        "rows": rows,
        "has_token": bool(last and last[0]),
    }
