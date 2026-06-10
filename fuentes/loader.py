# data/loader.py — Carga de datos desde todas las fuentes

import io
import base64

import pandas as pd
import requests

from config import BACKEND_URL
from modulos.categorizer import categorizar_dataframe


# ---------------------------------------------------------------------------
# 1. DESDE TRUELAYER (vía FastAPI backend)
# ---------------------------------------------------------------------------

def cargar_desde_truelayer(month: str = None) -> pd.DataFrame:
    """
    Llama al endpoint /statement del backend FastAPI y devuelve un DataFrame.
    Si no hay token activo, lanza RuntimeError.
    """
    try:
        url = f"{BACKEND_URL}/statement"
        params = {"month": month} if month else {}
        r = requests.get(url, params=params, timeout=10)
        r.raise_for_status()
        data = r.json()
    except requests.exceptions.ConnectionError:
        raise RuntimeError(
            "No se puede conectar con el backend FastAPI. "
            "Asegúrate de que está corriendo en localhost:8000."
        )
    except requests.exceptions.HTTPError as e:
        raise RuntimeError(f"Error del backend: {e.response.text}")

    transacciones = data.get("transactions", [])
    if not transacciones:
        raise RuntimeError("El backend no devolvió transacciones. Verifica el token OAuth.")

    df = pd.DataFrame(transacciones)
    df = df.rename(columns={"date": "fecha", "description": "descripcion",
                             "amount": "importe", "currency": "divisa"})
    df["fecha"] = pd.to_datetime(df["fecha"])
    df = df.sort_values("fecha").reset_index(drop=True)
    df["saldo_acumulado"] = df["importe"].cumsum()

    df = categorizar_dataframe(df)
    return df


# ---------------------------------------------------------------------------
# 2. DESDE CSV / EXCEL (upload de Dash)
# ---------------------------------------------------------------------------

COLUMNAS_REQUERIDAS = {
    # nombres que el usuario puede traer → nombre interno
    "fecha":       ["fecha", "date", "Fecha", "Date", "FECHA"],
    "descripcion": ["descripcion", "descripción", "description", "concepto",
                    "Concepto", "Descripcion", "Description"],
    "importe":     ["importe", "amount", "cantidad", "Importe", "Amount",
                    "Cantidad", "importe (eur)"],
    "divisa":      ["divisa", "currency", "moneda", "Divisa", "Currency"],
}


def cargar_desde_upload(contents: str, filename: str) -> pd.DataFrame:
    """
    Recibe el contenido base64 de un componente dcc.Upload de Dash
    y devuelve un DataFrame limpio.
    """
    content_type, content_string = contents.split(",")
    decoded = base64.b64decode(content_string)

    if filename.endswith(".csv"):
        df = pd.read_csv(io.StringIO(decoded.decode("utf-8")), sep=None, engine="python")
    elif filename.endswith((".xls", ".xlsx")):
        df = pd.read_excel(io.BytesIO(decoded))
    else:
        raise ValueError(f"Formato no soportado: {filename}. Usa CSV o Excel.")

    df = _normalizar_columnas(df)
    df = _limpiar(df)
    df = categorizar_dataframe(df)
    return df


def _normalizar_columnas(df: pd.DataFrame) -> pd.DataFrame:
    """Renombra columnas del archivo al esquema interno."""
    rename_map = {}
    df.columns = df.columns.str.strip()
    for col_interna, alias in COLUMNAS_REQUERIDAS.items():
        for a in alias:
            if a in df.columns:
                rename_map[a] = col_interna
                break
    df = df.rename(columns=rename_map)

    # Asegurar que existen las columnas mínimas
    for col in ["fecha", "descripcion", "importe"]:
        if col not in df.columns:
            raise ValueError(
                f"No se encontró la columna '{col}' en el archivo. "
                f"Columnas detectadas: {list(df.columns)}"
            )
    if "divisa" not in df.columns:
        df["divisa"] = "EUR"
    return df


def _limpiar(df: pd.DataFrame) -> pd.DataFrame:
    df = df.dropna(subset=["fecha", "importe"]).copy()
    df["fecha"] = pd.to_datetime(df["fecha"], dayfirst=True, errors="coerce")
    df = df.dropna(subset=["fecha"])

    # Si Excel ya leyó el número como float, no lo tocamos
    if pd.api.types.is_numeric_dtype(df["importe"]):
        df["importe"] = pd.to_numeric(df["importe"], errors="coerce")
    else:
        col = df["importe"].astype(str).str.strip().str.replace("€", "", regex=False).str.replace(" ", "", regex=False)

        def _parse_importe(s):
            s = s.strip()
            if "," in s and "." in s:
                if s.rfind(",") > s.rfind("."):
                    s = s.replace(".", "").replace(",", ".")
                else:
                    s = s.replace(",", "")
            elif "," in s:
                s = s.replace(",", ".")
            try:
                return float(s)
            except ValueError:
                return float("nan")

        df["importe"] = col.apply(_parse_importe)

    df["importe"] = pd.to_numeric(df["importe"], errors="coerce")
    df = df.dropna(subset=["importe"])
    df = df.sort_values("fecha").reset_index(drop=True)
    df["saldo_acumulado"] = df["importe"].cumsum()
    return df


# ---------------------------------------------------------------------------
# 3. DESDE DATOS SINTÉTICOS
# ---------------------------------------------------------------------------

def cargar_sinteticos(meses: int = 12, ingreso_base: float = 1800.0,
                      perfil: str = "medio", seed: int = 42) -> pd.DataFrame:
    from fuentes.generator import generar_transacciones
    return generar_transacciones(
        meses=meses,
        ingreso_base=ingreso_base,
        perfil_riesgo=perfil,
        seed=seed,
    )
