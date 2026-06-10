# modules/analyzer.py — Análisis de comportamiento financiero

from datetime import date

import numpy as np
import pandas as pd

from config import CATEGORIAS


# ---------------------------------------------------------------------------
# Métricas de resumen
# ---------------------------------------------------------------------------

def resumen_mensual(df: pd.DataFrame) -> pd.DataFrame:
    """
    Devuelve un DataFrame con ingreso, gasto, ahorro y balance neto por mes.
    """
    df = df.copy()
    df["mes"] = df["fecha"].dt.to_period("M")
    gastos = df[df["importe"] < 0].groupby("mes")["importe"].sum().abs().rename("gasto_total")
    ingresos = df[df["importe"] > 0].groupby("mes")["importe"].sum().rename("ingreso_total")
    resumen = pd.concat([ingresos, gastos], axis=1).fillna(0)
    resumen["ahorro_neto"] = resumen["ingreso_total"] - resumen["gasto_total"]
    resumen["tasa_ahorro"] = (resumen["ahorro_neto"] / resumen["ingreso_total"].replace(0, np.nan)).fillna(0)
    resumen.index = resumen.index.astype(str)
    return resumen.reset_index().rename(columns={"mes": "mes"})


def gasto_por_categoria_mes(df: pd.DataFrame) -> pd.DataFrame:
    """DataFrame pivotado: filas=mes, columnas=categoría, valores=gasto €."""
    df = df.copy()
    gastos = df[df["importe"] < 0].copy()
    gastos["mes"] = gastos["fecha"].dt.to_period("M").astype(str)
    gastos["importe_abs"] = gastos["importe"].abs()
    pivot = gastos.pivot_table(
        index="mes", columns="categoria", values="importe_abs",
        aggfunc="sum", fill_value=0
    )
    return pivot.reset_index()


def top_categorias(df: pd.DataFrame, n: int = 5) -> pd.Series:
    """Las n categorías con mayor gasto total."""
    gastos = df[df["importe"] < 0].copy()
    gastos["importe_abs"] = gastos["importe"].abs()
    return gastos.groupby("categoria")["importe_abs"].sum().nlargest(n)


def kpis_globales(df: pd.DataFrame) -> dict:
    """KPIs de resumen para el dashboard."""
    gastos = df[df["importe"] < 0]["importe"].sum()
    ingresos = df[df["importe"] > 0]["importe"].sum()
    ahorro = ingresos + gastos  # gastos ya es negativo
    saldo_actual = df["saldo_acumulado"].iloc[-1] if "saldo_acumulado" in df.columns else ahorro
    n_meses = df["fecha"].dt.to_period("M").nunique() or 1

    return {
        "saldo_actual":        round(saldo_actual, 2),
        "gasto_total":         round(abs(gastos), 2),
        "ingreso_total":       round(ingresos, 2),
        "ahorro_total":        round(ahorro, 2),
        "gasto_mensual_medio": round(abs(gastos) / n_meses, 2),
        "ingreso_mensual_medio": round(ingresos / n_meses, 2),
        "tasa_ahorro_media":   round(ahorro / ingresos * 100 if ingresos else 0, 1),
    }


# ---------------------------------------------------------------------------
# Detección de anomalías
# ---------------------------------------------------------------------------

def detectar_anomalias(df: pd.DataFrame, umbral_z: float = 2.5) -> pd.DataFrame:
    """
    Marca transacciones cuyo importe absoluto está a más de `umbral_z`
    desviaciones estándar de la media de su categoría.
    """
    df = df.copy()
    df["anomalia"] = False
    df["z_score"]  = 0.0

    gastos = df[df["importe"] < 0].copy()
    gastos["importe_abs"] = gastos["importe"].abs()

    for cat, grupo in gastos.groupby("categoria"):
        if len(grupo) < 3:
            continue
        media = grupo["importe_abs"].mean()
        std   = grupo["importe_abs"].std()
        if std == 0:
            continue
        z = (grupo["importe_abs"] - media) / std
        idx_anomalos = z[z.abs() > umbral_z].index
        df.loc[idx_anomalos, "anomalia"] = True
        df.loc[idx_anomalos, "z_score"]  = z[idx_anomalos].round(2)

    return df


# ---------------------------------------------------------------------------
# Tendencias temporales
# ---------------------------------------------------------------------------

def tendencia_gasto(df: pd.DataFrame) -> pd.DataFrame:
    """
    Regresión lineal simple sobre el gasto mensual total.
    Devuelve pendiente (€/mes) e intercepto.
    """
    resumen = resumen_mensual(df)
    if len(resumen) < 2:
        return {"pendiente": 0.0, "interpretacion": "Datos insuficientes"}

    x = np.arange(len(resumen))
    y = resumen["gasto_total"].values
    pendiente, intercept = np.polyfit(x, y, 1)

    if pendiente > 5:
        interp = f"⚠️ Gasto creciente (+{pendiente:.1f} €/mes)"
    elif pendiente < -5:
        interp = f"✅ Gasto decreciente ({pendiente:.1f} €/mes)"
    else:
        interp = "➡️ Gasto estable"

    return {"pendiente": round(pendiente, 2), "interpretacion": interp}


def gasto_diario_semana(df: pd.DataFrame) -> pd.DataFrame:
    """Gasto medio por día de la semana."""
    gastos = df[df["importe"] < 0].copy()
    gastos["dia_semana"] = gastos["fecha"].dt.day_name()
    orden = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    nombres_es = ["Lun", "Mar", "Mié", "Jue", "Vie", "Sáb", "Dom"]
    gastos["importe_abs"] = gastos["importe"].abs()
    resultado = (
        gastos.groupby("dia_semana")["importe_abs"]
        .mean()
        .reindex(orden)
        .fillna(0)
        .reset_index()
    )
    resultado["dia_es"] = nombres_es
    return resultado
