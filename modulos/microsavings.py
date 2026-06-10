# modules/microsavings.py — Módulo de micro-ahorro automático

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Redondeo y cálculo de micro-ahorro
# ---------------------------------------------------------------------------

OPCIONES_REDONDEO = {
    "€1 (redondeo estándar)":   1,
    "€2 (redondeo moderado)":   2,
    "€5 (redondeo agresivo)":   5,
    "€10 (máximo ahorro)":      10,
}


def calcular_microahorro(
    df: pd.DataFrame,
    unidad_redondeo: float = 1.0,
) -> pd.DataFrame:
    """
    Para cada gasto (importe < 0), calcula cuánto se habría ahorrado
    redondeando al siguiente múltiplo de `unidad_redondeo`.

    Ejemplo: gasto de 2.30 € con redondeo a 1 € → ahorro de 0.70 €.

    Devuelve el DataFrame original con columna adicional `microahorro`.
    """
    df = df.copy()
    gastos_mask = df["importe"] < 0
    importes_abs = df.loc[gastos_mask, "importe"].abs()

    # Redondeo hacia arriba al siguiente múltiplo
    redondeado = np.ceil(importes_abs / unidad_redondeo) * unidad_redondeo
    df["microahorro"] = 0.0
    df.loc[gastos_mask, "microahorro"] = (redondeado - importes_abs).round(2)

    return df


def resumen_microahorro(
    df: pd.DataFrame,
    unidad_redondeo: float = 1.0,
) -> dict:
    """
    Devuelve métricas clave del micro-ahorro para el período completo.
    """
    df_micro = calcular_microahorro(df, unidad_redondeo)

    total_microahorro = df_micro["microahorro"].sum()
    n_meses = df["fecha"].dt.to_period("M").nunique() or 1
    microahorro_mensual = total_microahorro / n_meses

    # Por mes
    df_micro["mes"] = df_micro["fecha"].dt.to_period("M").astype(str)
    por_mes = df_micro.groupby("mes")["microahorro"].sum().reset_index()
    por_mes.columns = ["mes", "microahorro"]
    por_mes["acumulado"] = por_mes["microahorro"].cumsum()

    # Proyección anual
    proyeccion_anual = microahorro_mensual * 12

    # Número de transacciones con redondeo
    n_transacciones = int((df_micro["microahorro"] > 0).sum())

    return {
        "total":               round(total_microahorro, 2),
        "mensual_medio":       round(microahorro_mensual, 2),
        "proyeccion_anual":    round(proyeccion_anual, 2),
        "n_transacciones":     n_transacciones,
        "por_mes":             por_mes,
        "unidad_redondeo":     unidad_redondeo,
    }


def objetivos_ahorro(
    microahorro_anual: float,
) -> list[dict]:
    """
    Muestra en cuánto tiempo se alcanzarían metas típicas de ahorro.
    """
    metas = [
        {"nombre": "Fondo de emergencia (1 mes nómina)", "importe": 1800},
        {"nombre": "Vacaciones",                          "importe": 800},
        {"nombre": "Ordenador nuevo",                     "importe": 1200},
        {"nombre": "Fondo emergencia (3 meses)",          "importe": 5400},
        {"nombre": "Entrada de coche",                    "importe": 3000},
    ]

    if microahorro_anual <= 0:
        for m in metas:
            m["meses"] = "∞"
            m["alcanzable"] = False
        return metas

    for meta in metas:
        meses_necesarios = meta["importe"] / (microahorro_anual / 12)
        meta["meses"]     = round(meses_necesarios, 1)
        meta["alcanzable"] = meses_necesarios <= 120  # máx 10 años

    return metas


def microahorro_por_categoria(
    df: pd.DataFrame,
    unidad_redondeo: float = 1.0,
) -> pd.DataFrame:
    """Micro-ahorro potencial agrupado por categoría."""
    df_micro = calcular_microahorro(df, unidad_redondeo)
    gastos = df_micro[df_micro["importe"] < 0]
    resultado = (
        gastos.groupby("categoria")["microahorro"]
        .agg(total="sum", media="mean", count="count")
        .sort_values("total", ascending=False)
        .reset_index()
    )
    resultado["total"] = resultado["total"].round(2)
    resultado["media"] = resultado["media"].round(2)
    return resultado
