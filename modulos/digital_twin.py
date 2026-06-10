# modules/digital_twin.py — Gemelo digital financiero y simulación de escenarios

from datetime import date

import numpy as np
import pandas as pd

from modulos.analyzer import resumen_mensual, gasto_por_categoria_mes


# ---------------------------------------------------------------------------
# Estado financiero actual (snapshot del gemelo)
# ---------------------------------------------------------------------------

def estado_actual(df: pd.DataFrame) -> dict:
    """
    Calcula el estado financiero actual del usuario a partir del último mes
    con datos suficientes.
    """
    resumen = resumen_mensual(df)
    cats    = gasto_por_categoria_mes(df)

    if resumen.empty:
        return {}

    # Promedio de los últimos 3 meses (o todos si hay menos)
    n = min(3, len(resumen))
    ultimos = resumen.tail(n)
    cats_ultimos = cats.tail(n) if len(cats) >= n else cats

    ingreso_medio = ultimos["ingreso_total"].mean()
    gasto_medio   = ultimos["gasto_total"].mean()
    ahorro_medio  = ultimos["ahorro_neto"].mean()

    # Gasto por categoría (media de últimos meses)
    cat_medias = {}
    for col in cats_ultimos.columns:
        if col != "mes":
            cat_medias[col] = round(cats_ultimos[col].mean(), 2)

    saldo = df["saldo_acumulado"].iloc[-1] if "saldo_acumulado" in df.columns else 0

    return {
        "saldo_actual":     round(saldo, 2),
        "ingreso_mensual":  round(ingreso_medio, 2),
        "gasto_mensual":    round(gasto_medio, 2),
        "ahorro_mensual":   round(ahorro_medio, 2),
        "tasa_ahorro":      round(ahorro_medio / ingreso_medio * 100 if ingreso_medio else 0, 1),
        "gasto_por_categoria": cat_medias,
    }


# ---------------------------------------------------------------------------
# Simulador de escenarios (gemelo digital)
# ---------------------------------------------------------------------------

def simular_escenario(
    estado: dict,
    meses: int = 12,
    cambio_ingreso_pct: float = 0.0,         # % de cambio en ingresos
    cambios_categoria: dict = None,           # {categoria: % de cambio}
    evento_imprevisto: float = 0.0,           # gasto puntual en mes 1
    meta_ahorro_mensual: float = None,        # objetivo de ahorro mensual
) -> pd.DataFrame:
    """
    Proyecta la situación financiera del gemelo digital durante `meses` meses.

    Parámetros
    ----------
    estado               : resultado de estado_actual()
    meses                : horizonte de simulación (máx 36)
    cambio_ingreso_pct   : variación porcentual del ingreso base (ej: 10 = +10%)
    cambios_categoria    : {categoría: % de cambio} en gastos (ej: {"Ocio": -20})
    evento_imprevisto    : gasto adicional único en el primer mes
    meta_ahorro_mensual  : si se establece, se descuenta del disponible cada mes

    Devuelve un DataFrame con columnas:
    mes, ingreso, gasto, ahorro, saldo_acumulado, escenario
    """
    if not estado:
        return pd.DataFrame()

    cambios_categoria = cambios_categoria or {}
    meses = min(meses, 36)

    ingreso_base  = estado["ingreso_mensual"] * (1 + cambio_ingreso_pct / 100)
    gasto_base    = estado["gasto_mensual"]
    saldo_inicial = estado["saldo_actual"]

    # Aplicar cambios por categoría al gasto base
    cats = estado.get("gasto_por_categoria", {})
    gasto_ajustado = gasto_base
    for cat, pct in cambios_categoria.items():
        if cat in cats:
            gasto_original = cats[cat]
            delta = gasto_original * (pct / 100)
            gasto_ajustado += delta

    filas_actual   = []
    filas_escenario = []
    saldo_act = saldo_inicial
    saldo_esc = saldo_inicial

    for i in range(meses):
        # --- Escenario actual (sin cambios) ---
        ingreso_a = estado["ingreso_mensual"]
        gasto_a   = gasto_base + (np.random.normal(0, gasto_base * 0.03))  # ruido leve
        ahorro_a  = ingreso_a - gasto_a
        saldo_act += ahorro_a
        filas_actual.append({
            "mes": i + 1,
            "ingreso":           round(ingreso_a, 2),
            "gasto":             round(abs(gasto_a), 2),
            "ahorro":            round(ahorro_a, 2),
            "saldo_acumulado":   round(saldo_act, 2),
            "escenario":         "Actual",
        })

        # --- Escenario simulado ---
        ingreso_s = ingreso_base
        gasto_s   = gasto_ajustado + (np.random.normal(0, gasto_ajustado * 0.03))
        if i == 0:
            gasto_s += evento_imprevisto
        if meta_ahorro_mensual:
            # Forzar ahorro mínimo mensual
            gasto_s = min(gasto_s, ingreso_s - meta_ahorro_mensual)
        ahorro_s  = ingreso_s - gasto_s
        saldo_esc += ahorro_s
        filas_escenario.append({
            "mes": i + 1,
            "ingreso":           round(ingreso_s, 2),
            "gasto":             round(abs(gasto_s), 2),
            "ahorro":            round(ahorro_s, 2),
            "saldo_acumulado":   round(saldo_esc, 2),
            "escenario":         "Simulado",
        })

    return pd.concat([
        pd.DataFrame(filas_actual),
        pd.DataFrame(filas_escenario),
    ], ignore_index=True)


def resumen_simulacion(sim_df: pd.DataFrame) -> dict:
    """Calcula el impacto del escenario simulado vs actual."""
    if sim_df.empty:
        return {}

    actual   = sim_df[sim_df["escenario"] == "Actual"]
    simulado = sim_df[sim_df["escenario"] == "Simulado"]

    saldo_final_act = actual["saldo_acumulado"].iloc[-1]
    saldo_final_sim = simulado["saldo_acumulado"].iloc[-1]
    delta_saldo     = saldo_final_sim - saldo_final_act

    ahorro_total_act = actual["ahorro"].sum()
    ahorro_total_sim = simulado["ahorro"].sum()

    return {
        "saldo_final_actual":   round(saldo_final_act, 2),
        "saldo_final_simulado": round(saldo_final_sim, 2),
        "diferencia_saldo":     round(delta_saldo, 2),
        "ahorro_total_actual":  round(ahorro_total_act, 2),
        "ahorro_total_simulado":round(ahorro_total_sim, 2),
        "mejora_ahorro":        round(ahorro_total_sim - ahorro_total_act, 2),
    }
