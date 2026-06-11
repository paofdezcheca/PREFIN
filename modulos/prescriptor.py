# modulos/prescriptor.py — Motor prescriptivo (Fase 3)
"""
Da el salto de PREDECIR a PRESCRIBIR: en lugar de que el usuario mueva sliders a
mano, el motor **busca el mejor plan de acción** usando el gemelo de Monte Carlo
(Fase 2) como entorno de evaluación.

Pregunta que responde, en lenguaje llano:
    «¿Cuánto puedo apartar cada mes —combinando recortes en gastos discrecionales
     y redondeo de compras— sin que mi riesgo de quedarme sin dinero supere un
     umbral?»

Palancas de acción
------------------
  * `ahorro_programado` : importe fijo que se aparta cada mes (€).
  * `redondeo_unidad`   : agresividad del redondeo (0, 1, 2, 5 €); su micro-ahorro
                          medio mensual se calcula de los datos reales.
  * `recortes`          : % de recorte en categorías discrecionales.
  * `mes_evento`        : si hay un gasto puntual previsto, cuándo afrontarlo.

Objetivo y restricción
----------------------
Se MAXIMIZA el **ahorro mensual protegido** (lo que se aparta a una hucha: ahorro
programado + redondeo), SUJETO A que la probabilidad de iliquidez en el horizonte
sea ≤ `umbral`. Los recortes liberan liquidez en la cuenta, lo que permite apartar
más sin incumplir la restricción: por eso el motor los explora.

Método
------
Búsqueda aleatoria sobre el espacio de palancas (transparente y defendible). Cada
plan candidato se evalúa con una simulación de Monte Carlo de baja resolución; los
mejores planes factibles se reevalúan con alta resolución. Se devuelve un ranking
explicado con impacto cuantificado frente al escenario «sin plan».

Dependencias: numpy / pandas (ya presentes) + el gemelo MC. Sin librerías nuevas.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from modulos.digital_twin import simular_montecarlo, calibrar_perfiles
from modulos.microsavings import resumen_microahorro

# Categorías sobre las que es razonable proponer recortes (gasto discrecional).
# No se recorta vivienda, suministros, salud ni transporte esencial.
CATEGORIAS_DISCRECIONALES = ["Restaurantes y Ocio", "Ropa y Compras", "Suscripciones"]

# Rejilla de valores por palanca.
_AHORROS = [0, 25, 50, 75, 100, 150, 200]
_REDONDEOS = [0, 1, 2, 5]
_RECORTES = [0, -10, -20, -30]


def _microahorro_por_unidad(df: pd.DataFrame) -> dict:
    """Micro-ahorro medio mensual para cada unidad de redondeo (cacheado)."""
    cache = {0: 0.0}
    for u in (1, 2, 5):
        try:
            cache[u] = float(resumen_microahorro(df, unidad_redondeo=float(u))["mensual_medio"])
        except Exception:
            cache[u] = 0.0
    return cache


def _esfuerzo(plan: dict, ingreso_mensual: float) -> float:
    """Mide el 'esfuerzo/incomodidad' de un plan en [0, 1] aprox.

    Combina la magnitud de los recortes y la fracción de ingreso que se aparta.
    Sirve para, a igualdad de ahorro, preferir el plan menos doloroso.
    """
    recorte = sum(abs(p) for p in plan["recortes"].values()) / 100.0
    aparta = (plan["ahorro_programado"]) / ingreso_mensual if ingreso_mensual else 0.0
    redondeo = {0: 0, 1: 0.05, 2: 0.1, 5: 0.2}.get(plan["redondeo_unidad"], 0)
    return round(recorte * 0.5 + aparta + redondeo, 3)


def evaluar_plan(df: pd.DataFrame, plan: dict, meses: int, n_sim: int,
                 micro_cache: dict, gasto_puntual: float = 0.0,
                 umbral_iliquidez: float = 0.0, seed: int = 42) -> dict:
    """Evalúa un plan con el gemelo MC y devuelve sus métricas.

    El ahorro programado y el redondeo se suman como dinero apartado cada mes
    (reduce la liquidez de la cuenta y, por tanto, sube el riesgo, que es lo que
    la restricción controla).
    """
    redondeo_mensual = micro_cache.get(plan["redondeo_unidad"], 0.0)
    aparta_mensual = plan["ahorro_programado"] + redondeo_mensual

    mc = simular_montecarlo(
        df, meses=meses, n_sim=n_sim,
        cambios_categoria=plan["recortes"],
        ahorro_extra_mensual=aparta_mensual,
        evento_imprevisto=gasto_puntual,
        mes_evento=plan.get("mes_evento", 1),
        umbral_iliquidez=umbral_iliquidez,
        seed=seed,
    )
    if not mc:
        return {}

    ahorro_protegido = round(aparta_mensual * meses, 2)
    return {
        **plan,
        "aparta_mensual": round(aparta_mensual, 2),
        "redondeo_mensual": round(redondeo_mensual, 2),
        "ahorro_protegido": ahorro_protegido,
        "prob_iliquidez": mc["prob_iliquidez_horizonte"],
        "var_95": mc["var_95"],
        "saldo_final_esperado": mc["saldo_final_esperado"],
    }


def optimizar_plan(
    df: pd.DataFrame,
    meses: int = 12,
    umbral_iliquidez_max: float = 0.10,
    gasto_puntual: float = 0.0,
    n_candidatos: int = 300,
    n_sim_busqueda: int = 1500,
    n_sim_refino: int = 5000,
    top_k: int = 5,
    seed: int = 42,
) -> dict:
    """Busca los mejores planes de acción y los devuelve ranqueados y explicados.

    Devuelve un diccionario con:
      * `baseline`  : métricas del escenario «sin plan» (no hacer nada),
      * `planes`    : lista de hasta `top_k` planes factibles, ordenados por
                      ahorro protegido (desc.) y, a igualdad, por menor esfuerzo,
      * `umbral`, `meses`, `n_evaluados`.
    """
    perfiles = calibrar_perfiles(df)
    if perfiles["n_meses"] == 0:
        return {"baseline": {}, "planes": [], "umbral": umbral_iliquidez_max,
                "meses": meses, "n_evaluados": 0}

    ingreso_mensual = perfiles["ingreso"][0]
    micro_cache = _microahorro_por_unidad(df)
    cats = [c for c in CATEGORIAS_DISCRECIONALES if c in perfiles["categorias"]]
    rng = np.random.default_rng(seed)

    # --- Baseline: no hacer nada ---
    plan_nulo = {"ahorro_programado": 0, "redondeo_unidad": 0,
                 "recortes": {}, "mes_evento": 1}
    baseline = evaluar_plan(df, plan_nulo, meses, n_sim_refino, micro_cache,
                            gasto_puntual, seed=seed)

    # --- Búsqueda aleatoria de candidatos ---
    vistos = set()
    candidatos = []
    for _ in range(n_candidatos):
        recortes = {c: int(rng.choice(_RECORTES)) for c in cats}
        recortes = {c: v for c, v in recortes.items() if v != 0}
        plan = {
            "ahorro_programado": int(rng.choice(_AHORROS)),
            "redondeo_unidad": int(rng.choice(_REDONDEOS)),
            "recortes": recortes,
            "mes_evento": int(rng.integers(1, meses + 1)) if gasto_puntual else 1,
        }
        clave = (plan["ahorro_programado"], plan["redondeo_unidad"],
                 tuple(sorted(recortes.items())), plan["mes_evento"])
        if clave in vistos:
            continue
        vistos.add(clave)
        candidatos.append(plan)

    # Evaluación de baja resolución + filtro de factibilidad.
    evaluados = []
    for plan in candidatos:
        res = evaluar_plan(df, plan, meses, n_sim_busqueda, micro_cache,
                           gasto_puntual, seed=seed)
        if res and res["ahorro_protegido"] > 0 and \
                res["prob_iliquidez"] <= umbral_iliquidez_max:
            res["esfuerzo"] = _esfuerzo(plan, ingreso_mensual)
            evaluados.append(res)

    # Ranking: más ahorro protegido; a igualdad, menos esfuerzo.
    evaluados.sort(key=lambda r: (-r["ahorro_protegido"], r["esfuerzo"]))

    # Refino de alta resolución de los mejores (revalida la factibilidad).
    finalistas = []
    for res in evaluados[:top_k * 2]:
        plan = {k: res[k] for k in ("ahorro_programado", "redondeo_unidad",
                                    "recortes", "mes_evento")}
        ref = evaluar_plan(df, plan, meses, n_sim_refino, micro_cache,
                           gasto_puntual, seed=seed)
        if ref and ref["prob_iliquidez"] <= umbral_iliquidez_max:
            ref["esfuerzo"] = _esfuerzo(plan, ingreso_mensual)
            ref["explicacion"] = _explicar(ref, baseline, meses)
            finalistas.append(ref)

    finalistas.sort(key=lambda r: (-r["ahorro_protegido"], r["esfuerzo"]))
    return {
        "baseline": baseline,
        "planes": finalistas[:top_k],
        "umbral": umbral_iliquidez_max,
        "meses": meses,
        "n_evaluados": len(candidatos),
    }


def _explicar(plan: dict, baseline: dict, meses: int) -> str:
    """Genera una explicación en lenguaje llano del plan, con impacto cuantificado."""
    partes = []
    if plan["ahorro_programado"] > 0:
        partes.append(f"apartar {plan['ahorro_programado']:.0f} €/mes")
    if plan["redondeo_unidad"] > 0:
        partes.append(f"redondear tus compras al múltiplo de {plan['redondeo_unidad']} € "
                      f"(≈{plan['redondeo_mensual']:.0f} €/mes)")
    for cat, pct in plan["recortes"].items():
        partes.append(f"recortar un {abs(pct):.0f}% en {cat}")

    accion = "; ".join(partes) if partes else "sin cambios"
    riesgo_base = baseline.get("prob_iliquidez", 0) if baseline else 0
    frase = (f"Si decides {accion}, en {meses} meses acumularías "
             f"≈ {plan['ahorro_protegido']:.0f} € de ahorro protegido, "
             f"manteniendo el riesgo de quedarte sin dinero en un "
             f"{plan['prob_iliquidez']:.0%}")
    if riesgo_base:
        frase += f" (frente al {riesgo_base:.0%} actual)"
    frase += "."
    return frase
