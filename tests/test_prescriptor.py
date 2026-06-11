# tests/test_prescriptor.py — Pruebas del motor prescriptivo (Fase 3)
"""
Pruebas de `modulos.prescriptor`.

Cubren: que el optimizador devuelve planes factibles (riesgo ≤ umbral), que el
ranking respeta el umbral de iliquidez, que los planes generan ahorro protegido
positivo, que la explicación se construye, y que un umbral más estricto no
produce planes que lo violen.
"""
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fuentes.loader import cargar_sinteticos
from modulos.prescriptor import optimizar_plan, evaluar_plan, _microahorro_por_unidad


@pytest.fixture(scope="module")
def df():
    return cargar_sinteticos(meses=24, ingreso_base=1800, perfil="medio", seed=42)


def test_optimizar_devuelve_planes(df):
    res = optimizar_plan(df, meses=12, umbral_iliquidez_max=0.15,
                         n_candidatos=120, n_sim_busqueda=800, n_sim_refino=2000)
    assert res["n_evaluados"] > 0
    assert isinstance(res["planes"], list)
    assert "baseline" in res


def test_planes_respetan_umbral(df):
    umbral = 0.10
    res = optimizar_plan(df, meses=12, umbral_iliquidez_max=umbral,
                         n_candidatos=150, n_sim_busqueda=800, n_sim_refino=2000)
    for plan in res["planes"]:
        assert plan["prob_iliquidez"] <= umbral + 1e-9
        assert plan["ahorro_protegido"] > 0
        assert isinstance(plan["explicacion"], str) and len(plan["explicacion"]) > 0


def test_ranking_ordenado_por_ahorro(df):
    res = optimizar_plan(df, meses=12, umbral_iliquidez_max=0.20,
                         n_candidatos=150, n_sim_busqueda=800, n_sim_refino=2000)
    ahorros = [p["ahorro_protegido"] for p in res["planes"]]
    assert ahorros == sorted(ahorros, reverse=True)


def test_evaluar_plan_individual(df):
    micro = _microahorro_por_unidad(df)
    plan = {"ahorro_programado": 50, "redondeo_unidad": 2,
            "recortes": {"Restaurantes y Ocio": -20}, "mes_evento": 1}
    res = evaluar_plan(df, plan, meses=12, n_sim=1500, micro_cache=micro)
    assert res["aparta_mensual"] >= 50
    assert 0.0 <= res["prob_iliquidez"] <= 1.0
    assert res["ahorro_protegido"] > 0


def test_microahorro_cache(df):
    micro = _microahorro_por_unidad(df)
    assert micro[0] == 0.0
    # A mayor unidad de redondeo, mayor (o igual) micro-ahorro medio.
    assert micro[5] >= micro[1] >= 0.0
