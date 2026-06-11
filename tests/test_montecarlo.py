# tests/test_montecarlo.py — Pruebas del gemelo Monte Carlo (Fase 2)
"""
Pruebas del gemelo digital estocástico (`modulos.digital_twin`).

Cubren: calibración de perfiles, forma y monotonía de las bandas, coherencia de
P(iliquidez) y de las métricas de cola (VaR ≤ CVaR), reproducibilidad por semilla
y respuesta esperada a las palancas (más ingreso ⇒ mejor saldo; meta de ahorro
⇒ no empeora la liquidez).
"""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fuentes.loader import cargar_sinteticos
from modulos.digital_twin import (
    calibrar_perfiles, simular_montecarlo, figura_cono_montecarlo,
)


@pytest.fixture(scope="module")
def df():
    return cargar_sinteticos(meses=24, ingreso_base=1800, perfil="medio", seed=42)


def test_calibrar_perfiles(df):
    p = calibrar_perfiles(df)
    assert p["n_meses"] == 24
    assert p["ingreso"][0] > 0          # media de ingreso positiva
    assert len(p["categorias"]) > 0     # al menos una categoría de gasto


def test_bandas_monotonas_y_completas(df):
    mc = simular_montecarlo(df, meses=12, n_sim=2000, seed=1)
    bandas = mc["bandas"]
    assert len(bandas) == 12
    assert (bandas["p10"] <= bandas["p50"]).all()
    assert (bandas["p50"] <= bandas["p90"]).all()
    assert (bandas["prob_iliquidez"].between(0, 1)).all()


def test_var_cvar_coherentes(df):
    mc = simular_montecarlo(df, meses=12, n_sim=3000, seed=1)
    # CVaR (media de la cola) es siempre ≤ VaR (percentil), por definición.
    assert mc["cvar_95"] <= mc["var_95"]
    assert 0.0 <= mc["prob_iliquidez_horizonte"] <= 1.0


def test_reproducibilidad_semilla(df):
    a = simular_montecarlo(df, meses=12, n_sim=2000, seed=7)
    b = simular_montecarlo(df, meses=12, n_sim=2000, seed=7)
    assert a["saldo_final_esperado"] == b["saldo_final_esperado"]
    assert a["var_95"] == b["var_95"]


def test_mas_ingreso_mejora_saldo(df):
    base = simular_montecarlo(df, meses=12, n_sim=3000, seed=3)
    mas = simular_montecarlo(df, meses=12, n_sim=3000, seed=3,
                             cambio_ingreso_pct=20)
    assert mas["saldo_final_esperado"] > base["saldo_final_esperado"]


def test_recorte_categoria_reduce_gasto(df):
    base = simular_montecarlo(df, meses=12, n_sim=3000, seed=4)
    cat = max(calibrar_perfiles(df)["categorias"],
              key=lambda c: calibrar_perfiles(df)["categorias"][c][0])
    recorte = simular_montecarlo(df, meses=12, n_sim=3000, seed=4,
                                 cambios_categoria={cat: -50})
    assert recorte["saldo_final_esperado"] >= base["saldo_final_esperado"]


def test_evento_imprevisto_empeora(df):
    base = simular_montecarlo(df, meses=12, n_sim=3000, seed=5)
    shock = simular_montecarlo(df, meses=12, n_sim=3000, seed=5,
                               evento_imprevisto=3000, mes_evento=1)
    assert shock["saldo_final_esperado"] < base["saldo_final_esperado"]
    assert shock["prob_iliquidez_horizonte"] >= base["prob_iliquidez_horizonte"]


def test_figura_cono_se_construye(df):
    mc = simular_montecarlo(df, meses=12, n_sim=1000, seed=6)
    fig = figura_cono_montecarlo(mc)
    assert len(fig.data) > 3  # trayectorias + banda + mediana
