# tests/test_validacion.py — Pruebas del rigor de validación (Fase 6)
"""
Pruebas de `modulos.validacion`. Tamaños acotados para que la suite sea rápida.

Cubren: que el backtest de forecast devuelve métricas y referencia, que la
validación cruzada del clasificador da AUC/Brier y curva de calibración, que el
backtesting del gemelo se ejecuta y degrada con elegancia, y que la figura de
calibración se construye.
"""
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fuentes.generator import generar_multiusuario
from fuentes.loader import cargar_sinteticos
from modulos.validacion import (
    validar_forecast, validar_clasificador, backtest_prob_iliquidez,
    figura_calibracion,
)


@pytest.fixture(scope="module")
def panel():
    return generar_multiusuario(n_usuarios=20, meses=30, seed=0, realista=True)


def test_validar_forecast():
    df = cargar_sinteticos(meses=30, seed=1, realista=True)
    m = validar_forecast(df)
    assert m["n_folds"] > 0
    assert m["mae"] is not None and m["rmse"] >= m["mae"]
    assert m["mae_baseline"] is not None
    assert 0.0 <= m["cobertura"] <= 1.0


def test_validar_clasificador(panel):
    res = validar_clasificador(panel, n_splits=4)
    assert res["ok"] is True
    if res["auc_media"] is not None:
        assert 0.0 <= res["auc_media"] <= 1.0
    assert 0.0 <= res["brier"] <= 1.0
    assert len(res["calibracion"]["prob_real"]) >= 2


def test_backtest_prob_iliquidez(panel):
    res = backtest_prob_iliquidez(panel, n_sim=600, max_puntos=40)
    assert "ok" in res
    if res["ok"]:
        assert res["n_puntos"] > 0
        assert 0.0 <= res["brier"] <= 1.0


def test_figura_calibracion():
    cal = {"prob_real": [0.1, 0.5, 0.9], "prob_predicha": [0.2, 0.5, 0.8]}
    fig = figura_calibracion(cal, "Test")
    assert len(fig.data) >= 2  # diagonal + modelo


def test_clasificador_datos_insuficientes():
    panel_min = generar_multiusuario(n_usuarios=3, meses=10, seed=0, realista=True)
    res = validar_clasificador(panel_min, n_splits=4)
    assert res["ok"] is False
