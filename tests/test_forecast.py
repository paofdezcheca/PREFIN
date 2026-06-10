# tests/test_forecast.py — Pruebas de la Fase 1 (previsión con incertidumbre)
"""
Pruebas del módulo `modulos.forecast`.

Se ejecutan sobre datos sintéticos reproducibles (seed=42). Cubren:
  * detección de flujos recurrentes (nómina + algún gasto fijo),
  * monotonía y no negatividad de la banda p10 ≤ p50 ≤ p90,
  * previsión a varios meses,
  * backtest walk-forward (métricas y cobertura bien formadas),
  * camino de respaldo (fallback) con histórico corto,
  * construcción de la figura.

Ejecutar:  pytest -q
"""
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fuentes.loader import cargar_sinteticos
from modulos.forecast import (
    PrevisorGasto, detectar_recurrentes, backtest_previsor, figura_prevision,
    serie_gasto_total_mensual,
)


@pytest.fixture(scope="module")
def df_largo():
    """Histórico sintético de 36 meses (perfil medio)."""
    return cargar_sinteticos(meses=36, ingreso_base=1800, perfil="medio", seed=42)


@pytest.fixture(scope="module")
def df_corto():
    """Histórico sintético de 4 meses (fuerza el camino de respaldo)."""
    return cargar_sinteticos(meses=4, ingreso_base=1800, perfil="medio", seed=42)


def test_detecta_nomina_recurrente(df_largo):
    recur = detectar_recurrentes(df_largo)
    assert not recur.empty, "Debería detectar flujos recurrentes en 36 meses"
    # La nómina es un ingreso recurrente mensual: debe aparecer como recurrente ingreso.
    assert recur["es_ingreso"].any(), "La nómina debería detectarse como ingreso recurrente"
    # Debe haber también algún gasto recurrente (alquiler / servicios / suscripción).
    assert (~recur["es_ingreso"]).any(), "Debería detectar al menos un gasto recurrente"


def test_recurrentes_periodicidad_mensual(df_largo):
    recur = detectar_recurrentes(df_largo)
    # Debe existir al menos un flujo genuinamente mensual (p. ej. el alquiler,
    # con una periodicidad mediana en torno a 30 días).
    periodos = recur["periodicidad_dias"].dropna()
    assert (periodos.between(25, 35)).any(), \
        "Debería detectarse algún compromiso mensual (~30 días) como el alquiler"


def test_banda_monotona_y_positiva(df_largo):
    prev = PrevisorGasto().fit(df_largo)
    banda = prev.predecir(meses_adelante=3)
    assert len(banda) == 3
    for _, fila in banda.iterrows():
        assert fila["p10"] <= fila["p50"] <= fila["p90"], "La banda debe ser monótona"
        assert fila["p10"] >= 0, "El gasto previsto no puede ser negativo"


def test_usa_modelo_con_historico_largo(df_largo):
    prev = PrevisorGasto().fit(df_largo)
    assert prev.usa_modelo_ is True
    assert prev.gasto_recurrente_ > 0


def test_fallback_con_historico_corto(df_corto):
    prev = PrevisorGasto().fit(df_corto)
    assert prev.usa_modelo_ is False, "Con 4 meses debe usar cuantiles empíricos"
    banda = prev.predecir(meses_adelante=2)
    # En modo respaldo la banda es constante entre horizontes.
    assert banda["p50"].nunique() == 1
    for _, fila in banda.iterrows():
        assert fila["p10"] <= fila["p50"] <= fila["p90"]


def test_backtest_metricas_bien_formadas(df_largo):
    m = backtest_previsor(df_largo, min_train=8)
    assert m["n_folds"] > 0
    assert m["mae"] is not None and m["mae"] >= 0
    assert m["rmse"] is not None and m["rmse"] >= m["mae"]  # RMSE ≥ MAE siempre
    assert 0.0 <= m["cobertura"] <= 1.0
    assert m["mae_baseline"] is not None


def test_banda_contiene_orden_de_magnitud_real(df_largo):
    # La mediana prevista debe estar en el mismo orden que el gasto real reciente.
    serie = serie_gasto_total_mensual(df_largo)
    gasto_medio = serie.tail(6).mean()
    p50 = PrevisorGasto().fit(df_largo).predecir(1).iloc[0]["p50"]
    assert 0.4 * gasto_medio <= p50 <= 2.0 * gasto_medio


def test_figura_se_construye(df_largo):
    fig = figura_prevision(df_largo, meses_adelante=6)
    # Debe tener al menos las trazas: real + 2 de banda + mediana.
    assert len(fig.data) >= 4
