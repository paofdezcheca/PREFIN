# tests/bench_rendimiento.py — Medición empírica de los SLOs de rendimiento
#
# Métrica 1: Monte Carlo 10 000 trayectorias × 36 meses  →  objetivo < X ms
# Métrica 2: Callbacks Dash principales                  →  objetivo < 1 s
#
# Uso:
#   pytest tests/bench_rendimiento.py --benchmark-only
#   pytest tests/bench_rendimiento.py --benchmark-only --benchmark-json=bench_results.json
#
# Para comparar runs sucesivos (detectar regresiones):
#   pytest tests/bench_rendimiento.py --benchmark-only --benchmark-compare

import os
import sys
import time

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fuentes.loader import cargar_sinteticos
from modulos.digital_twin import simular_montecarlo, estado_actual, simular_escenario


# ---------------------------------------------------------------------------
# Fixture compartido: datos sintéticos representativos (24 meses, perfil medio)
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def df():
    return cargar_sinteticos(meses=24, ingreso_base=1800, perfil="medio", seed=42)


# ---------------------------------------------------------------------------
# MÉTRICA 1 — Monte Carlo 10 000 trayectorias × 36 meses
# ---------------------------------------------------------------------------

def test_bench_montecarlo_10k_36m(benchmark, df):
    """
    SLO: la simulación completa debe terminar en < X ms.
    El valor de 'mean' en el informe es el número a poner en la especificación.
    """
    result = benchmark(simular_montecarlo, df, meses=36, n_sim=10_000, seed=1)
    assert result is not None, "simular_montecarlo devolvió None"
    assert "bandas" in result


def test_bench_montecarlo_5k_36m(benchmark, df):
    """Variante con 5 000 trayectorias para comparar escalado."""
    result = benchmark(simular_montecarlo, df, meses=36, n_sim=5_000, seed=1)
    assert result is not None


# ---------------------------------------------------------------------------
# MÉTRICA 2 — Tiempo de respuesta de las vistas principales (callbacks Dash)
#
# Dash ejecuta los callbacks en el servidor Flask; los medimos directamente
# llamando a las funciones de negocio que cada callback invoca, que es el
# trabajo real (el overhead HTTP en local es < 5 ms y no cuenta en producción
# porque se mide sobre el equipo de referencia).
# ---------------------------------------------------------------------------

def test_bench_estado_actual(benchmark, df):
    """Vista Resumen — carga del estado financiero actual."""
    result = benchmark(estado_actual, df)
    assert isinstance(result, dict)


def test_bench_simular_escenario(benchmark, df):
    """Vista Gemelo Digital — simulación de escenario a 36 meses."""
    estado = estado_actual(df)
    result = benchmark(simular_escenario, estado, meses=36)
    assert not result.empty


# ---------------------------------------------------------------------------
# MÉTRICA 2b — Benchmark de extremo a extremo vía HTTP (app arrancada aparte)
#
# Solo se ejecuta si la variable de entorno PREFIN_URL está definida.
# Ejemplo:
#   $env:PREFIN_URL = "http://localhost:8050"
#   pytest tests/bench_rendimiento.py -k http --benchmark-only
# ---------------------------------------------------------------------------

@pytest.mark.skipif(
    not os.environ.get("PREFIN_URL"),
    reason="Define PREFIN_URL=http://localhost:8050 para medir tiempos HTTP reales",
)
def test_bench_http_pagina_principal(benchmark):
    """Tiempo de carga de la página principal vía HTTP."""
    import requests
    url = os.environ["PREFIN_URL"]

    def cargar():
        r = requests.get(url, timeout=10)
        assert r.status_code == 200
        return r

    benchmark(cargar)
