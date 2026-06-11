# tests/test_deteccion.py — Pruebas de prevención (Fase 4)
"""
Pruebas de `modulos.deteccion`: cambios de régimen e IsolationForest.

Cubren: que un cambio de nivel evidente se detecta en el sitio correcto, que una
serie estable no genera cambios espurios, que el IsolationForest marca un importe
claramente atípico, y que la comparación con el z-score devuelve conteos válidos.
"""
import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fuentes.loader import cargar_sinteticos
from modulos.deteccion import (
    detectar_cambios_regimen, detectar_anomalias_isolation, comparar_detectores,
)


def _df_mensual(importes):
    """DataFrame de un gasto por mes a partir de una lista de importes (positivos)."""
    fechas = pd.date_range("2023-01-15", periods=len(importes), freq="MS")
    df = pd.DataFrame({
        "fecha": fechas,
        "descripcion": ["Gasto"] * len(importes),
        "importe": [-abs(x) for x in importes],
        "divisa": "EUR",
        "categoria": ["Otros"] * len(importes),
    })
    df["saldo_acumulado"] = df["importe"].cumsum() + 5000
    return df


def test_detecta_cambio_de_nivel():
    rng = np.random.default_rng(0)
    bajo = 500 + rng.normal(0, 15, 12)
    alto = 1500 + rng.normal(0, 15, 12)
    df = _df_mensual(list(bajo) + list(alto))
    cambios = detectar_cambios_regimen(df, min_size=3)
    assert not cambios.empty
    # El cambio principal debe estar cerca del mes 12 y ser una subida.
    indices = cambios["indice"].tolist()
    assert any(abs(i - 12) <= 1 for i in indices)
    assert (cambios["direccion"] == "subida").any()


def test_serie_estable_sin_cambios_espurios():
    rng = np.random.default_rng(1)
    estable = 800 + rng.normal(0, 20, 24)
    df = _df_mensual(list(estable))
    cambios = detectar_cambios_regimen(df, min_size=3)
    # Con ruido pequeño no debería trocear la serie (a lo sumo algún corte).
    assert len(cambios) <= 1


def test_isolation_marca_importe_atipico():
    df = cargar_sinteticos(meses=12, seed=3, realista=True)
    # Inyectar un gasto claramente atípico.
    fila = df.iloc[[0]].copy()
    fila["importe"] = -9999.0
    fila["descripcion"] = "Gasto Extraordinario"
    df2 = pd.concat([df, fila], ignore_index=True)
    out = detectar_anomalias_isolation(df2)
    assert out["anomalia_if"].sum() >= 1
    # La transacción de 9999 € debe quedar marcada.
    assert out.loc[out["importe"] == -9999.0, "anomalia_if"].any()


def test_comparar_detectores():
    df = cargar_sinteticos(meses=24, seed=4, realista=True)
    comp = comparar_detectores(df)
    for k in ("n_zscore", "n_isolation", "n_comunes", "solo_zscore", "solo_isolation"):
        assert k in comp and comp[k] >= 0
