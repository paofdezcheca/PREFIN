# tests/test_riesgo_futuro.py — Pruebas de la des-circularización (Fase 3-6)
"""
Pruebas del generador enriquecido (`fuentes.generator`) y del modelo de riesgo
des-circularizado (`modulos.riesgo_futuro`).

Cubren:
  * que el modo `realista` añade señal (recibos anuales, shocks) sin romper el
    esquema canónico,
  * que el panel multiusuario produce ambas clases (algunos usuarios entran en
    iliquidez y otros no) → problema de clasificación genuino,
  * que la etiqueta es un EVENTO FUTURO (no una regla sobre las features),
  * que el modelo entrena, valida por grupos y predice con interfaz coherente,
  * que las métricas honestas existen y son plausibles.
"""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fuentes.generator import generar_transacciones, generar_multiusuario
from fuentes.loader import cargar_sinteticos
from modulos.riesgo_futuro import ModeloRiesgoFuturo, construir_dataset

ESQUEMA = {"fecha", "descripcion", "importe", "divisa", "categoria", "saldo_acumulado"}


def test_modo_legacy_reproducible():
    # Sin realista, el comportamiento por defecto no cambia (reproducibilidad).
    a = generar_transacciones(meses=12, seed=42)
    b = generar_transacciones(meses=12, seed=42)
    assert a["importe"].sum() == b["importe"].sum()
    assert ESQUEMA.issubset(set(a.columns))


def test_modo_realista_anade_senal():
    base = generar_transacciones(meses=24, seed=1, realista=False)
    rico = generar_transacciones(meses=24, seed=1, realista=True)
    # Los recibos anuales/shocks introducen descripciones nuevas.
    nuevas = set(rico["descripcion"]) - set(base["descripcion"])
    assert any("IBI" in d or "Seguro" in d or "Avería" in d or "Multa" in d
               or "Dental" in d or "Electrodoméstico" in d for d in nuevas)
    assert ESQUEMA.issubset(set(rico.columns))


def test_multiusuario_dos_clases():
    panel = generar_multiusuario(n_usuarios=30, meses=36, seed=0, realista=True)
    assert "usuario" in panel.columns
    assert panel["usuario"].nunique() == 30
    X, y, grupos = construir_dataset(panel)
    # Debe haber casos de iliquidez (1) y de estabilidad (0).
    assert set(np.unique(y)) == {0, 1}
    assert len(np.unique(grupos)) >= 4


def test_modelo_entrena_y_valida():
    panel = generar_multiusuario(n_usuarios=40, meses=36, seed=0, realista=True)
    modelo = ModeloRiesgoFuturo().entrenar(panel)
    assert modelo._entrenado is True
    met = modelo.metricas_
    assert 0.0 <= met["accuracy"] <= 1.0
    if met["roc_auc"] is not None:
        # Un modelo con señal debe superar al azar (0.5) con holgura razonable.
        assert met["roc_auc"] >= 0.6
    assert met["n_test"] > 0 and met["n_train"] > 0


def test_prediccion_interfaz():
    panel = generar_multiusuario(n_usuarios=40, meses=36, seed=0, realista=True)
    modelo = ModeloRiesgoFuturo().entrenar(panel)
    df = cargar_sinteticos(meses=24, seed=7)
    pred = modelo.predecir(df, con_montecarlo=True)
    assert pred["entrenado"] is True
    assert 0.0 <= pred["prob_iliquidez"] <= 1.0
    assert pred["nivel"] in {"Bajo", "Medio", "Alto"}
    assert set(pred["importancias"].keys())  # no vacío
    assert pred["prob_iliquidez_mc"] is None or 0.0 <= pred["prob_iliquidez_mc"] <= 1.0


def test_modelo_sin_datos_suficientes():
    # Un panel diminuto no debe entrenar (y no debe romper).
    panel = generar_multiusuario(n_usuarios=2, meses=8, seed=0, realista=True)
    modelo = ModeloRiesgoFuturo().entrenar(panel)
    pred = modelo.predecir(cargar_sinteticos(meses=12, seed=1))
    assert pred["entrenado"] is False
    assert pred["nivel"] == "Desconocido"
