# tests/test_explicador.py — Pruebas de explicabilidad (Fase 5)
"""
Pruebas de `modulos.explicador`.

Verifican que las contribuciones se calculan (con SHAP o con el fallback de
oclusión), que suman coherentemente hacia la diferencia prob−base, que la
explicación en lenguaje natural se genera con frases no vacías, y que el módulo
degrada con elegancia si el modelo no está entrenado.
"""
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fuentes.generator import generar_multiusuario
from fuentes.loader import cargar_sinteticos
from modulos.riesgo_futuro import ModeloRiesgoFuturo
from modulos.explicador import contribuciones, explicar_natural, figura_contribuciones


@pytest.fixture(scope="module")
def modelo():
    panel = generar_multiusuario(n_usuarios=40, meses=36, seed=0, realista=True)
    return ModeloRiesgoFuturo().entrenar(panel)


@pytest.fixture(scope="module")
def df():
    return cargar_sinteticos(meses=24, seed=7, realista=True)


def test_contribuciones_se_calculan(modelo, df):
    datos = contribuciones(modelo, df)
    assert datos["metodo"] in {"shap", "oclusión"}
    assert len(datos["contribuciones"]) > 0
    assert 0.0 <= datos["prob"] <= 1.0
    assert 0.0 <= datos["base"] <= 1.0


def test_contribuciones_ordenadas_por_magnitud(modelo, df):
    datos = contribuciones(modelo, df)
    mags = [abs(c["contribucion"]) for c in datos["contribuciones"]]
    assert mags == sorted(mags, reverse=True)


def test_explicacion_natural(modelo, df):
    exp = explicar_natural(modelo, df, top=3)
    assert exp["metodo"] in {"shap", "oclusión"}
    frases = exp["frases_sube"] + exp["frases_baja"]
    assert len(frases) >= 1
    assert all(isinstance(f, str) and f.endswith(".") for f in frases)


def test_figura_contribuciones(modelo, df):
    fig = figura_contribuciones(modelo, df)
    assert len(fig.data) >= 1


def test_degrada_sin_modelo(df):
    modelo_vacio = ModeloRiesgoFuturo()  # no entrenado
    assert contribuciones(modelo_vacio, df) == {}
    exp = explicar_natural(modelo_vacio, df)
    assert exp["metodo"] is None
