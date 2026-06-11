# modulos/explicador.py — Explicabilidad del riesgo (Fase 5)
"""
Explica POR QUÉ el modelo de riesgo des-circularizado predice lo que predice,
y lo traduce a lenguaje natural fundamentado en los datos del usuario.

Método principal: **SHAP** (`shap.TreeExplainer`, exacto y rápido para modelos de
árbol). SHAP reparte la diferencia entre la predicción del usuario y la predicción
media de la población entre las variables, de forma teóricamente fundamentada
(valores de Shapley). Como la explicación solo es útil sobre un modelo que
APRENDE algo no trivial, se aplica sobre el modelo ya des-circularizado.

Fallback (si `shap` no está instalado): contribuciones por **oclusión** — cuánto
cambia la probabilidad al sustituir cada variable por su valor poblacional medio.
Es model-agnostic, interpretable y no requiere dependencias extra. La interfaz de
salida es idéntica, de modo que el resto de la app no depende del backend.

La traducción a lenguaje natural usa el VALOR REAL del usuario en cada variable
(p. ej. «gastas el 92% de lo que ingresas»), no jerga de modelos.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from modulos.ml_model import extraer_features, FEATURES

# ¿Está SHAP disponible? Se decide una vez al importar.
try:  # pragma: no cover - depende del entorno
    import shap  # noqa: F401
    _HAY_SHAP = True
except Exception:  # pragma: no cover
    _HAY_SHAP = False


# Cómo redactar cada variable: (nombre legible, descripción FACTUAL del valor).
# La descripción depende solo del VALOR del usuario (un hecho); el EFECTO sobre el
# riesgo (eleva/reduce) lo aporta aparte el signo de la contribución, para evitar
# frases contradictorias como «tu gasto baja (+257 €/mes)».
def _pct(v):
    return f"{v*100:.0f}%"


_EXPLICA = {
    "ratio_gasto_ingreso": (
        "proporción de gasto sobre ingresos",
        lambda v: f"gastas {_pct(v)} de lo que ingresas"),
    "ratio_ahorro": (
        "tasa de ahorro",
        lambda v: f"tu tasa de ahorro es de {_pct(v)}"),
    "gasto_ocio_ratio": (
        "gasto en ocio sobre ingresos",
        lambda v: f"destinas {_pct(v)} de tu ingreso a ocio"),
    "variabilidad_gasto": (
        "variabilidad del gasto",
        lambda v: f"tus gastos varían ±{v:,.0f} € entre meses"),
    "tendencia_gasto": (
        "tendencia del gasto",
        lambda v: (f"tu gasto viene creciendo (+{v:,.0f} €/mes)" if v > 0
                   else f"tu gasto viene bajando ({v:,.0f} €/mes)")),
    "ingreso_mensual": (
        "ingreso mensual",
        lambda v: f"tu ingreso mensual es de {v:,.0f} €"),
    "gasto_total": (
        "gasto total mensual",
        lambda v: f"tu gasto total es de {v:,.0f} €/mes"),
    "gasto_suscripciones": (
        "gasto en suscripciones",
        lambda v: f"gastas {v:,.0f} €/mes en suscripciones"),
    "gasto_supermercado": (
        "gasto en supermercado",
        lambda v: f"tu gasto en supermercado es de {v:,.0f} €/mes"),
    "n_categorias_activas": (
        "número de categorías de gasto",
        lambda v: f"gastas en {v:.0f} categorías distintas"),
    "colchon_meses": (
        "colchón de liquidez",
        lambda v: (f"tu colchón cubre {v:.1f} meses de gasto" if v >= 1
                   else f"tu colchón apenas cubre {v*30:.0f} días de gasto")),
}


def contribuciones(modelo, df: pd.DataFrame) -> dict:
    """Calcula las contribuciones de cada variable a la predicción del usuario.

    Devuelve un diccionario con:
      * `metodo` : "shap" u "oclusión",
      * `base`   : probabilidad de referencia (media poblacional),
      * `prob`   : probabilidad predicha para el usuario,
      * `contribuciones` : lista [{feature, contribucion, valor}] ordenada por
                           magnitud descendente (contribución a la prob. de riesgo).
    """
    if not getattr(modelo, "_entrenado", False) or modelo.X_ref_ is None:
        return {}

    feats = extraer_features(df)
    if feats.empty:
        return {}
    x = feats[FEATURES].iloc[-1].fillna(0).values.astype(float)
    valores = dict(zip(FEATURES, x))

    rf = modelo.clf.named_steps["rf"]
    scaler = modelo.clf.named_steps["scaler"]
    prob = float(modelo.clf.predict_proba(x.reshape(1, -1))[0, 1])
    base = float(modelo.clf.predict_proba(modelo.X_ref_.reshape(1, -1))[0, 1])

    if _HAY_SHAP:
        contribs, metodo = _contribuciones_shap(rf, scaler, x, modelo.X_ref_)
    else:
        contribs, metodo = _contribuciones_oclusion(modelo.clf, x, modelo.X_ref_, prob)

    lista = sorted(
        [{"feature": f, "contribucion": float(c), "valor": valores[f]}
         for f, c in zip(FEATURES, contribs)],
        key=lambda d: -abs(d["contribucion"]),
    )
    return {"metodo": metodo, "base": round(base, 4), "prob": round(prob, 4),
            "contribuciones": lista}


def _contribuciones_shap(rf, scaler, x, x_ref):
    """Valores SHAP para la clase 'en riesgo' (probabilidad)."""
    import shap
    x_s = scaler.transform(x.reshape(1, -1))
    ref_s = scaler.transform(x_ref.reshape(1, -1))
    explainer = shap.TreeExplainer(rf, data=ref_s, model_output="probability")
    sv = explainer.shap_values(x_s)
    # Compatibilidad entre versiones de SHAP (lista por clase o array 3D).
    arr = np.asarray(sv)
    if isinstance(sv, list):
        vals = np.asarray(sv[1])[0]
    elif arr.ndim == 3:
        vals = arr[0, :, 1]
    else:
        vals = arr[0]
    return np.asarray(vals, dtype=float), "shap"


def _contribuciones_oclusion(clf, x, x_ref, prob):
    """Contribución por oclusión: prob actual − prob con la variable en su media."""
    contribs = []
    for j in range(len(x)):
        x_mod = x.copy()
        x_mod[j] = x_ref[j]
        prob_mod = float(clf.predict_proba(x_mod.reshape(1, -1))[0, 1])
        contribs.append(prob - prob_mod)
    return np.asarray(contribs, dtype=float), "oclusión"


def explicar_natural(modelo, df: pd.DataFrame, top: int = 3) -> dict:
    """Explicación en lenguaje natural de la predicción de riesgo.

    Devuelve `{metodo, prob, frases_sube, frases_baja, contribuciones}`, donde las
    frases están redactadas con el valor real del usuario y ordenadas por impacto.
    """
    datos = contribuciones(modelo, df)
    if not datos:
        return {"metodo": None, "prob": None, "frases_sube": [], "frases_baja": [],
                "contribuciones": []}

    sube, baja = [], []
    for item in datos["contribuciones"]:
        frase = _frase(item["feature"], item["valor"], item["contribucion"])
        if frase is None:
            continue
        (sube if item["contribucion"] > 0 else baja).append(frase)

    return {
        "metodo": datos["metodo"],
        "prob": datos["prob"],
        "frases_sube": sube[:top],
        "frases_baja": baja[:top],
        "contribuciones": datos["contribuciones"],
    }


def _frase(feature: str, valor: float, contribucion: float) -> str | None:
    """Redacta una frase: hecho (según el valor) + efecto (según la contribución)."""
    if feature not in _EXPLICA or abs(contribucion) < 1e-4:
        return None
    _, describe = _EXPLICA[feature]
    cuerpo = describe(valor)
    efecto = "eleva tu riesgo" if contribucion > 0 else "reduce tu riesgo"
    return f"{cuerpo[0].upper()}{cuerpo[1:]}, lo que {efecto}."


def figura_contribuciones(modelo, df: pd.DataFrame, top: int = 6):
    """Gráfico de barras de las contribuciones (estilo waterfall horizontal)."""
    import plotly.graph_objects as go
    try:
        from config import PLOTLY_LAYOUT, PREFIN_ROJO, PREFIN_VERDE
    except Exception:  # pragma: no cover
        PLOTLY_LAYOUT, PREFIN_ROJO, PREFIN_VERDE = {}, "#DC2626", "#16A34A"

    datos = contribuciones(modelo, df)
    if not datos:
        return go.Figure()

    items = datos["contribuciones"][:top][::-1]
    nombres = [_EXPLICA.get(i["feature"], (i["feature"],))[0] for i in items]
    valores = [i["contribucion"] for i in items]
    colores = [PREFIN_ROJO if v > 0 else PREFIN_VERDE for v in valores]

    fig = go.Figure(go.Bar(x=valores, y=nombres, orientation="h",
                           marker_color=colores))
    fig.update_layout(**PLOTLY_LAYOUT)
    fig.update_layout(
        title_text=f"Por qué este nivel de riesgo · {datos['metodo']}",
        xaxis_title="Impacto en la probabilidad de iliquidez (← reduce | eleva →)")
    return fig
