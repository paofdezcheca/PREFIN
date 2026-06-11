# modulos/deteccion.py — Prevención: cambios de régimen + anomalías (Fase 4)
"""
Prevención real, más allá del z-score por categoría del `analyzer`.

1.  **Detección de cambios de régimen** (`detectar_cambios_regimen`): localiza
    puntos en el tiempo donde el nivel del gasto mensual cambia de forma
    sostenida (p. ej. una subida del alquiler que se queda, una caída de
    ingresos, un nuevo gasto recurrente). Se implementa con **segmentación
    binaria** y una penalización tipo BIC: pura numpy, sin dependencias nuevas.
    A diferencia del z-score —que solo marca importes puntuales atípicos—, esto
    detecta DESPLAZAMIENTOS persistentes del comportamiento.

2.  **Anomalías no supervisadas** (`detectar_anomalias_isolation`): un
    `IsolationForest` (scikit-learn) que aprende el patrón de gasto del PROPIO
    usuario y marca las transacciones que se salen de él, considerando importe,
    día del mes, día de la semana y categoría a la vez (multivariante).

3.  **Comparación con el baseline** (`comparar_detectores`): contrasta el
    IsolationForest con el z-score por categoría del `analyzer`, para poder
    discutir en la memoria qué aporta cada uno.

Dependencias: numpy / pandas / scikit-learn (ya presentes). Sin librerías nuevas.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

from modulos.analyzer import detectar_anomalias as detectar_anomalias_zscore
from modulos.forecast import serie_gasto_total_mensual


# ---------------------------------------------------------------------------
# 1. Cambios de régimen (segmentación binaria)
# ---------------------------------------------------------------------------

def _sse(x: np.ndarray) -> float:
    """Suma de errores al cuadrado respecto a la media del segmento."""
    if len(x) == 0:
        return 0.0
    return float(np.sum((x - x.mean()) ** 2))


def _mejor_corte(x: np.ndarray, min_size: int):
    """Devuelve (indice, reduccion_sse) del mejor corte de la serie x."""
    n = len(x)
    base = _sse(x)
    mejor_idx, mejor_red = None, 0.0
    for t in range(min_size, n - min_size + 1):
        red = base - _sse(x[:t]) - _sse(x[t:])
        if red > mejor_red:
            mejor_red, mejor_idx = red, t
    return mejor_idx, mejor_red


def detectar_cambios_regimen(df: pd.DataFrame, min_size: int = 3,
                             pen_mult: float = 2.5) -> pd.DataFrame:
    """Detecta cambios sostenidos de nivel en el gasto mensual total.

    Usa segmentación binaria: busca recursivamente el corte que más reduce el
    error, y lo acepta solo si la reducción supera una penalización tipo BIC
    (`pen_mult · σ² · log n`), donde σ² es una estimación robusta del ruido. Así
    se evita trocear la serie por fluctuaciones normales.

    Devuelve un DataFrame con una fila por cambio detectado: `mes`, `indice`,
    `media_antes`, `media_despues`, `cambio` (€), `direccion` ('subida'/'bajada').
    """
    serie = serie_gasto_total_mensual(df)
    cols = ["mes", "indice", "media_antes", "media_despues", "cambio", "direccion"]
    if len(serie) < 2 * min_size + 1:
        return pd.DataFrame(columns=cols)

    vals = serie.values.astype(float)
    meses = list(serie.index.astype(str))
    n = len(vals)

    # Estimación robusta del ruido (a partir de las diferencias mes a mes).
    sigma = 1.4826 * np.median(np.abs(np.diff(vals))) if n > 1 else vals.std()
    sigma = max(sigma, 1e-6)
    penalizacion = pen_mult * (sigma ** 2) * np.log(n)

    # Segmentación binaria recursiva.
    cortes = []
    pendientes = [(0, n)]
    while pendientes:
        a, b = pendientes.pop()
        if b - a < 2 * min_size:
            continue
        idx, red = _mejor_corte(vals[a:b], min_size)
        if idx is not None and red > penalizacion:
            corte = a + idx
            cortes.append(corte)
            pendientes.append((a, corte))
            pendientes.append((corte, b))

    cortes.sort()
    filas = []
    for c in cortes:
        # Nivel antes/después usando los segmentos vecinos.
        prev = max([0] + [x for x in cortes if x < c])
        sig = min([n] + [x for x in cortes if x > c])
        media_antes = float(vals[prev:c].mean())
        media_despues = float(vals[c:sig].mean())
        cambio = media_despues - media_antes
        filas.append({
            "mes": meses[c],
            "indice": int(c),
            "media_antes": round(media_antes, 2),
            "media_despues": round(media_despues, 2),
            "cambio": round(cambio, 2),
            "direccion": "subida" if cambio > 0 else "bajada",
        })
    return pd.DataFrame(filas, columns=cols)


# ---------------------------------------------------------------------------
# 2. Anomalías no supervisadas (IsolationForest)
# ---------------------------------------------------------------------------

def detectar_anomalias_isolation(df: pd.DataFrame, contaminacion: float = 0.03,
                                 seed: int = 42) -> pd.DataFrame:
    """Marca transacciones atípicas con un IsolationForest entrenado en el
    propio historial del usuario (aprende su patrón).

    Considera de forma conjunta: importe (valor absoluto del gasto), día del mes,
    día de la semana y categoría (codificada one-hot). Devuelve el DataFrame con
    columnas añadidas `anomalia_if` (bool) y `score_if` (cuanto más negativo, más
    anómala). Solo se evalúan los gastos (importe < 0).
    """
    out = df.copy()
    out["anomalia_if"] = False
    out["score_if"] = 0.0

    gastos = out[out["importe"] < 0]
    if len(gastos) < 10:
        return out

    feat = pd.DataFrame(index=gastos.index)
    feat["importe_abs"] = gastos["importe"].abs().values
    feat["dia_mes"] = gastos["fecha"].dt.day.values
    feat["dia_semana"] = gastos["fecha"].dt.dayofweek.values
    cats = pd.get_dummies(gastos["categoria"], prefix="cat")
    X = pd.concat([feat, cats.set_index(feat.index)], axis=1).fillna(0.0)

    modelo = IsolationForest(contamination=contaminacion, random_state=seed,
                             n_estimators=200)
    pred = modelo.fit_predict(X.values)
    scores = modelo.decision_function(X.values)

    out.loc[gastos.index, "anomalia_if"] = (pred == -1)
    out.loc[gastos.index, "score_if"] = scores.round(4)
    return out


# ---------------------------------------------------------------------------
# 3. Comparación con el baseline (z-score por categoría)
# ---------------------------------------------------------------------------

def comparar_detectores(df: pd.DataFrame) -> dict:
    """Compara IsolationForest (multivariante) con el z-score por categoría.

    Devuelve conteos y solapamiento, útil para discutir en la memoria qué aporta
    cada enfoque (el z-score es univariante por categoría; el IsolationForest es
    multivariante y aprende el patrón global del usuario).
    """
    z = detectar_anomalias_zscore(df)
    iso = detectar_anomalias_isolation(df)

    idx_z = set(z.index[z["anomalia"]])
    idx_if = set(iso.index[iso["anomalia_if"]])
    comunes = idx_z & idx_if

    return {
        "n_zscore": len(idx_z),
        "n_isolation": len(idx_if),
        "n_comunes": len(comunes),
        "solo_zscore": len(idx_z - idx_if),
        "solo_isolation": len(idx_if - idx_z),
    }


# ---------------------------------------------------------------------------
# Figuras
# ---------------------------------------------------------------------------

def figura_cambios_regimen(df: pd.DataFrame):
    """Serie de gasto mensual con líneas verticales en los cambios de régimen."""
    import plotly.graph_objects as go
    try:
        from config import PLOTLY_LAYOUT, PREFIN_INK, PREFIN_AMBAR
    except Exception:  # pragma: no cover
        PLOTLY_LAYOUT, PREFIN_INK, PREFIN_AMBAR = {}, "#0F172A", "#D97706"

    serie = serie_gasto_total_mensual(df)
    cambios = detectar_cambios_regimen(df)
    x = [str(p) for p in serie.index]

    fig = go.Figure()
    fig.add_scatter(x=x, y=serie.values, mode="lines+markers", name="Gasto mensual",
                    line=dict(color=PREFIN_INK, width=2))
    for _, c in cambios.iterrows():
        fig.add_vline(x=c["mes"], line=dict(color=PREFIN_AMBAR, width=1.5, dash="dash"))
    fig.update_layout(**PLOTLY_LAYOUT)
    fig.update_layout(title_text="Cambios de régimen en el gasto",
                      xaxis_tickangle=-30, yaxis_title="Gasto (€)")
    return fig
