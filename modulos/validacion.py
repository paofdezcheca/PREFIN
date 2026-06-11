# modulos/validacion.py — Rigor de validación (Fase 6)
"""
Consolida la validación honesta de los tres componentes predictivos, con sus
límites y supuestos documentados. Es el material de rigor del Capítulo 6.

1.  **Previsión de gasto** (`validar_forecast`): backtest *walk-forward* de un
    paso (entrenar con el pasado, predecir el mes siguiente nunca visto). Reporta
    MAE/RMSE/MAPE y la **cobertura** del intervalo p10–p90 frente a una referencia
    seasonal-naive. Límite conocido: con datos casi i.i.d. el intervalo queda algo
    infra-calibrado; se reporta tal cual.

2.  **Clasificador de riesgo** (`validar_clasificador`): validación cruzada por
    GRUPOS (GroupKFold por usuario, sin fuga entre usuarios). Reporta ROC-AUC por
    pliegue, **Brier score** (calidad de la probabilidad) y la **curva de
    calibración** (¿una probabilidad del 70% ocurre el 70% de las veces?).

3.  **Probabilidad de iliquidez del gemelo** (`backtest_prob_iliquidez`): para
    muchos puntos (usuario, mes) se compara la P(iliquidez) que estima el Monte
    Carlo con lo que REALMENTE ocurrió en los meses siguientes. Mide si el gemelo
    está bien calibrado (backtesting de probabilidades).

Supuestos: los datos son sintéticos; las conclusiones valen para el generador
descrito. La validación cruzada por usuario evita el optimismo de evaluar sobre
los mismos individuos del entrenamiento.

Dependencias: scikit-learn (ya presente). Sin librerías nuevas.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score, brier_score_loss
from sklearn.calibration import calibration_curve
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from modulos.forecast import backtest_previsor
from modulos.riesgo_futuro import construir_dataset, VENTANA_RIESGO
from modulos.digital_twin import simular_montecarlo
from modulos.ml_model import extraer_features, FEATURES


# ---------------------------------------------------------------------------
# 1. Forecast
# ---------------------------------------------------------------------------

def validar_forecast(df: pd.DataFrame, min_train: int = 8) -> dict:
    """Backtest walk-forward del previsor (envuelve `forecast.backtest_previsor`)."""
    m = backtest_previsor(df, min_train=min_train)
    m["mejora_mae_vs_baseline"] = (
        round(m["mae_baseline"] - m["mae"], 2)
        if m.get("mae") is not None and m.get("mae_baseline") is not None else None)
    return m


# ---------------------------------------------------------------------------
# 2. Clasificador de riesgo (CV por grupos + calibración)
# ---------------------------------------------------------------------------

def _nuevo_clf():
    return Pipeline([
        ("scaler", StandardScaler()),
        ("rf", RandomForestClassifier(
            n_estimators=300, max_depth=8, min_samples_leaf=5,
            random_state=42, class_weight="balanced")),
    ])


def validar_clasificador(panel: pd.DataFrame, n_splits: int = 4,
                         ventana: int = VENTANA_RIESGO) -> dict:
    """Validación cruzada por usuario del clasificador de riesgo.

    Devuelve ROC-AUC por pliegue (media y desviación), Brier score global sobre
    las predicciones out-of-fold y los puntos de la curva de calibración.
    """
    X, y, grupos = construir_dataset(panel, ventana)
    n_grupos = len(np.unique(grupos))
    if len(y) < 50 or len(np.unique(y)) < 2 or n_grupos < n_splits:
        return {"ok": False, "motivo": "datos insuficientes",
                "n_muestras": int(len(y)), "n_usuarios": int(n_grupos)}

    gkf = GroupKFold(n_splits=n_splits)
    aucs, oof_proba, oof_y = [], np.zeros(len(y)), y.copy()
    for tr, te in gkf.split(X, y, grupos):
        clf = _nuevo_clf().fit(X[tr], y[tr])
        proba = clf.predict_proba(X[te])[:, 1]
        oof_proba[te] = proba
        if len(np.unique(y[te])) > 1:
            aucs.append(roc_auc_score(y[te], proba))

    brier = float(brier_score_loss(oof_y, oof_proba))
    frac_pos, media_pred = calibration_curve(oof_y, oof_proba, n_bins=10,
                                             strategy="quantile")
    return {
        "ok": True,
        "n_muestras": int(len(y)),
        "n_usuarios": int(n_grupos),
        "tasa_positivos": round(float(y.mean()), 3),
        "auc_media": round(float(np.mean(aucs)), 3) if aucs else None,
        "auc_desv": round(float(np.std(aucs)), 3) if aucs else None,
        "auc_por_pliegue": [round(float(a), 3) for a in aucs],
        "brier": round(brier, 4),
        "calibracion": {"prob_real": [round(float(v), 3) for v in frac_pos],
                        "prob_predicha": [round(float(v), 3) for v in media_pred]},
    }


# ---------------------------------------------------------------------------
# 3. Backtesting de la P(iliquidez) del gemelo Monte Carlo
# ---------------------------------------------------------------------------

def backtest_prob_iliquidez(panel: pd.DataFrame, ventana: int = VENTANA_RIESGO,
                            n_sim: int = 1500, max_puntos: int = 150,
                            min_hist: int = 6, seed: int = 0) -> dict:
    """Compara la P(iliquidez) estimada por el gemelo con lo realmente ocurrido.

    Para una muestra de puntos (usuario, mes) se calcula, con datos HASTA ese mes,
    la P(iliquidez) en los próximos `ventana` meses según el Monte Carlo, y se
    contrasta con la iliquidez realmente observada. Devuelve Brier score y la
    curva de calibración. Se limita a `max_puntos` evaluaciones por coste.
    """
    rng = np.random.default_rng(seed)
    probs, reales = [], []

    usuarios = panel["usuario"].unique() if "usuario" in panel.columns else [None]
    candidatos = []
    for uid in usuarios:
        df_u = panel[panel["usuario"] == uid] if uid is not None else panel
        meses = sorted(df_u["fecha"].dt.to_period("M").unique())
        for i in range(min_hist, len(meses) - ventana):
            candidatos.append((uid, meses[i]))

    if not candidatos:
        return {"ok": False, "motivo": "sin puntos evaluables", "n_puntos": 0}

    # Muestreo de puntos para acotar el coste computacional.
    idx = rng.choice(len(candidatos), size=min(max_puntos, len(candidatos)),
                     replace=False)
    for k in idx:
        uid, periodo = candidatos[k]
        df_u = panel[panel["usuario"] == uid] if uid is not None else panel
        corte = periodo.to_timestamp()
        df_hist = df_u[df_u["fecha"] < corte]
        if df_hist["fecha"].dt.to_period("M").nunique() < min_hist:
            continue
        try:
            mc = simular_montecarlo(df_hist, meses=ventana, n_sim=n_sim,
                                    umbral_iliquidez=0.0, seed=int(k))
            p = mc.get("prob_iliquidez_horizonte")
        except Exception:
            continue
        if p is None:
            continue
        # Iliquidez realmente observada en los próximos `ventana` meses.
        fin = (periodo + ventana).to_timestamp()
        futuro = df_u[(df_u["fecha"] >= corte) & (df_u["fecha"] < fin)]
        if futuro.empty:
            continue
        real = 1 if futuro["saldo_acumulado"].min() < 0 else 0
        probs.append(p); reales.append(real)

    if len(probs) < 10 or len(set(reales)) < 2:
        return {"ok": False, "motivo": "muestra insuficiente o sin variabilidad",
                "n_puntos": len(probs)}

    probs = np.array(probs); reales = np.array(reales)
    brier = float(brier_score_loss(reales, probs))
    frac_pos, media_pred = calibration_curve(reales, probs, n_bins=5,
                                             strategy="quantile")
    return {
        "ok": True,
        "n_puntos": int(len(probs)),
        "tasa_iliquidez_real": round(float(reales.mean()), 3),
        "prob_media_mc": round(float(probs.mean()), 3),
        "brier": round(brier, 4),
        "calibracion": {"prob_real": [round(float(v), 3) for v in frac_pos],
                        "prob_predicha": [round(float(v), 3) for v in media_pred]},
    }


# ---------------------------------------------------------------------------
# Figuras de calibración
# ---------------------------------------------------------------------------

def figura_calibracion(calibracion: dict, titulo: str):
    """Curva de calibración (fiabilidad) a partir de {prob_real, prob_predicha}."""
    import plotly.graph_objects as go
    try:
        from config import PLOTLY_LAYOUT, PREFIN_INK, PREFIN_TEXTO_SEC
    except Exception:  # pragma: no cover
        PLOTLY_LAYOUT, PREFIN_INK, PREFIN_TEXTO_SEC = {}, "#0F172A", "#71717A"

    fig = go.Figure()
    fig.add_scatter(x=[0, 1], y=[0, 1], mode="lines", name="Calibración perfecta",
                    line=dict(color=PREFIN_TEXTO_SEC, width=1, dash="dot"))
    if calibracion:
        fig.add_scatter(x=calibracion["prob_predicha"], y=calibracion["prob_real"],
                        mode="lines+markers", name="Modelo",
                        line=dict(color=PREFIN_INK, width=2))
    fig.update_layout(**PLOTLY_LAYOUT)
    fig.update_layout(title_text=titulo, xaxis_title="Probabilidad predicha",
                      yaxis_title="Frecuencia real",
                      xaxis_range=[0, 1], yaxis_range=[0, 1])
    return fig
