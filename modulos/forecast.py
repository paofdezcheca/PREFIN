# modulos/forecast.py — Previsión de gasto con incertidumbre (Fase 1)
"""
Previsión del gasto mensual **con intervalos de incertidumbre** (no un punto único).

Estrategia híbrida, pensada para el régimen de pocos datos típico de finanzas
personales (12-36 meses de historia):

1.  **Descomposición recurrente / variable.**
    `detectar_recurrentes()` identifica los flujos que se repiten cada mes
    (nómina, alquiler, recibos, suscripciones) por descripción normalizada,
    importe estable y periodicidad ~mensual. Estos flujos se proyectan de forma
    casi determinista (su mediana), porque su varianza real es baja y conocemos
    incluso el día del mes en que ocurren.

2.  **Modelo cuantílico para el gasto variable.**
    El gasto NO recurrente (supermercado, ocio, transporte…) se agrega por mes y
    se modela con **regresión cuantílica lineal** (`sklearn.linear_model.QuantileRegressor`)
    para los cuantiles p10/p50/p90, usando como variables una tendencia, una
    codificación estacional (seno/coseno de 12 meses) y el rezago de un mes.
    El modelo es **lineal** a propósito: con 12-36 puntos mensuales, un modelo de
    árboles sobreajusta; un modelo lineal da intervalos honestos y estables.
    Si hay menos de `_MIN_MESES_MODELO` meses utilizables, se cae a los
    **cuantiles empíricos** de la serie histórica (robusto ante escasez de datos).

3.  **Previsión total = recurrente (constante) + variable (banda cuantílica).**

La previsión se devuelve como banda p10–p50–p90, y se valida con un *backtest*
walk-forward honesto (entrenar con el pasado, predecir el mes siguiente nunca
visto) que reporta MAE, RMSE, MAPE y **cobertura del intervalo** frente a una
referencia *seasonal-naive*.

NOTA (Fase 1): el generador sintético actual solo contiene recurrencia mensual,
no estacionalidad anual. El término estacional seno/coseno queda implementado y
listo, pero solo aportará señal cuando el generador se enriquezca en la fase de
des-circularización. Esto es deliberado y está documentado para la memoria.

Dependencias: solo numpy / pandas / scikit-learn (ya presentes). Sin librerías
nuevas en tiempo de ejecución.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import QuantileRegressor

# Cuantiles de la banda de previsión (inferior / mediana / superior).
CUANTILES = (0.1, 0.5, 0.9)

# Meses mínimos de entrenamiento para usar el modelo cuantílico; por debajo de
# este umbral se usan cuantiles empíricos.
_MIN_MESES_MODELO = 6

# Peso del modelo en la combinación de previsiones (resto = seasonal-naive).
# Combinar con una referencia robusta (Bates & Granger, 1969) reduce el error y,
# sobre todo, la varianza cuando hay pocos datos y estacionalidad anual.
_PESO_MODELO = 0.5

# Calibración conforme dividida (split conformal / CQR).
# Se reserva una fracción de la serie como conjunto de calibración: nunca se usa
# para entrenar el modelo cuantílico. El ajuste q_hat expande el intervalo p10–p90
# hasta alcanzar la cobertura nominal del 80 % sobre datos no vistos.
_N_CAL_MIN = 8      # puntos mínimos para el conjunto de calibración
_N_CAL_FRAC = 0.20  # fracción de la serie reservada para calibración
_ALPHA = 0.20       # nivel de error (cobertura objetivo = 1 − α = 80 %)

# Parámetros de detección de flujos recurrentes. Criterio conservador: preferimos
# perder algún recurrente (que se quedará en el gasto variable y aún se prevé con
# incertidumbre) antes que confundir gasto discrecional con un compromiso fijo.
_MIN_OCURRENCIAS = 3      # nº mínimo de meses distintos en que aparece el flujo
_TOL_CV_IMPORTE = 0.20    # coef. de variación máximo del importe para ser "estable"
_MIN_COBERTURA = 0.50     # fracción mínima de meses (de su rango activo) en que aparece
_RITMO_MIN, _RITMO_MAX = 0.8, 1.3   # ~una vez al mes (excluye gasto multi-mensual)

_RE_DIGITOS = re.compile(r"\d+")
_RE_ESPACIOS = re.compile(r"\s+")


# ---------------------------------------------------------------------------
# Detección de flujos recurrentes (nómina, alquiler, recibos, suscripciones)
# ---------------------------------------------------------------------------

def _normalizar_descripcion(desc: str) -> str:
    """Normaliza una descripción para agrupar flujos recurrentes.

    Pasa a minúsculas, elimina dígitos (números de referencia/factura) y colapsa
    espacios, de modo que «Endesa 3392» y «Endesa 7781» se agrupen juntas.
    """
    if not isinstance(desc, str):
        return ""
    s = desc.lower().strip()
    s = _RE_DIGITOS.sub("", s)
    s = _RE_ESPACIOS.sub(" ", s).strip()
    return s


def detectar_recurrentes(
    df: pd.DataFrame,
    min_ocurrencias: int = _MIN_OCURRENCIAS,
    tol_cv: float = _TOL_CV_IMPORTE,
    min_cobertura: float = _MIN_COBERTURA,
) -> pd.DataFrame:
    """Detecta flujos que se repiten ~mensualmente con importe estable.

    Un flujo es recurrente si:
      * aparece en al menos `min_ocurrencias` meses distintos,
      * a un ritmo cercano a uno por mes (no varias veces al mes),
      * cubriendo al menos `min_cobertura` de los meses de su rango activo
        (filtra comercios que aparecen solo de forma esporádica), y
      * con un importe cuyo coeficiente de variación es menor que `tol_cv`.

    Devuelve un DataFrame con una fila por flujo recurrente y columnas:
    `desc_norm`, `descripcion_ejemplo`, `categoria`, `importe_mediano` (con signo),
    `cv_importe`, `dia_mes_tipico`, `n_ocurrencias`, `meses_distintos`,
    `es_ingreso`, `periodicidad_dias`.
    """
    columnas = ["desc_norm", "descripcion_ejemplo", "categoria", "importe_mediano",
                "cv_importe", "dia_mes_tipico", "n_ocurrencias", "meses_distintos",
                "es_ingreso", "periodicidad_dias"]
    if df is None or df.empty:
        return pd.DataFrame(columns=columnas)

    trabajo = df.copy()
    trabajo["desc_norm"] = trabajo["descripcion"].apply(_normalizar_descripcion)
    trabajo = trabajo[trabajo["desc_norm"] != ""]

    filas = []
    for desc_norm, grupo in trabajo.groupby("desc_norm"):
        meses = grupo["fecha"].dt.to_period("M")
        meses_distintos = meses.nunique()
        if meses_distintos < min_ocurrencias:
            continue

        # Ritmo ~mensual: en promedio una transacción por mes (con holgura).
        ritmo = len(grupo) / meses_distintos
        if not (_RITMO_MIN <= ritmo <= _RITMO_MAX):
            continue

        # Cobertura: fracción de meses (dentro de su rango activo) en que aparece.
        # Excluye comercios que solo aparecen de forma esporádica unos pocos meses.
        span = (meses.max() - meses.min()).n + 1
        cobertura = meses_distintos / span if span else 0.0
        if cobertura < min_cobertura:
            continue

        importes = grupo["importe"].astype(float)
        media_abs = importes.abs().mean()
        if media_abs == 0:
            continue
        cv = importes.std(ddof=0) / media_abs if media_abs else np.inf
        if cv > tol_cv:
            continue

        fechas_ord = grupo["fecha"].sort_values()
        gaps = fechas_ord.diff().dt.days.dropna()
        periodicidad = float(gaps.median()) if not gaps.empty else np.nan

        filas.append({
            "desc_norm": desc_norm,
            "descripcion_ejemplo": grupo["descripcion"].iloc[0],
            "categoria": grupo["categoria"].mode().iloc[0]
            if "categoria" in grupo and not grupo["categoria"].mode().empty else "Otros",
            "importe_mediano": float(importes.median()),
            "cv_importe": round(float(cv), 3),
            "dia_mes_tipico": int(grupo["fecha"].dt.day.median()),
            "n_ocurrencias": int(len(grupo)),
            "meses_distintos": int(meses_distintos),
            "es_ingreso": bool(importes.median() > 0),
            "periodicidad_dias": round(periodicidad, 1) if periodicidad == periodicidad else None,
        })

    if not filas:
        return pd.DataFrame(columns=columnas)
    return pd.DataFrame(filas)[columnas].sort_values(
        "importe_mediano", key=lambda s: s.abs(), ascending=False
    ).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Series mensuales auxiliares
# ---------------------------------------------------------------------------

def _reindexar_mensual(serie: pd.Series) -> pd.Series:
    """Reindexa una serie con índice de periodos mensuales rellenando huecos a 0."""
    if serie.empty:
        return serie
    rango = pd.period_range(serie.index.min(), serie.index.max(), freq="M")
    return serie.reindex(rango, fill_value=0.0)


def serie_gasto_variable_mensual(df: pd.DataFrame, recurrentes_gasto: set) -> pd.Series:
    """Gasto mensual NO recurrente (variable), en euros positivos.

    `recurrentes_gasto` es el conjunto de `desc_norm` de los flujos de gasto
    recurrentes que se excluyen de la parte variable.
    """
    gastos = df[df["importe"] < 0].copy()
    if gastos.empty:
        return pd.Series(dtype=float)
    gastos["desc_norm"] = gastos["descripcion"].apply(_normalizar_descripcion)
    variables = gastos[~gastos["desc_norm"].isin(recurrentes_gasto)]
    serie = variables.groupby(variables["fecha"].dt.to_period("M"))["importe"].sum().abs()
    return _reindexar_mensual(serie)


def serie_gasto_total_mensual(df: pd.DataFrame) -> pd.Series:
    """Gasto mensual total (recurrente + variable), en euros positivos."""
    gastos = df[df["importe"] < 0].copy()
    if gastos.empty:
        return pd.Series(dtype=float)
    serie = gastos.groupby(gastos["fecha"].dt.to_period("M"))["importe"].sum().abs()
    return _reindexar_mensual(serie)


# ---------------------------------------------------------------------------
# Previsor cuantílico
# ---------------------------------------------------------------------------

class PrevisorGasto:
    """Previsor del gasto mensual total con banda de incertidumbre p10/p50/p90.

    Uso::

        prev = PrevisorGasto().fit(df)
        banda = prev.predecir(meses_adelante=3)   # DataFrame con p10/p50/p90

    Modela el **gasto total mensual** (que captura toda la varianza, incluida la
    de recibos de importe variable como el alquiler) con un modelo de mediana
    cuantílico, y construye la banda de incertidumbre a partir de los **cuantiles
    de los residuos** del propio modelo. Los flujos recurrentes detectados se
    usan como desglose explicativo (cuánto del gasto previsto es compromiso fijo
    frente a gasto discrecional), no para colapsar la incertidumbre.
    """

    def __init__(self, cuantiles=CUANTILES):
        self.cuantiles = tuple(sorted(cuantiles))
        self._modelo_med: QuantileRegressor | None = None
        self._res_q: dict[float, float] = {}      # cuantil -> desviación de residuo
        self._fallback: dict[float, float] | None = None
        self._q_hat: float = 0.0                  # corrección conforme
        self.n_cal_: int = 0
        self.usa_conformal_: bool = False
        self.recurrentes_: pd.DataFrame | None = None
        self.serie_total_: pd.Series | None = None
        self.gasto_recurrente_: float = 0.0
        self.usa_modelo_: bool = False

    # -- entrenamiento -------------------------------------------------------
    def fit(self, df: pd.DataFrame) -> "PrevisorGasto":
        """Ajusta el previsor sobre el histórico de transacciones `df`."""
        if df is None or df.empty:
            raise ValueError("No se puede ajustar el previsor sobre un DataFrame vacío.")

        # Recurrentes: solo para desglose explicativo del gasto previsto.
        self.recurrentes_ = detectar_recurrentes(df)
        recur_gasto = self.recurrentes_[~self.recurrentes_["es_ingreso"]]
        self.gasto_recurrente_ = float(recur_gasto["importe_mediano"].abs().sum())

        # La banda se modela sobre el gasto TOTAL mensual.
        serie = serie_gasto_total_mensual(df)
        self.serie_total_ = serie

        y = serie.values.astype(float)
        # Filas utilizables = n-1 (la primera se pierde por el rezago).
        if len(y) - 1 < _MIN_MESES_MODELO:
            self._fallback = {q: float(np.quantile(y, q)) for q in self.cuantiles}
            self.usa_modelo_ = False
            return self

        # Partición temporal: training + calibración conforme.
        # La calibración se toma de los ÚLTIMOS meses (respeta el orden temporal).
        n_full = len(y)
        n_cal = max(_N_CAL_MIN, int(n_full * _N_CAL_FRAC))
        n_train = n_full - n_cal
        if n_train - 1 < _MIN_MESES_MODELO:   # sin margen → usa todo, sin conformal
            n_train = n_full
            n_cal = 0

        X_all = self._matriz_features(serie)        # (n_full, 4)
        naive_all = self._naive_estacional(serie)   # (n_full,)

        X_tr = X_all[1:n_train]   # (n_train-1, 4)  — fila 0 descartada (no hay lag)
        y_tr = y[1:n_train]       # (n_train-1,)

        self._modelo_med = QuantileRegressor(quantile=0.5, alpha=0.001, solver="highs")
        self._modelo_med.fit(X_tr, y_tr)

        # Residuos sobre el conjunto de entrenamiento → anchura inicial de la banda.
        naive_tr = naive_all[1:n_train]
        blend_tr = _PESO_MODELO * self._modelo_med.predict(X_tr) + (1 - _PESO_MODELO) * naive_tr
        residuos = y_tr - blend_tr
        self._res_q = {q: float(np.quantile(residuos, q)) for q in self.cuantiles}

        # Calibración conforme dividida (CQR): calcula la corrección q_hat sobre
        # datos nunca vistos por el modelo, de modo que la cobertura real del
        # intervalo [p10 - q_hat, p90 + q_hat] alcance la cobertura nominal (80 %).
        if n_cal > 0:
            X_cal = X_all[n_train:]
            y_cal = y[n_train:]
            naive_cal = naive_all[n_train:]
            med_cal = (_PESO_MODELO * self._modelo_med.predict(X_cal)
                       + (1 - _PESO_MODELO) * naive_cal)
            p10_cal = med_cal + self._res_q[0.1]
            p90_cal = med_cal + self._res_q[0.9]
            # Puntuación CQR: positiva cuando y cae fuera del intervalo.
            scores = np.maximum(p10_cal - y_cal, y_cal - p90_cal)
            q_level = min((1 - _ALPHA) * (1 + 1.0 / n_cal), 1.0)
            self._q_hat = float(np.quantile(scores, q_level))
            self.n_cal_ = n_cal
            self.usa_conformal_ = True
        else:
            self._q_hat = 0.0
            self.n_cal_ = 0
            self.usa_conformal_ = False

        self.usa_modelo_ = True
        self._fallback = None
        return self

    # -- features ------------------------------------------------------------
    @staticmethod
    def _matriz_features(serie: pd.Series) -> np.ndarray:
        """Construye la matriz de variables para toda la serie (sin recortar).

        Variables: tendencia (t/12), estacionalidad anual (sen, cos de 12 meses)
        y rezago de un mes (lag1). El modelo es deliberadamente sencillo (lineal,
        pocos parámetros) para no sobreajustar con ~24-35 puntos; la señal
        estacional anual se aporta por separado mediante la combinación con el
        seasonal-naive (ver `_naive_estacional` y la mezcla en fit/predecir).
        """
        y = serie.values.astype(float)
        n = len(y)
        t = np.arange(n) / 12.0
        meses = np.array([p.month for p in serie.index], dtype=float)
        sen = np.sin(2 * np.pi * meses / 12.0)
        cos = np.cos(2 * np.pi * meses / 12.0)
        lag1 = np.roll(y, 1)
        lag1[0] = y[0]
        return np.column_stack([t, sen, cos, lag1])

    @staticmethod
    def _naive_estacional(serie: pd.Series) -> np.ndarray:
        """Predicción seasonal-naive en muestra: y[t-12] si existe, si no y[t-1]."""
        y = serie.values.astype(float)
        n = len(y)
        naive = np.empty(n)
        for i in range(n):
            if i >= 12:
                naive[i] = y[i - 12]
            elif i >= 1:
                naive[i] = y[i - 1]
            else:
                naive[i] = y[i]
        return naive

    def _construir_features(self, serie: pd.Series):
        """Devuelve (X, y) alineados, descartando la primera fila (rezago)."""
        X_all = self._matriz_features(serie)
        y = serie.values.astype(float)
        return X_all[1:], y[1:]

    # -- predicción ----------------------------------------------------------
    def predecir(self, meses_adelante: int = 1) -> pd.DataFrame:
        """Devuelve la banda de previsión del gasto total para los próximos meses.

        Columnas: `mes`, `horizonte`, `p10`, `p50`, `p90` (gasto total en €),
        `variable_p50` y `recurrente` (desglose del p50).
        """
        if self.serie_total_ is None:
            raise RuntimeError("Llama a fit() antes de predecir().")

        meses_adelante = max(1, int(meses_adelante))
        serie = self.serie_total_
        n = len(serie)
        ultimo_periodo = serie.index[-1]
        recurrente = self.gasto_recurrente_
        filas = []

        if not self.usa_modelo_:
            # Cuantiles empíricos: banda constante para todos los horizontes.
            banda = np.sort([self._fallback[q] for q in self.cuantiles])
            for h in range(1, meses_adelante + 1):
                filas.append(self._fila(ultimo_periodo + h, h, banda, recurrente))
            return pd.DataFrame(filas)

        cur_lag = float(serie.values[-1])
        for h in range(1, meses_adelante + 1):
            periodo = ultimo_periodo + h
            t = (n - 1 + h) / 12.0
            mes = periodo.month
            x = np.array([[t, np.sin(2 * np.pi * mes / 12.0),
                           np.cos(2 * np.pi * mes / 12.0), cur_lag]])
            med_modelo = float(self._modelo_med.predict(x)[0])
            # seasonal-naive: mismo mes del año anterior, si está; si no, último valor.
            periodo_12 = periodo - 12
            naive = float(serie.loc[periodo_12]) if periodo_12 in serie.index else cur_lag
            med = _PESO_MODELO * med_modelo + (1 - _PESO_MODELO) * naive
            # Banda = combinación + cuantiles de residuos (monótona por construcción).
            p10_raw = med + self._res_q[0.1] - self._q_hat
            p50_raw = med + self._res_q[0.5]
            p90_raw = med + self._res_q[0.9] + self._q_hat
            banda = np.clip(
                [min(p10_raw, p50_raw), p50_raw, max(p90_raw, p50_raw)],
                0.0, None)
            filas.append(self._fila(periodo, h, banda, recurrente))
            cur_lag = med  # rezago para el siguiente horizonte = mediana combinada
        return pd.DataFrame(filas)

    def _fila(self, periodo, horizonte, banda, recurrente) -> dict:
        p10, p50, p90 = banda[0], banda[len(banda) // 2], banda[-1]
        return {
            "mes": str(periodo),
            "horizonte": horizonte,
            "p10": round(p10, 2),
            "p50": round(p50, 2),
            "p90": round(p90, 2),
            "variable_p50": round(max(p50 - recurrente, 0.0), 2),
            "recurrente": round(recurrente, 2),
        }


# ---------------------------------------------------------------------------
# Validación honesta (backtest walk-forward)
# ---------------------------------------------------------------------------

def backtest_previsor(df: pd.DataFrame, min_train: int = 8) -> dict:
    """Backtest *walk-forward* de un paso del previsor sobre el gasto total.

    Para cada mes a partir de `min_train`, se entrena con todo el pasado y se
    predice ese mes (nunca visto). Se comparan las previsiones con el gasto real
    y se calcula MAE, RMSE, MAPE y **cobertura** del intervalo p10–p90 (fracción
    de meses reales que cayeron dentro de la banda; idealmente ≈ 0.80).

    Se incluye una referencia *seasonal-naive* (mismo mes del año anterior si
    existe, si no el mes previo) para poder afirmar que el modelo la mejora.

    Devuelve un diccionario con las métricas del modelo y de la referencia, más
    el número de pliegues evaluados.
    """
    serie_total = serie_gasto_total_mensual(df)
    periodos = list(serie_total.index)
    n = len(periodos)
    vacio = {
        "n_folds": 0, "mae": None, "rmse": None, "mape": None, "cobertura": None,
        "mae_baseline": None, "rmse_baseline": None, "nivel_cobertura_objetivo": 0.80,
    }
    if n <= min_train:
        return vacio

    err, err_base, dentro, reales = [], [], [], []
    for i in range(min_train, n):
        periodo_objetivo = periodos[i]
        corte = periodo_objetivo.to_timestamp()  # primer día del mes objetivo
        df_train = df[df["fecha"] < corte]
        if df_train["fecha"].dt.to_period("M").nunique() < min_train:
            continue

        real = float(serie_total.iloc[i])
        try:
            pred = PrevisorGasto().fit(df_train).predecir(1).iloc[0]
        except Exception:
            continue

        err.append(real - pred["p50"])
        dentro.append(pred["p10"] <= real <= pred["p90"])
        reales.append(real)

        # Referencia seasonal-naive.
        base = float(serie_total.iloc[i - 12]) if i - 12 >= 0 else float(serie_total.iloc[i - 1])
        err_base.append(real - base)

    if not err:
        return vacio

    err = np.array(err)
    err_base = np.array(err_base)
    reales = np.array(reales)
    mape = float(np.mean(np.abs(err) / np.where(reales == 0, np.nan, reales)) * 100)

    return {
        "n_folds": int(len(err)),
        "mae": round(float(np.mean(np.abs(err))), 2),
        "rmse": round(float(np.sqrt(np.mean(err ** 2))), 2),
        "mape": round(mape, 1),
        "cobertura": round(float(np.mean(dentro)), 3),
        "mae_baseline": round(float(np.mean(np.abs(err_base))), 2),
        "rmse_baseline": round(float(np.sqrt(np.mean(err_base ** 2))), 2),
        "nivel_cobertura_objetivo": 0.80,
    }


# ---------------------------------------------------------------------------
# Figura exportable
# ---------------------------------------------------------------------------

def figura_prevision(df: pd.DataFrame, meses_adelante: int = 6):
    """Figura Plotly: histórico de gasto + banda de previsión p10–p50–p90.

    Devuelve un `plotly.graph_objects.Figure` listo para mostrar en la app o
    exportar a la memoria. Importa Plotly de forma perezosa para no acoplar el
    módulo de análisis a la capa de visualización.
    """
    import plotly.graph_objects as go
    try:
        from config import PLOTLY_LAYOUT, PREFIN_INK, PREFIN_TEXTO_SEC
    except Exception:  # pragma: no cover - fallback si config no está disponible
        PLOTLY_LAYOUT, PREFIN_INK, PREFIN_TEXTO_SEC = {}, "#0F172A", "#71717A"

    serie_total = serie_gasto_total_mensual(df)
    prev = PrevisorGasto().fit(df)
    banda = prev.predecir(meses_adelante)

    fig = go.Figure()
    # Histórico.
    fig.add_scatter(
        x=[str(p) for p in serie_total.index], y=serie_total.values,
        mode="lines+markers", name="Gasto real",
        line=dict(color=PREFIN_INK, width=2), marker=dict(size=6),
    )
    # Banda de incertidumbre (p10–p90).
    meses = banda["mes"].tolist()
    fig.add_scatter(x=meses, y=banda["p90"], mode="lines", line=dict(width=0),
                    showlegend=False, hoverinfo="skip")
    fig.add_scatter(x=meses, y=banda["p10"], mode="lines", line=dict(width=0),
                    fill="tonexty", fillcolor="rgba(15,23,42,0.12)",
                    name="Banda p10–p90", hoverinfo="skip")
    # Mediana prevista.
    fig.add_scatter(x=meses, y=banda["p50"], mode="lines+markers",
                    name="Previsión (mediana)",
                    line=dict(color=PREFIN_TEXTO_SEC, width=2, dash="dash"),
                    marker=dict(size=6))

    fig.update_layout(**PLOTLY_LAYOUT)
    fig.update_layout(title_text="Previsión de gasto mensual con incertidumbre",
                      xaxis_tickangle=-30)
    return fig


def exportar_figura_png(df: pd.DataFrame, ruta: str, meses_adelante: int = 6) -> str:
    """Exporta la figura de previsión a PNG estático con matplotlib.

    Se usa matplotlib (no kaleido) por ser mucho más robusto en Windows para
    generar imágenes estáticas reutilizables en la memoria. Requiere matplotlib
    (dependencia de desarrollo). Devuelve la ruta del PNG escrito.
    """
    import matplotlib
    matplotlib.use("Agg")  # backend sin ventana, apto para scripts
    import matplotlib.pyplot as plt

    serie_total = serie_gasto_total_mensual(df)
    prev = PrevisorGasto().fit(df)
    banda = prev.predecir(meses_adelante)

    hist_x = [str(p) for p in serie_total.index]
    fut_x = banda["mes"].tolist()

    fig, ax = plt.subplots(figsize=(10, 5), dpi=150)
    ax.plot(hist_x, serie_total.values, color="#0F172A", linewidth=2,
            marker="o", markersize=4, label="Gasto real")
    ax.fill_between(fut_x, banda["p10"], banda["p90"], color="#0F172A",
                    alpha=0.12, label="Banda p10–p90")
    ax.plot(fut_x, banda["p50"], color="#71717A", linewidth=2, linestyle="--",
            marker="o", markersize=4, label="Previsión (mediana)")

    ax.set_title("Previsión de gasto mensual con incertidumbre", loc="left")
    ax.set_ylabel("Gasto (€)")
    ax.legend(frameon=False, fontsize=9)
    ax.grid(axis="y", color="#E7E5E4", linewidth=0.8)
    ax.spines[["top", "right"]].set_visible(False)
    plt.xticks(rotation=45, ha="right", fontsize=7)
    plt.tight_layout()
    fig.savefig(ruta)
    plt.close(fig)
    return ruta


# ---------------------------------------------------------------------------
# Generación de artefactos para la memoria (figura + métricas)
# ---------------------------------------------------------------------------

def generar_artefactos_fase1(df: pd.DataFrame, carpeta: str = "exports/fase1",
                             meses_adelante: int = 6) -> dict:
    """Genera y guarda los artefactos de la Fase 1 reutilizables en la memoria.

    Escribe en `carpeta`:
      * `prevision.html` — figura interactiva (siempre).
      * `prevision.png`  — figura estática (si `kaleido` está instalado).
      * `metricas.json`  — métricas del backtest.
      * `prevision_tabla.csv` — banda de previsión p10/p50/p90.
      * `recurrentes.csv` — flujos recurrentes detectados.

    Devuelve un diccionario {nombre: ruta} con los archivos efectivamente escritos.
    """
    destino = Path(carpeta)
    destino.mkdir(parents=True, exist_ok=True)
    escritos = {}

    prev = PrevisorGasto().fit(df)
    banda = prev.predecir(meses_adelante)
    metricas = backtest_previsor(df)
    fig = figura_prevision(df, meses_adelante)

    ruta_html = destino / "prevision.html"
    fig.write_html(str(ruta_html), include_plotlyjs="cdn")
    escritos["html"] = str(ruta_html)

    try:  # PNG estático con matplotlib (dependencia de desarrollo).
        ruta_png = destino / "prevision.png"
        exportar_figura_png(df, str(ruta_png), meses_adelante)
        escritos["png"] = str(ruta_png)
    except Exception as e:  # pragma: no cover - matplotlib ausente
        escritos["png_error"] = f"PNG no generado (instala matplotlib): {e}"

    ruta_metricas = destino / "metricas.json"
    ruta_metricas.write_text(json.dumps(metricas, indent=2, ensure_ascii=False),
                             encoding="utf-8")
    escritos["metricas"] = str(ruta_metricas)

    ruta_tabla = destino / "prevision_tabla.csv"
    banda.to_csv(ruta_tabla, index=False, encoding="utf-8")
    escritos["tabla"] = str(ruta_tabla)

    ruta_recur = destino / "recurrentes.csv"
    prev.recurrentes_.to_csv(ruta_recur, index=False, encoding="utf-8")
    escritos["recurrentes"] = str(ruta_recur)

    return escritos
