# modules/ml_model.py — Modelo de predicción de riesgo financiero

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from modulos.analyzer import resumen_mensual, gasto_por_categoria_mes


# ---------------------------------------------------------------------------
# Generación de features a partir del DataFrame de transacciones
# ---------------------------------------------------------------------------

FEATURES = [
    "ingreso_mensual",
    "gasto_total",
    "ratio_gasto_ingreso",      # gasto / ingreso
    "ratio_ahorro",             # (ingreso - gasto) / ingreso
    "gasto_ocio_ratio",         # ocio / ingreso
    "gasto_suscripciones",
    "gasto_supermercado",
    "variabilidad_gasto",       # desviación estándar mensual del gasto
    "tendencia_gasto",          # pendiente de la regresión (€/mes)
    "n_categorias_activas",
    "colchon_meses",            # saldo / gasto mensual (meses de runway)
]


def extraer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extrae un DataFrame de features por mes a partir del DF de transacciones.
    Cada fila es un mes; las features son las variables de entrada del modelo.
    """
    resumen = resumen_mensual(df)
    cats    = gasto_por_categoria_mes(df)

    # Unir resumen con gastos por categoría
    merged = resumen.merge(cats, on="mes", how="left").fillna(0)

    features = pd.DataFrame()
    features["mes"]              = merged["mes"]
    features["ingreso_mensual"]  = merged["ingreso_total"]
    features["gasto_total"]      = merged["gasto_total"]
    features["ratio_gasto_ingreso"] = (
        features["gasto_total"] / features["ingreso_mensual"].replace(0, np.nan)
    ).fillna(0)
    features["ratio_ahorro"] = 1 - features["ratio_gasto_ingreso"]

    ocio_col = "Restaurantes y Ocio" if "Restaurantes y Ocio" in merged.columns else None
    features["gasto_ocio_ratio"] = (
        merged[ocio_col] / features["ingreso_mensual"].replace(0, np.nan)
        if ocio_col else 0
    ).fillna(0)

    sus_col = "Suscripciones" if "Suscripciones" in merged.columns else None
    features["gasto_suscripciones"] = merged[sus_col] if sus_col else 0

    sup_col = "Supermercado" if "Supermercado" in merged.columns else None
    features["gasto_supermercado"] = merged[sup_col] if sup_col else 0

    # Variabilidad: desv. estándar expandida del gasto total (rolling)
    features["variabilidad_gasto"] = (
        features["gasto_total"].expanding().std().fillna(0)
    )

    # Tendencia: pendiente local (ventana 3 meses)
    gastos_vals = features["gasto_total"].values
    pendientes = [0.0, 0.0]
    for i in range(2, len(gastos_vals)):
        x = np.array([0, 1, 2])
        y = gastos_vals[i-2:i+1]
        pend, _ = np.polyfit(x, y, 1)
        pendientes.append(pend)
    features["tendencia_gasto"] = pendientes

    # Número de categorías activas
    cat_cols = [c for c in merged.columns if c in [
        "Supermercado", "Restaurantes y Ocio", "Transporte",
        "Servicios del Hogar", "Suscripciones", "Salud y Farmacia",
        "Ropa y Compras", "Educación",
    ]]
    features["n_categorias_activas"] = (merged[cat_cols] > 0).sum(axis=1)

    # Colchón de liquidez: saldo al cierre del mes / gasto mensual = meses de
    # margen. Es el factor más determinante del riesgo de iliquidez a corto plazo
    # (un gasto alto con colchón holgado NO es peligroso). Sin esta variable, el
    # clasificador sobreestimaba el riesgo de quienes gastan mucho pero tienen
    # ahorro acumulado.
    if "saldo_acumulado" in df.columns:
        saldo_mes = (df.assign(_m=df["fecha"].dt.to_period("M").astype(str))
                     .groupby("_m")["saldo_acumulado"].last())
        saldo_alineado = saldo_mes.reindex(features["mes"]).values
    else:
        saldo_alineado = np.zeros(len(features))
    features["colchon_meses"] = np.clip(
        saldo_alineado / features["gasto_total"].replace(0, np.nan), 0, None
    )
    features["colchon_meses"] = features["colchon_meses"].fillna(0)

    return features


def etiquetar_riesgo(features: pd.DataFrame) -> pd.Series:
    """
    Heurística para generar etiquetas de riesgo (0=Bajo, 1=Medio, 2=Alto).
    Se usa para entrenar el modelo cuando no hay datos etiquetados.
    """
    cond_alto = (
        (features["ratio_gasto_ingreso"] > 0.90) |
        (features["gasto_ocio_ratio"] > 0.25) |
        (features["ratio_ahorro"] < 0.05)
    )
    cond_medio = (
        (features["ratio_gasto_ingreso"] > 0.75) |
        (features["gasto_ocio_ratio"] > 0.15) |
        (features["ratio_ahorro"] < 0.12)
    )
    labels = pd.Series(0, index=features.index)  # Bajo por defecto
    labels[cond_medio] = 1
    labels[cond_alto]  = 2
    return labels


# ---------------------------------------------------------------------------
# Modelo de clasificación de riesgo
# ---------------------------------------------------------------------------

class ModeloRiesgo:
    """
    Envuelve un RandomForestClassifier entrenado sobre las features mensuales.
    También incluye un regresor de gasto futuro.
    """

    NIVEL_TEXTO = {0: "Bajo", 1: "Medio", 2: "Alto"}
    NIVEL_SCORE = {0: 20, 1: 55, 2: 85}  # puntuación 0-100 para el gauge

    def __init__(self):
        self.clf = Pipeline([
            ("scaler", StandardScaler()),
            ("rf", RandomForestClassifier(
                n_estimators=200, max_depth=6,
                random_state=42, class_weight="balanced"
            )),
        ])
        self.reg = Pipeline([
            ("scaler", StandardScaler()),
            ("gb", GradientBoostingRegressor(
                n_estimators=100, max_depth=4, random_state=42
            )),
        ])
        self._entrenado = False

    def entrenar(self, df: pd.DataFrame):
        """Entrena el modelo a partir del DataFrame de transacciones."""
        features = extraer_features(df)
        if len(features) < 3:
            # Datos insuficientes para entrenar
            self._entrenado = False
            return self

        X = features[FEATURES].fillna(0)
        y_cls = etiquetar_riesgo(features)
        y_reg = features["gasto_total"].shift(-1).ffill()

        self.clf.fit(X, y_cls)

        # Regresor (predice gasto del mes siguiente)
        X_reg = X.iloc[:-1]
        y_reg_fit = y_reg.iloc[:-1]
        if len(X_reg) >= 2:
            self.reg.fit(X_reg, y_reg_fit)

        self._entrenado = True
        self._features_names = FEATURES
        self._importancias = dict(zip(
            FEATURES,
            self.clf.named_steps["rf"].feature_importances_
        ))
        return self

    def predecir(self, df: pd.DataFrame) -> dict:
        """
        Devuelve predicciones para el último mes disponible.
        """
        if not self._entrenado:
            return self._resultado_defecto()

        features = extraer_features(df)
        if features.empty:
            return self._resultado_defecto()

        X = features[FEATURES].fillna(0)
        ultimo = X.iloc[[-1]]

        clase_pred = int(self.clf.predict(ultimo)[0])
        proba      = self.clf.predict_proba(ultimo)[0]

        gasto_futuro = None
        if hasattr(self.reg, "predict"):
            try:
                gasto_futuro = round(float(self.reg.predict(ultimo)[0]), 2)
            except Exception:
                gasto_futuro = None

        factores = self._factores_riesgo(features.iloc[-1])

        return {
            "nivel":           self.NIVEL_TEXTO[clase_pred],
            "score":           self.NIVEL_SCORE[clase_pred],
            "probabilidades":  {self.NIVEL_TEXTO[i]: round(p * 100, 1) for i, p in enumerate(proba)},
            "gasto_futuro_est": gasto_futuro,
            "factores":        factores,
            "importancias":    self._importancias,
            "features_ultimo": features.iloc[-1].to_dict(),
        }

    def _factores_riesgo(self, row: pd.Series) -> list:
        factores = []
        ratio_gasto = row.get("ratio_gasto_ingreso", 0)
        if ratio_gasto > 0.85:
            factores.append({
                "factor": "Gasto excesivo sobre ingresos",
                "valor":  f"{ratio_gasto*100:.0f}% del ingreso",
                "nivel":  "Alto",
            })
        elif ratio_gasto > 0.70:
            factores.append({
                "factor": "Gasto elevado sobre ingresos",
                "valor":  f"{ratio_gasto*100:.0f}% del ingreso",
                "nivel":  "Medio",
            })

        ratio_ocio = row.get("gasto_ocio_ratio", 0)
        if ratio_ocio > 0.20:
            factores.append({
                "factor": "Alto gasto en ocio y restaurantes",
                "valor":  f"{ratio_ocio*100:.0f}% del ingreso",
                "nivel":  "Medio",
            })

        ratio_ahorro = row.get("ratio_ahorro", 0)
        if ratio_ahorro < 0.10:
            factores.append({
                "factor": "Tasa de ahorro por debajo del mínimo recomendado (10%)",
                "valor":  f"{max(ratio_ahorro*100,0):.1f}% de ahorro",
                "nivel":  "Alto" if ratio_ahorro < 0.05 else "Medio",
            })

        tend = row.get("tendencia_gasto", 0)
        if tend > 30:
            factores.append({
                "factor": "Tendencia creciente en gastos mensuales",
                "valor":  f"+{tend:.0f} €/mes",
                "nivel":  "Medio",
            })

        if not factores:
            factores.append({
                "factor": "Situación financiera saludable",
                "valor":  "Sin alertas activas",
                "nivel":  "Bajo",
            })
        return factores

    @staticmethod
    def _resultado_defecto() -> dict:
        return {
            "nivel": "Desconocido",
            "score": 0,
            "probabilidades": {},
            "gasto_futuro_est": None,
            "factores": [{"factor": "Sin datos suficientes", "valor": "—", "nivel": "Bajo"}],
            "importancias": {},
            "features_ultimo": {},
        }


# Instancia global del modelo (se entrena al cargar datos)
modelo_riesgo = ModeloRiesgo()
