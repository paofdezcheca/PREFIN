# modulos/riesgo_futuro.py — Modelo de riesgo des-circularizado (Fases 3-6)
"""
Reemplaza el modelo de riesgo circular original por uno genuinamente predictivo.

EL PROBLEMA QUE RESUELVE
------------------------
El modelo anterior (`ml_model.ModeloRiesgo`) entrenaba un Random Forest para
reproducir `etiquetar_riesgo()`, una regla `if/else` sobre las MISMAS variables
que recibía como entrada. Era una tautología: el modelo no aportaba nada que la
regla no tuviera ya (un tribunal lo vería como «ML decorativo»).

LA SOLUCIÓN (des-circularización)
---------------------------------
1.  **La etiqueta es un EVENTO FUTURO observado, no una regla sobre el presente:**
    para cada mes se mira si el usuario entra en iliquidez (saldo < 0) en los
    próximos `ventana` meses. Eso es un RESULTADO del proceso (ingresos, gastos,
    shocks), independiente de las features.
2.  **Las features describen el PRESENTE** (ratios de gasto/ahorro, tendencia…)
    y el modelo aprende a anticipar el evento futuro a partir de ellas.
3.  **Entrenamiento sobre una POBLACIÓN de usuarios sintéticos** (multi-seed) con
    validación por GRUPOS: se entrena con unos usuarios y se evalúa en usuarios
    NUNCA vistos (GroupShuffleSplit), de modo que las métricas (ROC-AUC, etc.)
    son honestas y miden generalización real, no memorización.

El gemelo de Monte Carlo aporta, en inferencia, una segunda estimación de la
probabilidad de iliquidez (`prob_iliquidez_mc`), y en la Fase 6 se usa para el
backtesting de calibración.

Dependencias: scikit-learn (ya presente). Sin librerías nuevas.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import (
    roc_auc_score, accuracy_score, precision_score, recall_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from modulos.ml_model import extraer_features, FEATURES
from modulos.digital_twin import simular_montecarlo

VENTANA_RIESGO = 3  # meses hacia el futuro para definir la iliquidez


# ---------------------------------------------------------------------------
# Construcción del dataset (features del presente → etiqueta de evento futuro)
# ---------------------------------------------------------------------------

def _min_saldo_mensual(df_u: pd.DataFrame) -> pd.Series:
    """Saldo mínimo alcanzado dentro de cada mes (peor momento de liquidez)."""
    aux = df_u.copy()
    aux["mes"] = aux["fecha"].dt.to_period("M").astype(str)
    return aux.groupby("mes")["saldo_acumulado"].min()


def construir_dataset(df_multi: pd.DataFrame, ventana: int = VENTANA_RIESGO):
    """Construye (X, y, grupos) a partir de un panel multiusuario.

    Para cada usuario y cada mes con `ventana` meses de futuro disponibles:
      * X = features del presente (las de `ml_model.FEATURES`),
      * y = 1 si el saldo cae por debajo de 0 en alguno de los `ventana` meses
            siguientes, 0 en caso contrario,
      * grupo = identificador de usuario (para validar sin fuga entre usuarios).
    """
    if "usuario" not in df_multi.columns:
        df_multi = df_multi.assign(usuario=0)

    Xs, ys, grupos = [], [], []
    for uid, df_u in df_multi.groupby("usuario"):
        feats = extraer_features(df_u)
        if len(feats) < ventana + 2:
            continue
        min_saldo = _min_saldo_mensual(df_u)
        feats = feats[feats["mes"].isin(min_saldo.index)].reset_index(drop=True)
        mins = min_saldo.reindex(feats["mes"]).values
        n = len(feats)
        for i in range(n - ventana):
            etiqueta = 1 if np.nanmin(mins[i + 1:i + 1 + ventana]) < 0 else 0
            Xs.append(feats[FEATURES].iloc[i].fillna(0).values)
            ys.append(etiqueta)
            grupos.append(uid)

    return np.array(Xs, dtype=float), np.array(ys, dtype=int), np.array(grupos)


# ---------------------------------------------------------------------------
# Modelo
# ---------------------------------------------------------------------------

class ModeloRiesgoFuturo:
    """Clasifica el riesgo de iliquidez FUTURA a partir del estado presente.

    Se entrena una vez sobre una población de usuarios sintéticos y luego se
    aplica al usuario cargado. Expone una interfaz `predecir(df)` compatible con
    la página de Riesgo de la app.
    """

    NIVEL_SCORE = {"Bajo": 20, "Medio": 55, "Alto": 85}

    def __init__(self, ventana: int = VENTANA_RIESGO):
        self.ventana = ventana
        self.clf = Pipeline([
            ("scaler", StandardScaler()),
            ("rf", RandomForestClassifier(
                n_estimators=300, max_depth=8, min_samples_leaf=5,
                random_state=42, class_weight="balanced")),
        ])
        self._entrenado = False
        self.metricas_ = {}
        self.importancias_ = {}

    def entrenar(self, df_multi: pd.DataFrame) -> "ModeloRiesgoFuturo":
        """Entrena sobre el panel multiusuario con validación por grupos."""
        X, y, grupos = construir_dataset(df_multi, self.ventana)
        # Necesitamos ambas clases y suficientes muestras y usuarios.
        if len(y) < 40 or len(np.unique(y)) < 2 or len(np.unique(grupos)) < 4:
            self._entrenado = False
            return self

        gss = GroupShuffleSplit(n_splits=1, test_size=0.30, random_state=42)
        tr, te = next(gss.split(X, y, grupos))
        # Si el split deja una sola clase en test, se evalúa sin AUC.
        self.clf.fit(X[tr], y[tr])
        proba_te = self.clf.predict_proba(X[te])[:, 1]
        pred_te = (proba_te >= 0.5).astype(int)

        self.metricas_ = {
            "n_total": int(len(y)),
            "n_train": int(len(tr)),
            "n_test": int(len(te)),
            "n_usuarios": int(len(np.unique(grupos))),
            "tasa_positivos": round(float(y.mean()), 3),
            "accuracy": round(float(accuracy_score(y[te], pred_te)), 3),
            "precision": round(float(precision_score(y[te], pred_te, zero_division=0)), 3),
            "recall": round(float(recall_score(y[te], pred_te, zero_division=0)), 3),
            "roc_auc": (round(float(roc_auc_score(y[te], proba_te)), 3)
                        if len(np.unique(y[te])) > 1 else None),
        }

        # Reentrenar con TODOS los datos para el despliegue.
        self.clf.fit(X, y)
        self.importancias_ = dict(zip(
            FEATURES, self.clf.named_steps["rf"].feature_importances_))
        self._entrenado = True
        return self

    def _nivel(self, prob: float) -> str:
        if prob < 0.15:
            return "Bajo"
        if prob < 0.40:
            return "Medio"
        return "Alto"

    def predecir(self, df: pd.DataFrame, con_montecarlo: bool = True) -> dict:
        """Devuelve la predicción de riesgo para el último mes del usuario `df`."""
        if not self._entrenado:
            return self._resultado_defecto()
        feats = extraer_features(df)
        if feats.empty:
            return self._resultado_defecto()

        X = feats[FEATURES].iloc[[-1]].fillna(0).values
        prob = float(self.clf.predict_proba(X)[0, 1])
        nivel = self._nivel(prob)

        prob_mc = None
        if con_montecarlo:
            try:
                mc = simular_montecarlo(df, meses=self.ventana, n_sim=3000,
                                        umbral_iliquidez=0.0)
                prob_mc = mc.get("prob_iliquidez_horizonte") if mc else None
            except Exception:
                prob_mc = None

        return {
            "entrenado": True,
            "prob_iliquidez": round(prob, 4),
            "prob_iliquidez_mc": round(prob_mc, 4) if prob_mc is not None else None,
            "nivel": nivel,
            "score": int(round(prob * 100)),
            "probabilidades": {"Estable": round((1 - prob) * 100, 1),
                               "En riesgo": round(prob * 100, 1)},
            "factores": self._factores(feats.iloc[-1]),
            "importancias": self.importancias_,
            "features_ultimo": feats.iloc[-1].to_dict(),
            "metricas": self.metricas_,
            "ventana": self.ventana,
        }

    def _factores(self, row: pd.Series) -> list:
        """Factores descriptivos del estado presente (apoyo a la lectura).

        Nota: son una ayuda a la interpretación, NO la señal de entrenamiento.
        La explicación rigurosa por contribuciones llega en la Fase 5 (SHAP).
        """
        factores = []
        rg = row.get("ratio_gasto_ingreso", 0)
        if rg > 0.85:
            factores.append({"factor": "Gastas casi todo lo que ingresas",
                             "valor": f"{rg*100:.0f}% del ingreso", "nivel": "Alto"})
        elif rg > 0.70:
            factores.append({"factor": "Gasto elevado sobre ingresos",
                             "valor": f"{rg*100:.0f}% del ingreso", "nivel": "Medio"})
        ra = row.get("ratio_ahorro", 0)
        if ra < 0.10:
            factores.append({"factor": "Ahorro por debajo del 10% recomendado",
                             "valor": f"{max(ra*100,0):.1f}% de ahorro",
                             "nivel": "Alto" if ra < 0.05 else "Medio"})
        tend = row.get("tendencia_gasto", 0)
        if tend > 30:
            factores.append({"factor": "Tu gasto mensual viene creciendo",
                             "valor": f"+{tend:.0f} €/mes", "nivel": "Medio"})
        if not factores:
            factores.append({"factor": "Situación financiera estable",
                             "valor": "Sin alertas activas", "nivel": "Bajo"})
        return factores

    @staticmethod
    def _resultado_defecto() -> dict:
        return {
            "entrenado": False, "prob_iliquidez": None, "prob_iliquidez_mc": None,
            "nivel": "Desconocido", "score": 0, "probabilidades": {},
            "factores": [{"factor": "Sin datos suficientes", "valor": "—", "nivel": "Bajo"}],
            "importancias": {}, "features_ultimo": {}, "metricas": {}, "ventana": VENTANA_RIESGO,
        }
