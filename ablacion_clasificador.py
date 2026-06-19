#!/usr/bin/env python3
"""
ablacion_clasificador.py
========================
Prueba de ablacion del clasificador de riesgo de iliquidez.

Compara tres condiciones bajo GroupKFold (n=5, identica configuracion a Tabla 5.6):
  A) Regla umbral simple sobre colchon de liquidez   <- baseline trivial
  B) Random Forest sin la variable colchon           <- ablacion
  C) Random Forest completo (modelo de produccion)   <- referencia

Si C supera a A: el modelo aprende interacciones no triviales.
Si C ~ A:        el problema es esencialmente lineal en una variable
                 (hallazgo honesto y defendible).

Ejecutar desde C:\\Users\\paola.clemente\\PREFIN:
    python ablacion_clasificador.py
"""
from __future__ import annotations

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, brier_score_loss
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from fuentes.generator import generar_multiusuario
from modulos.ml_model import FEATURES
from modulos.riesgo_futuro import VENTANA_RIESGO, construir_dataset

SEMILLA = 42
SEED_DATOS = 0
N_SPLITS = 5
np.random.seed(SEMILLA)

SEP = "=" * 65

# ---------------------------------------------------------------------------
# Datos
# ---------------------------------------------------------------------------
print(SEP)
print("ABLACION - Clasificador de riesgo de iliquidez")
print(SEP)
print(f"\nGenerando panel: 40 usuarios x 36 meses (seed={SEED_DATOS})...")
panel = generar_multiusuario(n_usuarios=40, meses=36, seed=SEED_DATOS, realista=True)
X, y, grupos = construir_dataset(panel, VENTANA_RIESGO)

print(f"Dataset: {len(X)} muestras | {len(np.unique(grupos))} usuarios "
      f"| tasa positivos = {y.mean():.3f}")
print(f"Ventana de riesgo: {VENTANA_RIESGO} meses")
print(f"Features ({len(FEATURES)}): {FEATURES}")

idx_colchon = FEATURES.index("colchon_meses")
idx_sin_colchon = [i for i, f in enumerate(FEATURES) if f != "colchon_meses"]

gkf = GroupKFold(n_splits=N_SPLITS)


# ---------------------------------------------------------------------------
# Funciones de evaluacion
# ---------------------------------------------------------------------------

def evaluar_rf(X_data: np.ndarray, etiqueta: str) -> tuple:
    """GroupKFold sobre X_data; devuelve (auc_media, auc_std, brier, folds)."""
    aucs = []
    oof_proba = np.zeros(len(y))
    for tr, te in gkf.split(X_data, y, grupos):
        clf = Pipeline([
            ("sc", StandardScaler()),
            ("rf", RandomForestClassifier(
                n_estimators=300, max_depth=8, min_samples_leaf=5,
                random_state=SEMILLA, class_weight="balanced")),
        ]).fit(X_data[tr], y[tr])
        proba = clf.predict_proba(X_data[te])[:, 1]
        oof_proba[te] = proba
        if len(np.unique(y[te])) > 1:
            aucs.append(roc_auc_score(y[te], proba))
    brier = float(brier_score_loss(y, oof_proba))
    auc_m = float(np.mean(aucs)) if aucs else None
    auc_s = float(np.std(aucs)) if aucs else None
    print(f"  {etiqueta}")
    print(f"    folds AUC : {[round(a, 3) for a in aucs]}")
    print(f"    media={auc_m:.3f}  std={auc_s:.3f}  Brier={brier:.4f}")
    return auc_m, auc_s, brier, aucs


# ---------------------------------------------------------------------------
# A) Baseline trivial: score continuo = -colchon_meses
#    (mayor colchon -> menor riesgo; invertimos para que alto score = alto riesgo)
#    AUC es independiente del umbral, por lo que es la metrica correcta aqui.
# ---------------------------------------------------------------------------
print(f"\n{SEP}")
print("A) Baseline trivial: regla sobre colchon de liquidez")
print(SEP)

colchon_vals = X[:, idx_colchon]
score_trivial = -colchon_vals          # cuanto menor colchon, mayor riesgo predicho

aucs_trivial, oof_trivial = [], np.zeros(len(y))
for tr, te in gkf.split(X, y, grupos):
    oof_trivial[te] = score_trivial[te]
    if len(np.unique(y[te])) > 1:
        aucs_trivial.append(roc_auc_score(y[te], score_trivial[te]))

# Brier con umbral natural colchon < VENTANA_RIESGO
prob_trivial = (colchon_vals < VENTANA_RIESGO).astype(float)
brier_trivial = float(brier_score_loss(y, prob_trivial))

auc_triv_m = float(np.mean(aucs_trivial)) if aucs_trivial else None
auc_triv_s = float(np.std(aucs_trivial)) if aucs_trivial else None
print(f"  folds AUC : {[round(a, 3) for a in aucs_trivial]}")
print(f"  media={auc_triv_m:.3f}  std={auc_triv_s:.3f}  "
      f"Brier={brier_trivial:.4f}  (umbral colchon < {VENTANA_RIESGO})")

# ---------------------------------------------------------------------------
# B) RF sin colchon (ablacion)
# ---------------------------------------------------------------------------
print(f"\n{SEP}")
print("B) RF sin colchon_meses (ablacion)")
print(SEP)
auc_sin_m, auc_sin_s, brier_sin, aucs_sin = evaluar_rf(
    X[:, idx_sin_colchon], "RF sin colchon")

# ---------------------------------------------------------------------------
# C) RF completo (produccion)
# ---------------------------------------------------------------------------
print(f"\n{SEP}")
print("C) RF completo (modelo de produccion)")
print(SEP)
auc_com_m, auc_com_s, brier_com, aucs_com = evaluar_rf(X, "RF completo")

# ---------------------------------------------------------------------------
# Resultados y tabla
# ---------------------------------------------------------------------------
print(f"\n{SEP}")
print("RESUMEN")
print(SEP)

def _fmt(m, s):
    return f"{m:.3f} +/- {s:.3f}" if m is not None else "N/A"

print(f"\n{'Condicion':<42} {'ROC-AUC':>16}  {'Brier':>8}")
print("-" * 68)
print(f"{'A) Umbral colchon (trivial)':<42} {_fmt(auc_triv_m, auc_triv_s):>16}  {brier_trivial:>8.4f}")
print(f"{'B) RF sin colchon (ablacion)':<42} {_fmt(auc_sin_m, auc_sin_s):>16}  {brier_sin:>8.4f}")
print(f"{'C) RF completo (produccion)':<42} {_fmt(auc_com_m, auc_com_s):>16}  {brier_com:>8.4f}")

delta_vs_trivial = (auc_com_m or 0.0) - (auc_triv_m or 0.0)
delta_vs_sin     = (auc_com_m or 0.0) - (auc_sin_m  or 0.0)

print(f"\n  C vs A (RF completo sobre baseline trivial) : delta AUC = {delta_vs_trivial:+.3f}")
print(f"  C vs B (RF completo sobre RF-sin-colchon)   : delta AUC = {delta_vs_sin:+.3f}")

print()
if delta_vs_trivial > 0.01:
    print("=> El RF aprende interacciones no triviales: la discriminacion supera "
          "claramente al umbral simple sobre el colchon.")
elif abs(delta_vs_trivial) <= 0.01:
    print("=> El colchon de liquidez es practicamente suficiente para la "
          "discriminacion. El RF no anade mejora significativa sobre el umbral.")
    print("   Hallazgo honesto: el problema es casi lineal en una variable. "
          "La IA no es decorativa (calibra probabilidades y combina features), "
          "pero la sencillez del patron es un resultado en si mismo.")
else:
    print("=> Resultado inesperado: el RF completo es inferior al umbral trivial. Revisar.")

print(f"\n--- Tabla Markdown lista para copiar ---")
print()
print("| Condicion | ROC-AUC (media +/- std) | Brier |")
print("|-----------|------------------------|-------|")
print(f"| A) Umbral colchon (trivial) | {_fmt(auc_triv_m, auc_triv_s)} | {brier_trivial:.4f} |")
print(f"| B) RF sin colchon (ablacion) | {_fmt(auc_sin_m, auc_sin_s)} | {brier_sin:.4f} |")
print(f"| C) RF completo (produccion) | {_fmt(auc_com_m, auc_com_s)} | {brier_com:.4f} |")
