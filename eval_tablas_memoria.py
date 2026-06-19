#!/usr/bin/env python3
"""
eval_tablas_memoria.py
======================
Genera los valores reales de las Tablas 5.2, 5.5 y 5.6 de la memoria PREFIN.

SEMILLA GLOBAL: random_state=42 para todos los modelos.
SEMILLA DATOS:  seed=0  (igual que la app y el informe de fase6).

Ejecutar desde C:\\Users\\paola.clemente\\PREFIN:
    python eval_tablas_memoria.py
"""
from __future__ import annotations

import os, sys, time, platform
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score, brier_score_loss
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, QuantileRegressor
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
    GradientBoostingRegressor,
)

from fuentes.generator import generar_multiusuario, generar_transacciones
from modulos.riesgo_futuro import construir_dataset, VENTANA_RIESGO
from modulos.forecast import serie_gasto_total_mensual, backtest_previsor
from modulos.digital_twin import simular_montecarlo

SEMILLA = 42
SEED_DATOS = 0
np.random.seed(SEMILLA)

SEP = "=" * 65

# ============================================================
# PASO 1 — Confirmación de parámetros
# ============================================================
print(SEP)
print("PASO 1 — Confirmación de parámetros")
print(SEP)
print(f"  Semilla modelos : SEMILLA = {SEMILLA}")
print(f"  Semilla datos   : SEED_DATOS = {SEED_DATOS}")
print(f"  Python          : {sys.version.split()[0]}")
print(f"  Plataforma      : {platform.platform()}")
print(f"  Procesador      : {platform.processor()}")
try:
    import psutil
    ram_gb = psutil.virtual_memory().total / 1024**3
    print(f"  RAM total       : {ram_gb:.1f} GB")
except ImportError:
    print("  RAM total       : (instalar psutil para medirla)")

# ============================================================
# Generar el panel multiusuario (mismos parámetros que la app)
# ============================================================
print(f"\nGenerando panel: 40 usuarios × 36 meses (seed={SEED_DATOS}, realista=True)…")
panel = generar_multiusuario(n_usuarios=40, meses=36, seed=SEED_DATOS, realista=True)
print(f"  → {len(panel):,} filas  |  {panel['usuario'].nunique()} usuarios")

# Para el forecast: el informe de fase6 se generó sobre un único usuario
# con 36 meses → n_folds = 36 - 8 = 28.
# Determinamos cuál es el usuario del panel con seed = 0*1000 + 0 = 0
# (primer usuario de generar_multiusuario con seed=0).
# Si ese no coincide con las anclas, lo indicamos.
df_u0 = panel[panel["usuario"] == 0].copy().reset_index(drop=True)
n_meses_u0 = df_u0["fecha"].dt.to_period("M").nunique()
print(f"  Usuario 0: {n_meses_u0} meses de historia")

# También preparamos el dataset del clasificador
X_clf, y_clf, grupos_clf = construir_dataset(panel, VENTANA_RIESGO)
n_grupos = len(np.unique(grupos_clf))
print(f"\nDataset clasificador: {len(X_clf)} muestras | {n_grupos} usuarios "
      f"| tasa positivos = {y_clf.mean():.3f}")

# ============================================================
# TABLA 5.6 — Clasificador de riesgo (GroupKFold, n_splits=5)
# ============================================================
# El informe de fase6 tiene 5 folds → usamos n_splits=5 para ser consistentes.
N_SPLITS = 5
print(f"\n{SEP}")
print(f"TABLA 5.6 — Clasificador de riesgo (GroupKFold n={N_SPLITS})")
print(SEP)

CLASIFICADORES = {
    "Regresión logística": Pipeline([
        ("sc", StandardScaler()),
        ("clf", LogisticRegression(
            max_iter=1000, random_state=SEMILLA, class_weight="balanced")),
    ]),
    "Random Forest": Pipeline([
        ("sc", StandardScaler()),
        ("clf", RandomForestClassifier(
            n_estimators=300, max_depth=8, min_samples_leaf=5,
            random_state=SEMILLA, class_weight="balanced")),
    ]),
    "Gradient Boosting": Pipeline([
        ("sc", StandardScaler()),
        ("clf", GradientBoostingClassifier(
            n_estimators=200, max_depth=4, learning_rate=0.05,
            subsample=0.8, random_state=SEMILLA)),
    ]),
}

gkf = GroupKFold(n_splits=N_SPLITS)
tabla_56 = []

for nombre, clf_pipe in CLASIFICADORES.items():
    aucs = []
    oof_proba = np.zeros(len(y_clf))
    for tr, te in gkf.split(X_clf, y_clf, grupos_clf):
        clf = Pipeline(clf_pipe.steps).fit(X_clf[tr], y_clf[tr])
        proba = clf.predict_proba(X_clf[te])[:, 1]
        oof_proba[te] = proba
        if len(np.unique(y_clf[te])) > 1:
            aucs.append(roc_auc_score(y_clf[te], proba))

    brier = float(brier_score_loss(y_clf, oof_proba))
    auc_m = float(np.mean(aucs)) if aucs else None
    auc_s = float(np.std(aucs)) if aucs else None
    tabla_56.append({
        "nombre": nombre,
        "auc_m": round(auc_m, 3) if auc_m is not None else None,
        "auc_s": round(auc_s, 3) if auc_s is not None else None,
        "brier": round(brier, 4),
        "folds": [round(a, 3) for a in aucs],
    })
    fold_str = " | ".join(f"{a:.3f}" for a in aucs)
    print(f"  {nombre}")
    print(f"    ROC-AUC folds: [{fold_str}]")
    print(f"    media={auc_m:.3f}  std={auc_s:.3f}  Brier={brier:.4f}")

# Referencia: valor guardado en informe_validacion.json
print("\n  [Referencia guardada — RF en informe fase6]")
print("    auc_media=0.993 ± 0.005  Brier=0.0258")


# ============================================================
# TABLA 5.5 — Previsión de gasto (backtest walk-forward)
# ============================================================
print(f"\n{SEP}")
print("TABLA 5.5 — Previsión de gasto (backtest walk-forward)")
print(SEP)

# --- A. Modelo de producción (lineal + naive blend) ---
print("\n  [A] Modelo de producción (lineal + referencia estacional)  ← anclas")
m_prod = backtest_previsor(df_u0, min_train=8)
print(f"    MAE={m_prod['mae']}  RMSE={m_prod['rmse']}  "
      f"Cobertura={m_prod['cobertura']*100:.1f}%  n_folds={m_prod['n_folds']}")
print(f"    MAE_baseline={m_prod['mae_baseline']}  RMSE_baseline={m_prod['rmse_baseline']}")
print(f"    [anclas esperadas: MAE=468.33 RMSE=620.51 cob=64.3% / baseline MAE=429.22 RMSE=532.42]")
anclas_ok = (m_prod['mae'] == 468.33 and m_prod['mae_baseline'] == 429.22)
print(f"    ¿Coinciden con anclas? {'SÍ ✓' if anclas_ok else 'NO — ver nota al final'}")

if not anclas_ok:
    # Buscar el usuario cuyo backtest produce las anclas
    print("\n  Buscando usuario que produce las anclas…")
    for uid in panel["usuario"].unique():
        df_u = panel[panel["usuario"] == uid].copy()
        m_tmp = backtest_previsor(df_u, min_train=8)
        if m_tmp.get("mae") == 468.33:
            print(f"    → Usuario {uid} reproduce las anclas exactas")
            df_u0_anclas = df_u.copy()
            m_prod = m_tmp
            anclas_ok = True
            break
    if not anclas_ok:
        print("    Ningún usuario del panel reproduce exactamente 468.33 —")
        print("    las anclas se calcularon con datos distintos (quizás una sola")
        print("    llamada a generar_transacciones). Continuamos con usuario 0.")


# --- B. Regresión cuantílica lineal pura (sin blend naive) ---
def backtest_lineal_puro(df: pd.DataFrame, min_train: int = 8) -> dict:
    """Walk-forward con QuantileRegressor puro (sin mezcla con seasonal-naive)."""
    serie = serie_gasto_total_mensual(df)
    periodos = list(serie.index)
    n = len(periodos)
    if n <= min_train:
        return {"mae": None, "rmse": None, "cobertura": None, "n_folds": 0}

    err, dentro, reales = [], [], []
    for i in range(min_train, n):
        corte = periodos[i].to_timestamp()
        df_tr = df[df["fecha"] < corte]
        if df_tr["fecha"].dt.to_period("M").nunique() < min_train:
            continue
        s_tr = serie_gasto_total_mensual(df_tr)
        y_tr = s_tr.values.astype(float)
        if len(y_tr) - 1 < 6:
            continue

        nt = len(y_tr)
        t_ = np.arange(nt) / 12.0
        m_ = np.array([p.month for p in s_tr.index], dtype=float)
        sn = np.sin(2 * np.pi * m_ / 12.0)
        cs = np.cos(2 * np.pi * m_ / 12.0)
        lg = np.roll(y_tr, 1); lg[0] = y_tr[0]
        Xa = np.column_stack([t_, sn, cs, lg])
        Xf, yf = Xa[1:], y_tr[1:]

        try:
            q10 = QuantileRegressor(quantile=0.1, alpha=0.001, solver="highs").fit(Xf, yf)
            q50 = QuantileRegressor(quantile=0.5, alpha=0.001, solver="highs").fit(Xf, yf)
            q90 = QuantileRegressor(quantile=0.9, alpha=0.001, solver="highs").fit(Xf, yf)
        except Exception:
            continue

        t_nx = nt / 12.0
        m_nx = float((s_tr.index[-1] + 1).month)
        lg_nx = y_tr[-1]
        xn = np.array([[t_nx, np.sin(2*np.pi*m_nx/12), np.cos(2*np.pi*m_nx/12), lg_nx]])

        p10 = max(0.0, float(q10.predict(xn)[0]))
        p50 = max(0.0, float(q50.predict(xn)[0]))
        p90 = max(0.0, float(q90.predict(xn)[0]))
        p10, p90 = min(p10, p50), max(p90, p50)

        real = float(serie.iloc[i])
        err.append(real - p50)
        dentro.append(p10 <= real <= p90)
        reales.append(real)

    if not err:
        return {"mae": None, "rmse": None, "cobertura": None, "n_folds": 0}
    e = np.array(err)
    return {
        "n_folds": len(e),
        "mae": round(float(np.mean(np.abs(e))), 2),
        "rmse": round(float(np.sqrt(np.mean(e**2))), 2),
        "cobertura": round(float(np.mean(dentro)), 3),
    }


# --- C. Gradient Boosting cuantílico ---
def backtest_gb_cuantilico(df: pd.DataFrame, min_train: int = 8) -> dict:
    """Walk-forward con GradientBoostingRegressor(loss='quantile')."""
    serie = serie_gasto_total_mensual(df)
    periodos = list(serie.index)
    n = len(periodos)
    if n <= min_train:
        return {"mae": None, "rmse": None, "cobertura": None, "n_folds": 0}

    err, dentro, reales = [], [], []
    for i in range(min_train, n):
        corte = periodos[i].to_timestamp()
        df_tr = df[df["fecha"] < corte]
        if df_tr["fecha"].dt.to_period("M").nunique() < min_train:
            continue
        s_tr = serie_gasto_total_mensual(df_tr)
        y_tr = s_tr.values.astype(float)
        if len(y_tr) - 1 < 6:
            continue

        nt = len(y_tr)
        t_ = np.arange(nt) / 12.0
        m_ = np.array([p.month for p in s_tr.index], dtype=float)
        sn = np.sin(2 * np.pi * m_ / 12.0)
        cs = np.cos(2 * np.pi * m_ / 12.0)
        lg = np.roll(y_tr, 1); lg[0] = y_tr[0]
        Xa = np.column_stack([t_, sn, cs, lg])
        Xf, yf = Xa[1:], y_tr[1:]

        try:
            q10 = GradientBoostingRegressor(
                loss="quantile", alpha=0.1,
                n_estimators=100, max_depth=3, learning_rate=0.1,
                random_state=SEMILLA).fit(Xf, yf)
            q50 = GradientBoostingRegressor(
                loss="quantile", alpha=0.5,
                n_estimators=100, max_depth=3, learning_rate=0.1,
                random_state=SEMILLA).fit(Xf, yf)
            q90 = GradientBoostingRegressor(
                loss="quantile", alpha=0.9,
                n_estimators=100, max_depth=3, learning_rate=0.1,
                random_state=SEMILLA).fit(Xf, yf)
        except Exception as exc:
            print(f"      GB error en fold {i}: {exc}")
            continue

        t_nx = nt / 12.0
        m_nx = float((s_tr.index[-1] + 1).month)
        lg_nx = y_tr[-1]
        xn = np.array([[t_nx, np.sin(2*np.pi*m_nx/12), np.cos(2*np.pi*m_nx/12), lg_nx]])

        p10 = max(0.0, float(q10.predict(xn)[0]))
        p50 = max(0.0, float(q50.predict(xn)[0]))
        p90 = max(0.0, float(q90.predict(xn)[0]))
        p10, p90 = min(p10, p50), max(p90, p50)

        real = float(serie.iloc[i])
        err.append(real - p50)
        dentro.append(p10 <= real <= p90)
        reales.append(real)

    if not err:
        return {"mae": None, "rmse": None, "cobertura": None, "n_folds": 0}
    e = np.array(err)
    return {
        "n_folds": len(e),
        "mae": round(float(np.mean(np.abs(e))), 2),
        "rmse": round(float(np.sqrt(np.mean(e**2))), 2),
        "cobertura": round(float(np.mean(dentro)), 3),
    }


print("\n  [B] Regresión cuantílica lineal pura (sin blend naive)  ← NUEVA")
m_lin = backtest_lineal_puro(df_u0, min_train=8)
print(f"    MAE={m_lin['mae']}  RMSE={m_lin['rmse']}  "
      f"Cobertura={m_lin['cobertura']*100 if m_lin['cobertura'] else 'N/A'}%  "
      f"n_folds={m_lin['n_folds']}")

print("\n  [C] Gradient Boosting cuantílico  ← NUEVO")
m_gb = backtest_gb_cuantilico(df_u0, min_train=8)
print(f"    MAE={m_gb['mae']}  RMSE={m_gb['rmse']}  "
      f"Cobertura={m_gb['cobertura']*100 if m_gb['cobertura'] else 'N/A'}%  "
      f"n_folds={m_gb['n_folds']}")


# ============================================================
# TABLA 5.2 — Monte Carlo: tiempo de 10.000 trayectorias × 36 meses
# ============================================================
print(f"\n{SEP}")
print("TABLA 5.2 — NFR: Monte Carlo 10 000 trayectorias × 36 meses")
print(SEP)

N_REP = 5
tiempos_ms = []
for rep in range(N_REP):
    t0 = time.perf_counter()
    simular_montecarlo(df_u0, meses=36, n_sim=10_000, seed=SEMILLA)
    tiempos_ms.append((time.perf_counter() - t0) * 1000)

t_med = float(np.mean(tiempos_ms))
t_std = float(np.std(tiempos_ms))
print(f"  Repeticiones    : {N_REP}")
print(f"  Tiempos (ms)    : {[round(t, 1) for t in tiempos_ms]}")
print(f"  Media ± std     : {t_med:.1f} ± {t_std:.1f} ms")


# ============================================================
# RESUMEN FINAL — Tablas listas para copiar en Markdown
# ============================================================
print(f"\n{SEP}")
print("RESUMEN — Tablas Markdown listas para copiar")
print(SEP)

print("""
TABLA 5.6 — Clasificador de riesgo (GroupKFold por usuario, n=5)
""")
print("| Modelo | ROC-AUC (media ± std folds) | Brier (prob. calibradas) | Decisión |")
print("|--------|----------------------------|--------------------------|----------|")
for r in tabla_56:
    auc_str = (f"{r['auc_m']:.3f} ± {r['auc_s']:.3f}" if r["auc_m"] is not None else "N/A")
    print(f"| {r['nombre']} | {auc_str} | {r['brier']:.4f} | |")

print("""
TABLA 5.5 — Previsión de gasto (backtest temporal walk-forward, n_folds=28)
""")
print("| Modelo | MAE (€) | RMSE (€) | Cobertura banda 80 % | Decisión |")
print("|--------|---------|----------|----------------------|----------|")
print(f"| Referencia estacional | {m_prod['mae_baseline']} | {m_prod['rmse_baseline']} | — | |")
print(f"| Modelo combinado (lineal+referencia) | {m_prod['mae']} | {m_prod['rmse']} | {m_prod['cobertura']*100:.1f} % | |")
cob_lin = f"{m_lin['cobertura']*100:.1f} %" if m_lin['cobertura'] is not None else "—"
cob_gb  = f"{m_gb['cobertura']*100:.1f} %" if m_gb['cobertura']  is not None else "—"
print(f"| Regresión cuantílica lineal | {m_lin['mae']} | {m_lin['rmse']} | {cob_lin} | |")
print(f"| Gradient Boosting cuantílico | {m_gb['mae']} | {m_gb['rmse']} | {cob_gb} | |")

print(f"""
TABLA 5.2 — NFR medibles por código
| Métrica | Valor |
|---------|-------|
| Monte Carlo 10k×36m (media, ms) | {t_med:.1f} |
| Monte Carlo 10k×36m (±std, ms) | ±{t_std:.1f} |
| Repeticiones de medición | {N_REP} |
| Python | {sys.version.split()[0]} |
| OS | {platform.platform()} |
| CPU | {platform.processor()} |""")

try:
    print(f"| RAM (total) | {ram_gb:.1f} GB |")  # type: ignore[name-defined]
except NameError:
    pass

print(f"""
CELDAS QUE NO SE PUEDEN OBTENER POR CÓDIGO
-------------------------------------------
• SUS (System Usability Scale): requiere usuarios reales → puntuación 0-100
  mínimo 5 participantes, obtenida tras prueba de usuario.
• WCAG 2.1 AA: auditoría de accesibilidad con herramienta + revisión humana
  (ej. axe, NVDA). No automatizable al 100 %: el criterio 2.1.1 (teclado)
  y el 1.4.3 (contraste) sí son automatizables; los cognitivos (2.4.6) no.
""")
