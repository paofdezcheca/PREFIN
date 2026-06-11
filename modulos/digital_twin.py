# modules/digital_twin.py — Gemelo digital financiero y simulación de escenarios

from datetime import date

import numpy as np
import pandas as pd

from modulos.analyzer import resumen_mensual, gasto_por_categoria_mes

# Percentil para las métricas de riesgo de cola (VaR/CVaR): 95% de confianza.
_NIVEL_VAR = 0.05


# ---------------------------------------------------------------------------
# Estado financiero actual (snapshot del gemelo)
# ---------------------------------------------------------------------------

def estado_actual(df: pd.DataFrame) -> dict:
    """
    Calcula el estado financiero actual del usuario a partir del último mes
    con datos suficientes.
    """
    resumen = resumen_mensual(df)
    cats    = gasto_por_categoria_mes(df)

    if resumen.empty:
        return {}

    # Promedio de los últimos 3 meses (o todos si hay menos)
    n = min(3, len(resumen))
    ultimos = resumen.tail(n)
    cats_ultimos = cats.tail(n) if len(cats) >= n else cats

    ingreso_medio = ultimos["ingreso_total"].mean()
    gasto_medio   = ultimos["gasto_total"].mean()
    ahorro_medio  = ultimos["ahorro_neto"].mean()

    # Gasto por categoría (media de últimos meses)
    cat_medias = {}
    for col in cats_ultimos.columns:
        if col != "mes":
            cat_medias[col] = round(cats_ultimos[col].mean(), 2)

    saldo = df["saldo_acumulado"].iloc[-1] if "saldo_acumulado" in df.columns else 0

    return {
        "saldo_actual":     round(saldo, 2),
        "ingreso_mensual":  round(ingreso_medio, 2),
        "gasto_mensual":    round(gasto_medio, 2),
        "ahorro_mensual":   round(ahorro_medio, 2),
        "tasa_ahorro":      round(ahorro_medio / ingreso_medio * 100 if ingreso_medio else 0, 1),
        "gasto_por_categoria": cat_medias,
    }


# ---------------------------------------------------------------------------
# Simulador de escenarios (gemelo digital)
# ---------------------------------------------------------------------------

def simular_escenario(
    estado: dict,
    meses: int = 12,
    cambio_ingreso_pct: float = 0.0,         # % de cambio en ingresos
    cambios_categoria: dict = None,           # {categoria: % de cambio}
    evento_imprevisto: float = 0.0,           # gasto puntual en mes 1
    meta_ahorro_mensual: float = None,        # objetivo de ahorro mensual
) -> pd.DataFrame:
    """
    Proyecta la situación financiera del gemelo digital durante `meses` meses.

    Parámetros
    ----------
    estado               : resultado de estado_actual()
    meses                : horizonte de simulación (máx 36)
    cambio_ingreso_pct   : variación porcentual del ingreso base (ej: 10 = +10%)
    cambios_categoria    : {categoría: % de cambio} en gastos (ej: {"Ocio": -20})
    evento_imprevisto    : gasto adicional único en el primer mes
    meta_ahorro_mensual  : si se establece, se descuenta del disponible cada mes

    Devuelve un DataFrame con columnas:
    mes, ingreso, gasto, ahorro, saldo_acumulado, escenario
    """
    if not estado:
        return pd.DataFrame()

    cambios_categoria = cambios_categoria or {}
    meses = min(meses, 36)

    ingreso_base  = estado["ingreso_mensual"] * (1 + cambio_ingreso_pct / 100)
    gasto_base    = estado["gasto_mensual"]
    saldo_inicial = estado["saldo_actual"]

    # Aplicar cambios por categoría al gasto base
    cats = estado.get("gasto_por_categoria", {})
    gasto_ajustado = gasto_base
    for cat, pct in cambios_categoria.items():
        if cat in cats:
            gasto_original = cats[cat]
            delta = gasto_original * (pct / 100)
            gasto_ajustado += delta

    filas_actual   = []
    filas_escenario = []
    saldo_act = saldo_inicial
    saldo_esc = saldo_inicial

    for i in range(meses):
        # --- Escenario actual (sin cambios) ---
        ingreso_a = estado["ingreso_mensual"]
        gasto_a   = gasto_base + (np.random.normal(0, gasto_base * 0.03))  # ruido leve
        ahorro_a  = ingreso_a - gasto_a
        saldo_act += ahorro_a
        filas_actual.append({
            "mes": i + 1,
            "ingreso":           round(ingreso_a, 2),
            "gasto":             round(abs(gasto_a), 2),
            "ahorro":            round(ahorro_a, 2),
            "saldo_acumulado":   round(saldo_act, 2),
            "escenario":         "Actual",
        })

        # --- Escenario simulado ---
        ingreso_s = ingreso_base
        gasto_s   = gasto_ajustado + (np.random.normal(0, gasto_ajustado * 0.03))
        if i == 0:
            gasto_s += evento_imprevisto
        if meta_ahorro_mensual:
            # Forzar ahorro mínimo mensual
            gasto_s = min(gasto_s, ingreso_s - meta_ahorro_mensual)
        ahorro_s  = ingreso_s - gasto_s
        saldo_esc += ahorro_s
        filas_escenario.append({
            "mes": i + 1,
            "ingreso":           round(ingreso_s, 2),
            "gasto":             round(abs(gasto_s), 2),
            "ahorro":            round(ahorro_s, 2),
            "saldo_acumulado":   round(saldo_esc, 2),
            "escenario":         "Simulado",
        })

    return pd.concat([
        pd.DataFrame(filas_actual),
        pd.DataFrame(filas_escenario),
    ], ignore_index=True)


def resumen_simulacion(sim_df: pd.DataFrame) -> dict:
    """Calcula el impacto del escenario simulado vs actual."""
    if sim_df.empty:
        return {}

    actual   = sim_df[sim_df["escenario"] == "Actual"]
    simulado = sim_df[sim_df["escenario"] == "Simulado"]

    saldo_final_act = actual["saldo_acumulado"].iloc[-1]
    saldo_final_sim = simulado["saldo_acumulado"].iloc[-1]
    delta_saldo     = saldo_final_sim - saldo_final_act

    ahorro_total_act = actual["ahorro"].sum()
    ahorro_total_sim = simulado["ahorro"].sum()

    return {
        "saldo_final_actual":   round(saldo_final_act, 2),
        "saldo_final_simulado": round(saldo_final_sim, 2),
        "diferencia_saldo":     round(delta_saldo, 2),
        "ahorro_total_actual":  round(ahorro_total_act, 2),
        "ahorro_total_simulado":round(ahorro_total_sim, 2),
        "mejora_ahorro":        round(ahorro_total_sim - ahorro_total_act, 2),
    }


# ===========================================================================
# GEMELO DIGITAL ESTOCÁSTICO — Simulación de Monte Carlo (Fase 2)
# ===========================================================================
#
# A diferencia de `simular_escenario` (una única trayectoria con ruido fijo del
# 3%), el gemelo estocástico:
#   * calibra una distribución por categoría DIRECTAMENTE de los datos del
#     usuario (método de los momentos sobre log-normal: importes no negativos,
#     asimetría a la derecha, típica del gasto),
#   * simula miles de trayectorias de saldo (N≈5.000-10.000),
#   * resume el futuro como bandas de probabilidad p10/p50/p90,
#   * cuantifica la **probabilidad de iliquidez** por mes (P(saldo < umbral)), y
#   * reporta métricas de riesgo de cola tipo **VaR / CVaR** sobre el peor saldo
#     del horizonte.
#
# Todo se hace con numpy vectorizado (sin librerías nuevas): N×meses×categorías
# se resuelve en milisegundos.


def calibrar_perfiles(df: pd.DataFrame) -> dict:
    """Calibra el gemelo a partir del histórico: distribución de ingresos y de
    gasto por categoría, más el saldo inicial.

    Para cada categoría de gasto se estima la media y la desviación típica del
    **total mensual** (no de cada transacción), que es la magnitud que importa
    para la liquidez. Los ingresos se tratan como un agregado mensual.

    Devuelve un diccionario con:
      * `ingreso` : (media, desv) del ingreso mensual,
      * `categorias` : {categoria: (media, desv)} del gasto mensual,
      * `saldo_inicial` : último saldo acumulado conocido,
      * `n_meses` : nº de meses observados.
    """
    if df is None or df.empty:
        return {"ingreso": (0.0, 0.0), "categorias": {}, "saldo_inicial": 0.0, "n_meses": 0}

    trabajo = df.copy()
    trabajo["mes"] = trabajo["fecha"].dt.to_period("M")
    meses = trabajo["mes"].drop_duplicates().sort_values()
    n_meses = len(meses)

    # Ingresos mensuales (suma de importes positivos por mes, reindexado a 0).
    ingresos_mes = (trabajo[trabajo["importe"] > 0]
                    .groupby("mes")["importe"].sum()
                    .reindex(meses, fill_value=0.0))
    ingreso = (float(ingresos_mes.mean()), float(ingresos_mes.std(ddof=0)))

    # Gasto mensual por categoría (importes negativos, en valor absoluto).
    gastos = trabajo[trabajo["importe"] < 0].copy()
    gastos["abs"] = gastos["importe"].abs()
    categorias = {}
    for cat, grupo in gastos.groupby("categoria"):
        serie = grupo.groupby("mes")["abs"].sum().reindex(meses, fill_value=0.0)
        categorias[cat] = (float(serie.mean()), float(serie.std(ddof=0)))

    saldo_inicial = float(df["saldo_acumulado"].iloc[-1]) if "saldo_acumulado" in df.columns else 0.0
    return {"ingreso": ingreso, "categorias": categorias,
            "saldo_inicial": saldo_inicial, "n_meses": n_meses}


def _muestrear_lognormal(media: float, desv: float, tam: tuple, rng) -> np.ndarray:
    """Muestrea de una log-normal ajustada por momentos a (media, desv).

    Garantiza valores no negativos y asimetría realista del gasto. Casos límite:
    media≈0 → ceros; desv≈0 → constante igual a la media.
    """
    if media <= 1e-9:
        return np.zeros(tam)
    if desv <= 1e-9:
        return np.full(tam, media)
    sigma2 = np.log(1.0 + (desv ** 2) / (media ** 2))
    mu = np.log(media) - 0.5 * sigma2
    return rng.lognormal(mean=mu, sigma=np.sqrt(sigma2), size=tam)


def simular_montecarlo(
    df: pd.DataFrame,
    meses: int = 12,
    n_sim: int = 5000,
    cambio_ingreso_pct: float = 0.0,
    cambios_categoria: dict = None,
    evento_imprevisto: float = 0.0,
    mes_evento: int = 1,
    meta_ahorro_mensual: float = None,
    ahorro_extra_mensual: float = 0.0,
    umbral_iliquidez: float = 0.0,
    seed: int = 42,
) -> dict:
    """Simula `n_sim` trayectorias estocásticas del saldo durante `meses`.

    Palancas (compatibles con la pestaña de simulación):
      * `cambio_ingreso_pct`   : variación porcentual del ingreso.
      * `cambios_categoria`    : {categoria: % de cambio} sobre el gasto medio.
      * `evento_imprevisto`    : gasto puntual añadido en `mes_evento`.
      * `meta_ahorro_mensual`  : si se fija, limita el gasto para garantizar ese
                                 ahorro mínimo cada mes.
      * `ahorro_extra_mensual` : importe que se aparta cada mes (lo usa el motor
                                 prescriptivo de la Fase 3; reduce el saldo
                                 disponible, no cuenta como iliquidez negativa).

    Devuelve un diccionario con:
      * `bandas`        : DataFrame[mes, p10, p50, p90, prob_iliquidez],
      * `trayectorias`  : muestra de trayectorias para el cono (ndarray k×meses),
      * `saldo_inicial`, `n_sim`, `umbral`,
      * `var_95`        : percentil 5 del PEOR saldo del horizonte (VaR),
      * `cvar_95`       : media del 5% de peores casos (Expected Shortfall),
      * `prob_iliquidez_horizonte` : P(tocar iliquidez en algún mes),
      * `saldo_final_esperado`, `ahorro_total_esperado`.
    """
    perfiles = calibrar_perfiles(df)
    if perfiles["n_meses"] == 0:
        return {}

    meses = int(np.clip(meses, 1, 60))
    n_sim = int(np.clip(n_sim, 100, 50000))
    cambios_categoria = cambios_categoria or {}
    rng = np.random.default_rng(seed)
    tam = (n_sim, meses)

    # --- Ingresos ---
    ing_media, ing_desv = perfiles["ingreso"]
    ing_media *= (1 + cambio_ingreso_pct / 100.0)
    ingresos = rng.normal(ing_media, max(ing_desv, 1e-9), size=tam)
    ingresos = np.clip(ingresos, 0.0, None)

    # --- Gastos por categoría ---
    gastos = np.zeros(tam)
    for cat, (media, desv) in perfiles["categorias"].items():
        factor = 1 + cambios_categoria.get(cat, 0) / 100.0
        gastos += _muestrear_lognormal(media * factor, desv * abs(factor), tam, rng)

    # Evento puntual en el mes indicado (1-based).
    if evento_imprevisto and 1 <= mes_evento <= meses:
        gastos[:, mes_evento - 1] += evento_imprevisto

    # Ahorro forzado: limita el gasto para asegurar la meta mensual.
    if meta_ahorro_mensual:
        tope_gasto = np.clip(ingresos - meta_ahorro_mensual, 0.0, None)
        gastos = np.minimum(gastos, tope_gasto)

    # Flujo neto y saldo acumulado. El ahorro extra se aparta (sale del saldo
    # disponible cada mes, simulando un traspaso a una hucha separada).
    neto = ingresos - gastos - ahorro_extra_mensual
    saldo = perfiles["saldo_inicial"] + np.cumsum(neto, axis=1)

    # --- Bandas de probabilidad por mes ---
    p10 = np.percentile(saldo, 10, axis=0)
    p50 = np.percentile(saldo, 50, axis=0)
    p90 = np.percentile(saldo, 90, axis=0)
    prob_iliq_mes = np.mean(saldo < umbral_iliquidez, axis=0)

    bandas = pd.DataFrame({
        "mes": np.arange(1, meses + 1),
        "p10": np.round(p10, 2),
        "p50": np.round(p50, 2),
        "p90": np.round(p90, 2),
        "prob_iliquidez": np.round(prob_iliq_mes, 4),
    })

    # --- Riesgo de cola sobre el PEOR saldo del horizonte ---
    peor_saldo = saldo.min(axis=1)
    var_95 = float(np.percentile(peor_saldo, _NIVEL_VAR * 100))
    cola = peor_saldo[peor_saldo <= var_95]
    cvar_95 = float(cola.mean()) if cola.size else var_95
    prob_iliq_horizonte = float(np.mean(peor_saldo < umbral_iliquidez))

    # Muestra de trayectorias para el cono (máx 200, sin re-muestrear).
    k = min(200, n_sim)
    trayectorias = saldo[:k]

    return {
        "bandas": bandas,
        "trayectorias": trayectorias,
        "saldo_inicial": round(perfiles["saldo_inicial"], 2),
        "n_sim": n_sim,
        "umbral": umbral_iliquidez,
        "var_95": round(var_95, 2),
        "cvar_95": round(cvar_95, 2),
        "prob_iliquidez_horizonte": round(prob_iliq_horizonte, 4),
        "saldo_final_esperado": round(float(saldo[:, -1].mean()), 2),
        "ahorro_total_esperado": round(float(neto.sum(axis=1).mean()), 2),
    }


def figura_cono_montecarlo(mc: dict, titulo: str = "Proyección Monte Carlo del saldo"):
    """Cono de trayectorias de Monte Carlo (elemento visual protagonista).

    Dibuja una muestra de trayectorias tenues, la banda p10–p90, la mediana y la
    línea de iliquidez. Importa Plotly de forma perezosa.
    """
    import plotly.graph_objects as go
    try:
        from config import PLOTLY_LAYOUT, PREFIN_INK, PREFIN_VERDE, PREFIN_ROJO
    except Exception:  # pragma: no cover
        PLOTLY_LAYOUT, PREFIN_INK, PREFIN_VERDE, PREFIN_ROJO = {}, "#0F172A", "#16A34A", "#DC2626"

    if not mc:
        return go.Figure()

    bandas = mc["bandas"]
    x = bandas["mes"].tolist()
    fig = go.Figure()

    # Trayectorias individuales (tenues): transmiten la incertidumbre real.
    tray = mc["trayectorias"]
    paso = max(1, tray.shape[0] // 60)  # como mucho ~60 líneas para no saturar
    for i in range(0, tray.shape[0], paso):
        fig.add_scatter(x=x, y=tray[i], mode="lines",
                        line=dict(color="rgba(15,23,42,0.06)", width=1),
                        hoverinfo="skip", showlegend=False)

    # Banda p10–p90.
    fig.add_scatter(x=x, y=bandas["p90"], mode="lines", line=dict(width=0),
                    hoverinfo="skip", showlegend=False)
    fig.add_scatter(x=x, y=bandas["p10"], mode="lines", line=dict(width=0),
                    fill="tonexty", fillcolor="rgba(15,23,42,0.15)",
                    name="Banda p10–p90", hoverinfo="skip")
    # Mediana.
    fig.add_scatter(x=x, y=bandas["p50"], mode="lines",
                    line=dict(color=PREFIN_INK, width=2.5), name="Saldo mediano")
    # Línea de iliquidez.
    fig.add_hline(y=mc["umbral"], line=dict(color=PREFIN_ROJO, width=1.5, dash="dot"),
                  annotation_text="Iliquidez", annotation_position="bottom right")

    fig.update_layout(**PLOTLY_LAYOUT)
    fig.update_layout(title_text=titulo, xaxis_title="Mes", yaxis_title="Saldo (€)")
    return fig


def exportar_cono_png(df: pd.DataFrame, ruta: str, meses: int = 12,
                      n_sim: int = 5000, **kwargs) -> str:
    """Exporta el cono de Monte Carlo a PNG estático (matplotlib) para la memoria."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    mc = simular_montecarlo(df, meses=meses, n_sim=n_sim, **kwargs)
    if not mc:
        raise ValueError("Sin datos suficientes para simular.")
    bandas = mc["bandas"]
    x = bandas["mes"].values
    tray = mc["trayectorias"]

    fig, ax = plt.subplots(figsize=(10, 5), dpi=150)
    paso = max(1, tray.shape[0] // 60)
    for i in range(0, tray.shape[0], paso):
        ax.plot(x, tray[i], color="#0F172A", alpha=0.05, linewidth=0.8)
    ax.fill_between(x, bandas["p10"], bandas["p90"], color="#0F172A", alpha=0.15,
                    label="Banda p10–p90")
    ax.plot(x, bandas["p50"], color="#0F172A", linewidth=2.5, label="Saldo mediano")
    ax.axhline(mc["umbral"], color="#DC2626", linestyle=":", linewidth=1.5,
               label="Iliquidez")
    ax.set_title("Proyección Monte Carlo del saldo", loc="left")
    ax.set_xlabel("Mes"); ax.set_ylabel("Saldo (€)")
    ax.legend(frameon=False, fontsize=9)
    ax.grid(axis="y", color="#E7E5E4", linewidth=0.8)
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    fig.savefig(ruta)
    plt.close(fig)
    return ruta
