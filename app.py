# app.py — PREFIN: Plataforma Inteligente de Predicción y Prevención Financiera
# Interfaz Dash con estética minimalista

import sys
import os
import re
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import dash
from dash import dcc, html, Input, Output, State, callback_context, dash_table
import dash_bootstrap_components as dbc

from config import (
    COLORES_CATEGORIA, COLOR_RIESGO, ESCALA_INTENSIDAD,
    PREFIN_INK, PREFIN_FONDO, PREFIN_SUPERFICIE, PREFIN_BORDE,
    PREFIN_TEXTO_SEC, PREFIN_TEXTO_MUTED, PREFIN_VERDE, PREFIN_ROJO, PREFIN_AMBAR,
    PREFIN_ACENTO, PLOTLY_LAYOUT,
)
from fuentes.loader import cargar_desde_upload, cargar_sinteticos, cargar_desde_truelayer
from modulos.analyzer import (
    resumen_mensual, gasto_por_categoria_mes,
    kpis_globales, detectar_anomalias, tendencia_gasto, gasto_diario_semana,
)
from modulos.riesgo_futuro import ModeloRiesgoFuturo
from fuentes.generator import generar_multiusuario
from modulos.forecast import PrevisorGasto, figura_prevision
from modulos.explicador import explicar_natural, figura_contribuciones
from modulos.deteccion import (
    detectar_cambios_regimen, figura_cambios_regimen, comparar_detectores,
)
from modulos.digital_twin import (
    estado_actual, simular_escenario, resumen_simulacion,
    simular_montecarlo, figura_cono_montecarlo,
)
from modulos.prescriptor import optimizar_plan
from modulos.microsavings import (
    resumen_microahorro, objetivos_ahorro, microahorro_por_categoria, OPCIONES_REDONDEO,
)

# ============================================================
# INICIALIZACIÓN
# ============================================================
app = dash.Dash(
    __name__,
    external_stylesheets=[
        dbc.themes.BOOTSTRAP,
        dbc.icons.BOOTSTRAP,
    ],
    suppress_callback_exceptions=True,
    title="PREFIN",
    meta_tags=[{"name": "viewport", "content": "width=device-width, initial-scale=1"}],
)
server = app.server

# Favicon SVG (el propio logo) + meta, preservando los marcadores de Dash.
app.index_string = """<!DOCTYPE html>
<html>
    <head>
        {%metas%}
        <title>{%title%}</title>
        <link rel="icon" type="image/svg+xml" href="/assets/logo.svg">
        {%favicon%}
        {%css%}
    </head>
    <body>
        {%app_entry%}
        <footer>{%config%}{%scripts%}{%renderer%}</footer>
    </body>
</html>"""

# Modelo de riesgo des-circularizado: se entrena UNA vez sobre una población de
# usuarios sintéticos y luego se aplica al usuario cargado. El entrenamiento es
# perezoso (en la primera predicción) para no ralentizar el arranque.
modelo = ModeloRiesgoFuturo()
_POBLACION = {"entrenada": False}


def _asegurar_modelo():
    """Entrena el modelo de riesgo poblacional la primera vez que se necesita."""
    if not _POBLACION["entrenada"]:
        try:
            panel = generar_multiusuario(n_usuarios=40, meses=36, seed=0, realista=True)
            modelo.entrenar(panel)
            _POBLACION["entrenada"] = modelo._entrenado
        except Exception:
            pass
    return modelo


# Cache en memoria del servidor (no en el navegador)
_CACHE = {"df": None}


# ============================================================
# COMPONENTES REUTILIZABLES
# ============================================================

def aplicar_layout_plotly(fig):
    """Aplica el layout minimalista por defecto a una figura Plotly."""
    fig.update_layout(**PLOTLY_LAYOUT)
    return fig


def kpi_card(titulo, valor, icono, color=PREFIN_INK, subtitulo="", ayuda=""):
    """Tarjeta KPI minimalista. `ayuda` añade un icono con tooltip explicativo."""
    label_children = [titulo]
    extra = []
    if ayuda:
        tip_id = "tip-" + re.sub(r"[^a-z0-9]+", "-", titulo.lower()).strip("-")
        label_children.append(
            html.I(className="bi bi-info-circle help-icon", id=tip_id))
        extra.append(dbc.Tooltip(ayuda, target=tip_id, placement="top"))
    return html.Div([
        html.Div([
            html.Div(
                html.I(className=f"bi {icono}", style={"color": color}),
                className="kpi-icon",
                style={"backgroundColor": f"{color}14"},  # 8% opacidad
            ),
            html.Div([
                html.Div(label_children, className="kpi-label"),
                html.Div(valor, className="kpi-value"),
                html.Div(subtitulo, className="kpi-sub") if subtitulo else html.Span(),
            ], style={"flex": "1", "marginLeft": "0.85rem"}),
        ], style={"display": "flex", "alignItems": "flex-start"}),
        *extra,
    ], className="kpi-card")


def _metrica_fila(nombre, valor):
    """Fila 'nombre … valor' para tarjetas de métricas (números tabulares)."""
    if isinstance(valor, bool) or valor is None:
        txt = "—"
    elif isinstance(valor, int):
        txt = f"{valor:,}"
    elif isinstance(valor, float):
        txt = f"{valor:.2f}"
    else:
        txt = "—"
    return html.Div([
        html.Span(nombre, style={"color": PREFIN_TEXTO_SEC, "fontSize": "0.85rem"}),
        html.Strong(txt, style={"marginLeft": "auto",
                                "fontVariantNumeric": "tabular-nums"}),
    ], style={"display": "flex", "padding": "5px 0",
              "borderBottom": f"1px solid {PREFIN_BORDE}"})


def badge_riesgo(nivel):
    """Badge sutil con el nivel de riesgo."""
    colores = {
        "Bajo":   ("#F0FDF4", "#14532D", "#BBF7D0"),
        "Medio":  ("#FFFBEB", "#78350F", "#FDE68A"),
        "Alto":   ("#FEF2F2", "#7F1D1D", "#FECACA"),
    }
    bg, fg, border = colores.get(nivel, ("#F4F4F5", PREFIN_TEXTO_SEC, PREFIN_BORDE))
    return html.Span(
        nivel,
        style={
            "backgroundColor": bg, "color": fg, "border": f"1px solid {border}",
            "padding": "3px 10px", "borderRadius": "6px",
            "fontSize": "0.78rem", "fontWeight": "500",
        },
    )


def alerta_riesgo_banner(nivel):
    """Banner discreto que indica el nivel de riesgo actual."""
    color_map = {
        "Bajo":   ("#F0FDF4", "#14532D", "#BBF7D0", "bi-shield-check"),
        "Medio":  ("#FFFBEB", "#78350F", "#FDE68A", "bi-shield-exclamation"),
        "Alto":   ("#FEF2F2", "#7F1D1D", "#FECACA", "bi-shield-x"),
    }
    bg, fg, border, icono = color_map.get(nivel, ("#F4F4F5", PREFIN_TEXTO_SEC, PREFIN_BORDE, "bi-shield"))
    return html.Div([
        html.I(className=f"bi {icono} me-2", style={"fontSize": "1.05rem"}),
        html.Span("Nivel de riesgo financiero: ", style={"fontWeight": "500"}),
        html.Strong(nivel),
    ], style={
        "backgroundColor": bg, "color": fg, "border": f"1px solid {border}",
        "padding": "0.65rem 1rem", "borderRadius": "8px", "fontSize": "0.9rem",
    })


# ============================================================
# NAVBAR
# ============================================================
NAVBAR = dbc.Navbar(
    dbc.Container([
        html.A(
            html.Div([
                html.Img(src="/assets/logo.svg", className="prefin-logo-img",
                         alt="Logo PREFIN"),
                html.Span("PREFIN", className="navbar-brand mb-0 prefin-brand-display"),
            ], style={"display": "flex", "alignItems": "center"}),
            href="/", style={"textDecoration": "none"},
        ),
        dbc.Nav([
            dbc.NavItem(dbc.NavLink([html.I(className="bi bi-grid me-1"), "Dashboard"],
                                    href="/", active="exact")),
            dbc.NavItem(dbc.NavLink([html.I(className="bi bi-bar-chart me-1"), "Análisis"],
                                    href="/analisis", active="exact")),
            dbc.NavItem(dbc.NavLink([html.I(className="bi bi-cpu me-1"), "Riesgo ML"],
                                    href="/riesgo", active="exact")),
            dbc.NavItem(dbc.NavLink([html.I(className="bi bi-diagram-3 me-1"), "Gemelo Digital"],
                                    href="/simulacion", active="exact")),
            dbc.NavItem(dbc.NavLink([html.I(className="bi bi-stars me-1"), "Mi Plan"],
                                    href="/plan", active="exact")),
            dbc.NavItem(dbc.NavLink([html.I(className="bi bi-piggy-bank me-1"), "Micro-Ahorro"],
                                    href="/ahorro", active="exact")),
            dbc.NavItem(dbc.NavLink([html.I(className="bi bi-database me-1"), "Datos"],
                                    href="/datos", active="exact")),
        ], navbar=True, className="ms-auto"),
    ], fluid=True),
    className="prefin-navbar mb-0",
    expand="lg",
)

# ============================================================
# LAYOUT PRINCIPAL
# ============================================================
app.layout = html.Div([
    dcc.Location(id="url", refresh=False),
    dcc.Store(id="store-df", data=None),
    NAVBAR,
    dcc.Loading(
        html.Div(id="page-content", style={
            "backgroundColor": PREFIN_FONDO,
            "minHeight": "calc(100vh - 60px)",
            "paddingBottom": "60px",
        }),
        type="default", color=PREFIN_INK,
    ),
    dbc.Toast(
        id="toast-notif",
        header="PREFIN",
        is_open=False,
        dismissable=True,
        duration=4000,
        style={"position": "fixed", "top": 80, "right": 20, "zIndex": 9999,
               "minWidth": "280px"},
    ),
])


# ============================================================
# AUXILIAR: pantalla vacía
# ============================================================
def _pantalla_sin_datos():
    return dbc.Container([
        html.Div([
            html.Div(html.I(className="bi bi-bar-chart-line"),
                     className="estado-vacio-icono"),
            html.H4("Empieza cargando tus datos", className="mt-4",
                    style={"color": PREFIN_INK}),
            html.P("Genera un conjunto realista en un clic, sube tu extracto o "
                   "conecta tu banco. Después podrás explorar previsiones, riesgo "
                   "y tu plan de ahorro.",
                   style={"color": PREFIN_TEXTO_SEC, "fontSize": "0.95rem",
                          "maxWidth": "440px", "margin": "0.5rem auto 0"}),
            dbc.Button([html.I(className="bi bi-arrow-right me-2"), "Ir a Datos"],
                       href="/datos", color="primary", className="mt-3"),
        ], className="text-center", style={"padding": "5rem 1rem"}),
    ], fluid=True, className="px-4")


# ============================================================
# PÁGINA: DATOS
# ============================================================
def layout_datos():
    return dbc.Container([
        html.Div([
            html.H1("Fuente de datos", className="page-title"),
            html.P("Genera datos sintéticos, sube un extracto bancario o conecta tu banco vía Open Banking.",
                   className="page-subtitle"),
        ], className="pt-4"),

        # Modo demo: un clic para una demostración realista (sin banco real).
        dbc.Alert([
            html.Div([
                html.Div([
                    html.I(className="bi bi-stars me-2"),
                    html.Strong("¿Primera vez? Prueba el modo demo."),
                    html.Span("  Cargamos 36 meses de un usuario realista para que "
                              "explores toda la app al instante.",
                              style={"fontSize": "0.88rem"}),
                ]),
                dbc.Button([html.I(className="bi bi-play-fill me-2"),
                            "Cargar demo"],
                           id="btn-demo", color="primary", size="sm", n_clicks=0,
                           className="ms-3", style={"whiteSpace": "nowrap"}),
            ], style={"display": "flex", "alignItems": "center",
                      "justifyContent": "space-between", "flexWrap": "wrap",
                      "gap": "0.5rem"}),
        ], color="light", className="mb-4"),

        dbc.Row([
            # --- Datos sintéticos ---
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader([
                        html.I(className="bi bi-magic me-2"),
                        "Datos sintéticos",
                    ]),
                    dbc.CardBody([
                        html.P("Crea un histórico bancario ficticio realista para probar la plataforma.",
                               style={"color": PREFIN_TEXTO_SEC, "fontSize": "0.88rem"}),

                        dbc.Label("Meses de historia"),
                        dcc.Slider(3, 36, 3, value=12, id="sl-meses",
                                   marks={i: str(i) for i in range(3, 37, 3)}),

                        dbc.Label("Nómina mensual (€)", className="mt-3"),
                        dbc.Input(id="inp-nomina", type="number", value=1800,
                                  min=800, max=6000, step=50),

                        dbc.Label("Perfil de gasto", className="mt-3"),
                        dbc.RadioItems(
                            id="radio-perfil",
                            options=[
                                {"label": "Austero", "value": "bajo"},
                                {"label": "Estándar", "value": "medio"},
                                {"label": "Elevado", "value": "alto"},
                            ],
                            value="medio", inline=True,
                        ),

                        dbc.Button(
                            [html.I(className="bi bi-play-circle me-2"), "Generar datos"],
                            id="btn-sintetico", color="primary",
                            className="mt-4 w-100", n_clicks=0,
                        ),
                    ]),
                ], className="h-100"),
            ], md=4, className="mb-3"),

            # --- Subir CSV / Excel ---
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader([
                        html.I(className="bi bi-cloud-upload me-2"),
                        "Subir archivo CSV / Excel",
                    ]),
                    dbc.CardBody([
                        html.P("Sube tu extracto bancario en formato .csv, .xls o .xlsx.",
                               style={"color": PREFIN_TEXTO_SEC, "fontSize": "0.88rem"}),
                        html.P([
                            "Columnas mínimas: ",
                            html.Code("fecha"), ", ",
                            html.Code("descripcion"), ", ",
                            html.Code("importe"),
                            ". Importes negativos = gastos.",
                        ], style={"color": PREFIN_TEXTO_SEC, "fontSize": "0.8rem"}),

                        dcc.Upload(
                            id="upload-data",
                            children=html.Div([
                                html.I(className="bi bi-cloud-arrow-up",
                                       style={"fontSize": "1.8rem",
                                              "color": PREFIN_TEXTO_SEC}),
                                html.Div("Arrastra el archivo o haz click",
                                         style={"color": PREFIN_TEXTO_SEC,
                                                "fontSize": "0.88rem",
                                                "marginTop": "8px"}),
                            ]),
                            className="upload-area",
                            style={
                                "width": "100%", "height": "140px",
                                "display": "flex", "alignItems": "center",
                                "justifyContent": "center",
                                "flexDirection": "column",
                            },
                            multiple=False,
                        ),
                        html.Div(id="upload-status",
                                 className="mt-2",
                                 style={"fontSize": "0.85rem"}),
                    ]),
                ], className="h-100"),
            ], md=4, className="mb-3"),

            # --- TrueLayer ---
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader([
                        html.I(className="bi bi-bank me-2"),
                        "Conectar banco (TrueLayer)",
                    ]),
                    dbc.CardBody([
                        html.P("Conecta tu cuenta bancaria real vía Open Banking (OAuth 2.0, sandbox).",
                               style={"color": PREFIN_TEXTO_SEC, "fontSize": "0.88rem"}),

                        dbc.Alert([
                            html.I(className="bi bi-info-circle me-2"),
                            "Requiere backend FastAPI en localhost:8000.",
                        ], color="info", className="small"),

                        dbc.Label("Mes a consultar (YYYY-MM)"),
                        dbc.Input(id="inp-mes-tl", type="text",
                                  placeholder="2026-01", value=""),

                        dbc.Button(
                            [html.I(className="bi bi-box-arrow-up-right me-2"),
                             "Iniciar conexión OAuth"],
                            id="btn-open-connect", color="secondary",
                            className="mt-3 w-100",
                            href="http://localhost:8000/connect",
                            target="_blank", external_link=True,
                        ),
                        dbc.Button(
                            [html.I(className="bi bi-download me-2"),
                             "Cargar transacciones"],
                            id="btn-truelayer", color="primary",
                            className="mt-2 w-100", n_clicks=0,
                        ),
                    ]),
                ], className="h-100"),
            ], md=4, className="mb-3"),
        ]),

        html.Div(id="datos-feedback", className="mt-3"),
    ], fluid=True, className="px-4")


# ============================================================
# Listas de transacciones (diseño "feed": avatar + categoría + importe)
# ============================================================
ICONO_CATEGORIA = {
    "Supermercado": "bi-cart3", "Restaurantes y Ocio": "bi-cup-straw",
    "Transporte": "bi-bus-front", "Servicios del Hogar": "bi-house-door",
    "Suscripciones": "bi-arrow-repeat", "Salud y Farmacia": "bi-heart-pulse",
    "Ropa y Compras": "bi-bag", "Educación": "bi-mortarboard",
    "Transferencias": "bi-arrow-left-right", "Ingresos": "bi-cash-coin",
    "Otros": "bi-three-dots",
}


def _fila_tx(r, *, avatar_icono=None, avatar_color=None, extra=None):
    """Una fila tipo 'feed': avatar de categoría + descripción/categoría + extra + importe."""
    cat = r["categoria"]
    col = COLORES_CATEGORIA.get(cat, PREFIN_TEXTO_SEC)
    es_ingreso = r["importe"] >= 0
    col_imp = PREFIN_VERDE if es_ingreso else PREFIN_ROJO
    icono = avatar_icono or ICONO_CATEGORIA.get(cat, "bi-tag")
    color_av = avatar_color or col
    hijos = [
        html.Div(html.I(className=f"bi {icono}"), className="tx-avatar",
                 style={"backgroundColor": color_av + "1A", "color": color_av}),
        html.Div([
            html.Div(r["descripcion"], className="tx-desc"),
            html.Div(cat, className="tx-cat", style={"color": col}),
        ], className="tx-main"),
    ]
    if extra is not None:
        hijos.append(extra)
    hijos.append(html.Div(r["fecha"].strftime("%d/%m/%Y"), className="tx-fecha"))
    hijos.append(html.Div(f"{r['importe']:+,.2f} €", className="tx-importe",
                          style={"color": col_imp}))
    return html.Div(hijos, className="tx-row")


def _tabla_transacciones(df, n=12):
    filas = df.tail(n).iloc[::-1]
    return html.Div([_fila_tx(r) for _, r in filas.iterrows()], className="tx-list")


def _tabla_anomalias(df_anom):
    filas = []
    for _, r in df_anom.iterrows():
        sigma = html.Span(f"{r['z_score']:.1f}σ", className="tx-sigma")
        filas.append(_fila_tx(r, avatar_icono="bi-exclamation-triangle",
                              avatar_color=PREFIN_AMBAR, extra=sigma))
    return html.Div(filas, className="tx-list")


# ============================================================
# PÁGINA: DASHBOARD
# ============================================================
def layout_dashboard(df):
    if df is None or df.empty:
        return _pantalla_sin_datos()

    kpis = kpis_globales(df)
    pred = _asegurar_modelo().predecir(df)
    tend = tendencia_gasto(df)

    # --- Gráfica saldo ---
    fig_saldo = px.line(
        df.dropna(subset=["saldo_acumulado"]),
        x="fecha", y="saldo_acumulado",
        title="Evolución del saldo",
        labels={"saldo_acumulado": "Saldo (€)", "fecha": ""},
        color_discrete_sequence=[PREFIN_INK],
    )
    fig_saldo.update_traces(line=dict(width=2))
    aplicar_layout_plotly(fig_saldo)

    # --- Donut categorías último mes ---
    ultimo_mes = df["fecha"].dt.to_period("M").max()
    df_ult = df[(df["fecha"].dt.to_period("M") == ultimo_mes) & (df["importe"] < 0)].copy()
    df_ult["importe_abs"] = df_ult["importe"].abs()
    cat_ult = df_ult.groupby("categoria")["importe_abs"].sum().reset_index()
    fig_cat = px.pie(
        cat_ult, names="categoria", values="importe_abs",
        title=f"Distribución de gasto · {str(ultimo_mes)}",
        color="categoria",
        color_discrete_map=COLORES_CATEGORIA,
        hole=0.55,
    )
    fig_cat.update_traces(textposition="outside", textinfo="percent",
                          marker=dict(line=dict(color="white", width=2)))
    aplicar_layout_plotly(fig_cat)
    fig_cat.update_layout(showlegend=True,
                           legend=dict(orientation="v", y=0.5, x=1.05, font=dict(size=10)))

    return dbc.Container([
        html.Div([
            html.H1("Dashboard", className="page-title"),
            html.P(f"{len(df):,} transacciones · {tend['interpretacion']}",
                   className="page-subtitle"),
        ], className="pt-4"),

        # KPIs
        dbc.Row([
            dbc.Col(kpi_card("Saldo actual", f"{kpis['saldo_actual']:,.2f} €",
                             "bi-wallet2", PREFIN_INK), md=3, className="mb-3"),
            dbc.Col(kpi_card("Ingreso mensual medio", f"{kpis['ingreso_mensual_medio']:,.2f} €",
                             "bi-arrow-up-right", PREFIN_VERDE), md=3, className="mb-3"),
            dbc.Col(kpi_card("Gasto mensual medio", f"{kpis['gasto_mensual_medio']:,.2f} €",
                             "bi-arrow-down-right", PREFIN_ROJO), md=3, className="mb-3"),
            dbc.Col(kpi_card("Tasa de ahorro", f"{kpis['tasa_ahorro_media']:.1f}%",
                             "bi-piggy-bank", PREFIN_AMBAR,
                             "Recomendado: >15%",
                             ayuda="Parte de tus ingresos que NO gastas, de media. "
                                   "Por encima del 15% se considera saludable."),
                    md=3, className="mb-3"),
        ]),

        # Banner de riesgo
        dbc.Row([
            dbc.Col(alerta_riesgo_banner(pred["nivel"]), md=12),
        ], className="mb-3"),

        # Gráficas
        dbc.Row([
            dbc.Col(dbc.Card(dbc.CardBody(dcc.Graph(figure=fig_saldo,
                                                    config={"displayModeBar": False}))),
                    md=8, className="mb-3"),
            dbc.Col(dbc.Card(dbc.CardBody(dcc.Graph(figure=fig_cat,
                                                    config={"displayModeBar": False}))),
                    md=4, className="mb-3"),
        ]),

        # Tabla
        dbc.Card([
            dbc.CardHeader([html.I(className="bi bi-clock-history me-2"),
                             "Últimas transacciones"]),
            dbc.CardBody(_tabla_transacciones(df), style={"overflowX": "auto"}),
        ]),
    ], fluid=True, className="px-4")


# ============================================================
# PÁGINA: ANÁLISIS
# ============================================================
def layout_analisis(df):
    if df is None or df.empty:
        return _pantalla_sin_datos()

    cats_mes = gasto_por_categoria_mes(df)
    res = resumen_mensual(df)
    anomalias_df = detectar_anomalias(df)
    diario = gasto_diario_semana(df)

    # Prevención (Fase 4): cambios de régimen y comparación de detectores.
    cambios_reg = detectar_cambios_regimen(df)
    fig_regimen = figura_cambios_regimen(df)
    comp_det = comparar_detectores(df)

    # --- Evolución gasto por categoría (apilado) ---
    cats_long = cats_mes.melt(id_vars="mes", var_name="categoria", value_name="gasto")
    cats_long = cats_long[cats_long["gasto"] > 0]
    fig_evol = px.bar(
        cats_long, x="mes", y="gasto", color="categoria",
        title="Gasto mensual por categoría",
        labels={"gasto": "Gasto (€)", "mes": ""},
        color_discrete_map=COLORES_CATEGORIA,
        barmode="stack",
    )
    aplicar_layout_plotly(fig_evol)
    fig_evol.update_layout(xaxis_tickangle=-30)

    # --- Ingresos / gastos / ahorro ---
    fig_ahorro = go.Figure()
    fig_ahorro.add_bar(x=res["mes"], y=res["ingreso_total"],
                       name="Ingresos", marker_color=PREFIN_VERDE)
    fig_ahorro.add_bar(x=res["mes"], y=res["gasto_total"],
                       name="Gastos", marker_color=PREFIN_ROJO)
    fig_ahorro.add_scatter(x=res["mes"], y=res["ahorro_neto"],
                           name="Ahorro neto", mode="lines+markers",
                           line=dict(color=PREFIN_INK, width=2),
                           marker=dict(size=7))
    fig_ahorro.update_layout(title="Ingresos, gastos y ahorro mensual",
                              barmode="group", xaxis_tickangle=-30)
    aplicar_layout_plotly(fig_ahorro)

    # --- Día de la semana ---
    fig_dia = px.bar(
        diario, x="dia_es", y="importe_abs",
        title="Gasto medio por día de la semana",
        labels={"importe_abs": "Gasto medio (€)", "dia_es": ""},
        color_discrete_sequence=[PREFIN_INK],
    )
    aplicar_layout_plotly(fig_dia)

    # --- Anomalías ---
    anom_df = anomalias_df[anomalias_df["anomalia"]].copy()
    anom_df["fecha_str"] = anom_df["fecha"].dt.strftime("%d/%m/%Y")
    anom_df["importe_str"] = anom_df["importe"].apply(lambda x: f"{x:,.2f} €")
    anom_df["z_str"] = anom_df["z_score"].apply(lambda x: f"{x:.1f}σ")

    return dbc.Container([
        html.Div([
            html.H1("Análisis", className="page-title"),
            html.P("Patrones de gasto, tendencias y transacciones inusuales.",
                   className="page-subtitle"),
        ], className="pt-4"),

        dbc.Row([
            dbc.Col(dbc.Card(dbc.CardBody(dcc.Graph(figure=fig_evol,
                                                    config={"displayModeBar": False}))),
                    md=12, className="mb-3"),
        ]),

        dbc.Row([
            dbc.Col(dbc.Card(dbc.CardBody(dcc.Graph(figure=fig_ahorro,
                                                    config={"displayModeBar": False}))),
                    md=8, className="mb-3"),
            dbc.Col(dbc.Card(dbc.CardBody(dcc.Graph(figure=fig_dia,
                                                    config={"displayModeBar": False}))),
                    md=4, className="mb-3"),
        ]),

        # --- Prevención: cambios de régimen (Fase 4) ---
        dbc.Row([
            dbc.Col(dbc.Card([
                dbc.CardHeader([html.I(className="bi bi-activity me-2"),
                                 "Cambios de régimen en tu gasto"]),
                dbc.CardBody([
                    html.P("Momentos en los que tu nivel de gasto cambió de forma "
                           "sostenida (no un pico puntual, sino un cambio que se "
                           "mantiene).",
                           style={"color": PREFIN_TEXTO_SEC, "fontSize": "0.85rem"}),
                    dcc.Graph(figure=fig_regimen, config={"displayModeBar": False}),
                    html.Div([
                        html.Span([
                            html.I(className=f"bi bi-arrow-{'up' if c['direccion']=='subida' else 'down'}-right me-1",
                                   style={"color": PREFIN_ROJO if c['direccion']=='subida' else PREFIN_VERDE}),
                            f"{c['mes']}: {c['direccion']} de "
                            f"{c['media_antes']:,.0f} € a {c['media_despues']:,.0f} €",
                        ], style={"display": "block", "fontSize": "0.85rem",
                                  "marginBottom": "3px"})
                        for _, c in cambios_reg.iterrows()
                    ] or [html.Span("No se han detectado cambios de régimen.",
                                    style={"color": PREFIN_TEXTO_SEC,
                                           "fontSize": "0.85rem"})]),
                ]),
            ]), md=8, className="mb-3"),
            dbc.Col(dbc.Card([
                dbc.CardHeader([html.I(className="bi bi-search me-2"),
                                 "Detectores de anomalías"]),
                dbc.CardBody([
                    html.P("Comparamos dos métodos: el z-score por categoría "
                           "(univariante, baseline) y un IsolationForest que "
                           "aprende tu patrón global (multivariante).",
                           style={"color": PREFIN_TEXTO_SEC, "fontSize": "0.82rem"}),
                    _metrica_fila("z-score (baseline)", comp_det["n_zscore"]),
                    _metrica_fila("IsolationForest", comp_det["n_isolation"]),
                    _metrica_fila("Coinciden", comp_det["n_comunes"]),
                    _metrica_fila("Solo IsolationForest", comp_det["solo_isolation"]),
                ]),
            ]), md=4, className="mb-3"),
        ]),

        dbc.Card([
            dbc.CardHeader([
                html.I(className="bi bi-exclamation-triangle me-2"),
                f"Transacciones inusuales ({len(anom_df)})",
            ]),
            dbc.CardBody([
                html.P(
                    "Transacciones con un importe superior a 2.5σ por encima de la "
                    "media de su categoría.",
                    style={"color": PREFIN_TEXTO_SEC, "fontSize": "0.85rem"},
                ) if not anom_df.empty else None,
                _tabla_anomalias(anom_df) if not anom_df.empty else dbc.Alert(
                    [html.I(className="bi bi-check-circle me-2"),
                     "No se han detectado anomalías destacables."],
                    color="success", className="mb-0"),
            ]),
        ]),
    ], fluid=True, className="px-4")


# ============================================================
# PÁGINA: RIESGO ML
# ============================================================
def layout_riesgo(df):
    if df is None or df.empty:
        return _pantalla_sin_datos()

    _modelo = _asegurar_modelo()
    pred = _modelo.predecir(df)
    nivel = pred["nivel"]
    score = pred["score"]
    met = pred.get("metricas", {})

    # Explicabilidad (Fase 5): por qué el modelo predice este nivel.
    try:
        explic = explicar_natural(_modelo, df) if pred.get("entrenado") else None
        fig_contrib = figura_contribuciones(_modelo, df) if pred.get("entrenado") else None
    except Exception:
        explic, fig_contrib = None, None

    # --- Gauge ---
    fig_gauge = go.Figure(go.Indicator(
        mode="gauge+number",
        value=score,
        domain={"x": [0, 1], "y": [0, 1]},
        number={"font": {"size": 40, "color": PREFIN_INK}},
        gauge={
            "axis": {"range": [0, 100], "tickwidth": 1,
                     "tickcolor": PREFIN_TEXTO_SEC,
                     "tickfont": {"size": 10, "color": PREFIN_TEXTO_SEC}},
            "bar": {"color": COLOR_RIESGO.get(nivel, PREFIN_TEXTO_SEC),
                    "thickness": 0.25},
            "bgcolor": "white",
            "borderwidth": 0,
            "steps": [
                {"range": [0, 40],  "color": "#F0FDF4"},
                {"range": [40, 70], "color": "#FFFBEB"},
                {"range": [70, 100],"color": "#FEF2F2"},
            ],
            "threshold": {"line": {"color": PREFIN_ROJO, "width": 2},
                          "thickness": 0.75, "value": 70},
        },
    ))
    fig_gauge.update_layout(
        height=260, paper_bgcolor=PREFIN_SUPERFICIE,
        margin=dict(t=20, b=20, l=20, r=20),
        font=dict(family="Inter, sans-serif", color=PREFIN_INK),
    )

    # --- Importancias de features ---
    imp = pred.get("importancias", {})
    nombres_es = {
        "ingreso_mensual":       "Ingreso mensual",
        "gasto_total":           "Gasto total",
        "ratio_gasto_ingreso":   "Ratio gasto/ingreso",
        "ratio_ahorro":          "Ratio de ahorro",
        "gasto_ocio_ratio":      "Ratio ocio/ingreso",
        "gasto_suscripciones":   "Gasto en suscripciones",
        "gasto_supermercado":    "Gasto en supermercado",
        "variabilidad_gasto":    "Variabilidad del gasto",
        "tendencia_gasto":       "Tendencia de gasto",
        "n_categorias_activas":  "Nº categorías activas",
        "colchon_meses":         "Colchón de liquidez",
    }
    if imp:
        imp_df = pd.DataFrame([
            {"feature": nombres_es.get(k, k), "importancia": v}
            for k, v in sorted(imp.items(), key=lambda x: -x[1])
        ])
        fig_imp = px.bar(
            imp_df, x="importancia", y="feature", orientation="h",
            title="Importancia de variables · Random Forest",
            labels={"importancia": "Importancia relativa", "feature": ""},
            color="importancia", color_continuous_scale=ESCALA_INTENSIDAD,
        )
        aplicar_layout_plotly(fig_imp)
        fig_imp.update_layout(yaxis={"categoryorder": "total ascending"},
                              coloraxis_showscale=False)
    else:
        fig_imp = go.Figure()
        fig_imp.add_annotation(text="Datos insuficientes",
                                showarrow=False, font=dict(size=13))
        aplicar_layout_plotly(fig_imp)

    # --- Probabilidades ---
    proba = pred.get("probabilidades", {})
    proba_items = []
    for niv, pct in proba.items():
        color_map = {"Bajo": PREFIN_VERDE, "Medio": PREFIN_AMBAR, "Alto": PREFIN_ROJO,
                     "Estable": PREFIN_VERDE, "En riesgo": PREFIN_ROJO}
        color_barra = color_map.get(niv, PREFIN_TEXTO_SEC)
        proba_items.append(html.Div([
            html.Div([
                html.Span(niv, style={"fontWeight": "500", "fontSize": "0.88rem"}),
                html.Span(f"{pct:.1f}%",
                          style={"color": PREFIN_TEXTO_SEC, "fontSize": "0.85rem",
                                 "marginLeft": "auto"}),
            ], style={"display": "flex", "marginBottom": "4px"}),
            html.Div(style={
                "height": "6px", "backgroundColor": PREFIN_BORDE,
                "borderRadius": "3px", "overflow": "hidden",
            }, children=[
                html.Div(style={
                    "height": "100%", "width": f"{pct}%",
                    "backgroundColor": color_barra,
                    "borderRadius": "3px",
                }),
            ]),
        ], style={"marginBottom": "12px"}))

    # --- Factores de riesgo ---
    factores_items = []
    for f in pred.get("factores", []):
        factores_items.append(html.Div([
            html.Div([
                badge_riesgo(f["nivel"]),
                html.Strong(f["factor"], style={"marginLeft": "8px",
                                                  "fontSize": "0.88rem"}),
            ], style={"display": "flex", "alignItems": "center"}),
            html.Div(f["valor"], style={"color": PREFIN_TEXTO_SEC,
                                          "fontSize": "0.82rem",
                                          "marginTop": "4px",
                                          "marginLeft": "2px"}),
        ], style={
            "padding": "10px 12px",
            "borderBottom": f"1px solid {PREFIN_BORDE}",
        }))

    # --- Probabilidad de iliquidez (modelo vs gemelo MC) ---
    prob_iliq = pred.get("prob_iliquidez")
    prob_mc = pred.get("prob_iliquidez_mc")
    ventana = pred.get("ventana", 3)
    prob_txt = f"{prob_iliq:.0%}" if prob_iliq is not None else "—"
    prob_mc_txt = f"{prob_mc:.0%}" if prob_mc is not None else "—"

    # --- Previsión de gasto con incertidumbre (Fase 1) ---
    try:
        previsor = PrevisorGasto().fit(df)
        banda = previsor.predecir(meses_adelante=1).iloc[0]
        fig_prev = figura_prevision(df, meses_adelante=6)
        prev_disponible = True
        prev_modelo_txt = ("Regresión cuantílica" if previsor.usa_modelo_
                           else "Cuantiles empíricos (histórico corto)")
    except Exception:
        banda = None
        fig_prev = None
        prev_disponible = False
        prev_modelo_txt = ""

    return dbc.Container([
        html.Div([
            html.H1("Predicción de riesgo", className="page-title"),
            html.P(f"Probabilidad de quedarte sin dinero en los próximos {ventana} meses. "
                   "Modelo entrenado sobre una población de usuarios y validado en "
                   "usuarios nunca vistos.",
                   className="page-subtitle"),
        ], className="pt-4"),

        dbc.Row([
            # Gauge
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader("Puntuación de riesgo"),
                    dbc.CardBody([
                        dcc.Graph(figure=fig_gauge,
                                  config={"displayModeBar": False}),
                        html.Div([
                            html.Span("Nivel detectado: ",
                                       style={"color": PREFIN_TEXTO_SEC,
                                              "fontSize": "0.88rem"}),
                            badge_riesgo(nivel),
                        ], className="text-center mt-2"),
                    ]),
                ]),
            ], md=5, className="mb-3"),

            # Probabilidades + gasto futuro
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader("Distribución de probabilidad"),
                    dbc.CardBody(proba_items if proba_items
                                 else "Sin datos de probabilidad"),
                ], className="mb-3"),
                dbc.Card([
                    dbc.CardHeader("Riesgo de iliquidez"),
                    dbc.CardBody([
                        html.Div(prob_txt, style={
                            "fontSize": "1.9rem", "fontWeight": "700",
                            "color": COLOR_RIESGO.get(nivel, PREFIN_INK),
                            "fontVariantNumeric": "tabular-nums",
                        }),
                        html.Small("según el modelo predictivo",
                                   style={"color": PREFIN_TEXTO_SEC}),
                        html.Hr(style={"margin": "0.6rem 0", "borderColor": PREFIN_BORDE}),
                        html.Div([
                            html.Span("Gemelo Monte Carlo: ",
                                      style={"color": PREFIN_TEXTO_SEC, "fontSize": "0.82rem"}),
                            html.Strong(prob_mc_txt, style={"fontVariantNumeric": "tabular-nums"}),
                        ]),
                        html.Small("contraste independiente",
                                   style={"color": PREFIN_TEXTO_SEC}),
                    ]),
                ]),
            ], md=3, className="mb-3"),

            # Factores
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader("Factores de riesgo"),
                    dbc.CardBody(factores_items if factores_items
                                 else "Sin factores destacables",
                                 style={"padding": "0"}),
                ], className="h-100"),
            ], md=4, className="mb-3"),
        ]),

        dbc.Row([
            dbc.Col(dbc.Card(dbc.CardBody(dcc.Graph(figure=fig_imp,
                                                    config={"displayModeBar": False}))),
                    md=8),
            dbc.Col(dbc.Card([
                dbc.CardHeader("Calidad del modelo (validación honesta)"),
                dbc.CardBody([
                    html.P("Evaluado en usuarios nunca vistos durante el "
                           "entrenamiento (sin fuga de datos).",
                           style={"color": PREFIN_TEXTO_SEC, "fontSize": "0.8rem"}),
                    html.Div([
                        _metrica_fila("ROC-AUC", met.get("roc_auc")),
                        _metrica_fila("Exactitud", met.get("accuracy")),
                        _metrica_fila("Sensibilidad (recall)", met.get("recall")),
                        _metrica_fila("Precisión", met.get("precision")),
                    ]) if met else html.Span("Modelo no entrenado.",
                                              style={"color": PREFIN_TEXTO_SEC}),
                    html.Hr(style={"margin": "0.6rem 0", "borderColor": PREFIN_BORDE}),
                    html.Small(
                        f"{met.get('n_usuarios', 0)} usuarios · "
                        f"{met.get('n_total', 0)} muestras · "
                        f"{met.get('tasa_positivos', 0):.0%} en riesgo",
                        style={"color": PREFIN_TEXTO_SEC}),
                ]),
            ]), md=4),
        ]),

        # --- Explicabilidad: ¿por qué este nivel? (Fase 5) ---
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader([html.I(className="bi bi-chat-square-text me-2"),
                                     "¿Por qué este nivel de riesgo?"]),
                    dbc.CardBody([
                        html.Div([
                            html.Div([
                                html.Strong("Lo que eleva tu riesgo",
                                            style={"color": PREFIN_ROJO,
                                                   "fontSize": "0.9rem"}),
                                html.Ul([html.Li(f, style={"fontSize": "0.88rem",
                                                            "marginBottom": "4px"})
                                         for f in explic["frases_sube"]]
                                        or [html.Li("Nada destacable.",
                                                    style={"color": PREFIN_TEXTO_SEC})]),
                            ], style={"marginBottom": "0.8rem"}),
                            html.Div([
                                html.Strong("Lo que te protege",
                                            style={"color": PREFIN_VERDE,
                                                   "fontSize": "0.9rem"}),
                                html.Ul([html.Li(f, style={"fontSize": "0.88rem",
                                                            "marginBottom": "4px"})
                                         for f in explic["frases_baja"]]
                                        or [html.Li("Nada destacable.",
                                                    style={"color": PREFIN_TEXTO_SEC})]),
                            ]),
                            html.Small(f"Método: {explic['metodo']}",
                                       style={"color": PREFIN_TEXTO_SEC}),
                        ]) if explic else html.Span(
                            "Explicación no disponible.",
                            style={"color": PREFIN_TEXTO_SEC}),
                    ]),
                ], className="mt-3 h-100"),
            ], md=5),
            dbc.Col([
                dbc.Card(dbc.CardBody(
                    dcc.Graph(figure=fig_contrib, config={"displayModeBar": False})
                    if fig_contrib is not None else html.Span(
                        "Sin contribuciones que mostrar.",
                        style={"color": PREFIN_TEXTO_SEC})),
                    className="mt-3 h-100"),
            ], md=7),
        ]),

        # --- Previsión de gasto con incertidumbre (Fase 1) ---
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader([
                        html.I(className="bi bi-graph-up-arrow me-2"),
                        "Previsión de gasto con incertidumbre",
                    ]),
                    dbc.CardBody([
                        html.P(
                            "Banda de previsión p10–p90 del gasto mensual. En lugar de un "
                            "único número, se muestra el rango probable de gasto: cuanto más "
                            "estrecha es la banda, más predecible es tu gasto.",
                            style={"color": PREFIN_TEXTO_SEC, "fontSize": "0.85rem"},
                        ),
                        dbc.Row([
                            dbc.Col(kpi_card("Escenario optimista (p10)",
                                             f"{banda['p10']:,.2f} €",
                                             "bi-arrow-down", PREFIN_VERDE), md=4),
                            dbc.Col(kpi_card("Previsión central (p50)",
                                             f"{banda['p50']:,.2f} €",
                                             "bi-dot", PREFIN_INK), md=4),
                            dbc.Col(kpi_card("Escenario tensionado (p90)",
                                             f"{banda['p90']:,.2f} €",
                                             "bi-arrow-up", PREFIN_ROJO), md=4),
                        ], className="mb-3") if prev_disponible else None,
                        dcc.Graph(figure=fig_prev, config={"displayModeBar": False})
                        if prev_disponible else dbc.Alert(
                            "Sin datos suficientes para la previsión.", color="warning"),
                        html.Small(f"Método: {prev_modelo_txt}",
                                   style={"color": PREFIN_TEXTO_SEC})
                        if prev_disponible else None,
                    ]),
                ], className="mt-3"),
            ], md=12),
        ]),
    ], fluid=True, className="px-4")


# ============================================================
# PÁGINA: GEMELO DIGITAL
# ============================================================
def layout_simulacion(df):
    if df is None or df.empty:
        return _pantalla_sin_datos()

    estado = estado_actual(df)
    cats_disponibles = [c for c in estado.get("gasto_por_categoria", {}).keys()
                        if c != "Ingresos"]

    return dbc.Container([
        html.Div([
            html.H1("Gemelo digital", className="page-title"),
            html.P("Simula escenarios alternativos y anticipa el impacto de tus decisiones.",
                   className="page-subtitle"),
        ], className="pt-4"),

        dbc.Row([
            # Panel de control
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader([html.I(className="bi bi-sliders me-2"),
                                     "Configurar escenario"]),
                    dbc.CardBody([
                        dbc.Label("Horizonte (meses)"),
                        dcc.Slider(1, 36, 1, value=12, id="sl-horizonte",
                                   marks={6: "6", 12: "12", 24: "24", 36: "36"},
                                   tooltip={"always_visible": False}),

                        html.Hr(style={"borderColor": PREFIN_BORDE,
                                        "margin": "1rem 0"}),

                        dbc.Label("Cambio en ingresos (%)"),
                        dcc.Slider(-30, 50, 5, value=0, id="sl-ingreso",
                                   marks={-30: "-30%", 0: "0%", 25: "+25%",
                                           50: "+50%"},
                                   tooltip={"always_visible": False}),

                        html.Hr(style={"borderColor": PREFIN_BORDE,
                                        "margin": "1rem 0"}),

                        dbc.Label("Ajuste por categoría"),
                        html.P("Elige una categoría y un % de cambio.",
                               style={"color": PREFIN_TEXTO_SEC,
                                      "fontSize": "0.78rem"}),

                        dbc.Select(id="sel-cat-sim",
                                   options=[{"label": c, "value": c}
                                             for c in cats_disponibles],
                                   value=cats_disponibles[0] if cats_disponibles else None),
                        dcc.Slider(-60, 60, 5, value=0, id="sl-cat-cambio",
                                   marks={-60: "-60%", -30: "-30%", 0: "0",
                                           30: "+30%", 60: "+60%"},
                                   tooltip={"always_visible": False},
                                   className="mt-2"),
                        dbc.Button(
                            [html.I(className="bi bi-plus-lg me-1"),
                             "Añadir cambio"],
                            id="btn-add-cat", color="secondary",
                            size="sm", className="mt-2 w-100", n_clicks=0,
                        ),
                        html.Div(id="lista-cambios-cat", className="mt-2"),
                        dcc.Store(id="store-cambios-cat", data={}),

                        html.Hr(style={"borderColor": PREFIN_BORDE,
                                        "margin": "1rem 0"}),

                        dbc.Label("Gasto imprevisto en mes 1 (€)"),
                        dbc.Input(id="inp-imprevisto", type="number",
                                  value=0, min=0, step=50),

                        dbc.Label("Meta de ahorro mensual (€)",
                                  className="mt-3"),
                        dbc.Input(id="inp-meta-ahorro", type="number",
                                  value=0, min=0, step=50),
                        html.Small("0 = sin meta",
                                    style={"color": PREFIN_TEXTO_SEC,
                                           "fontSize": "0.75rem"}),

                        dbc.Button(
                            [html.I(className="bi bi-play-fill me-2"),
                             "Ejecutar simulación"],
                            id="btn-simular", color="primary",
                            className="mt-4 w-100", n_clicks=0,
                        ),
                    ]),
                ]),
            ], md=3, className="mb-3"),

            # Resultados
            dbc.Col([dcc.Loading(html.Div(id="div-simulacion-resultado"),
                                 type="default", color=PREFIN_INK)], md=9),
        ]),
    ], fluid=True, className="px-4")


# ============================================================
# PÁGINA: MI PLAN (motor prescriptivo)
# ============================================================
def layout_plan(df):
    if df is None or df.empty:
        return _pantalla_sin_datos()

    return dbc.Container([
        html.Div([
            html.H1("Mi plan", className="page-title"),
            html.P("PREFIN busca por ti el mejor plan de ahorro: cuánto puedes "
                   "apartar cada mes sin arriesgarte a quedarte sin dinero.",
                   className="page-subtitle"),
        ], className="pt-4"),

        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader([html.I(className="bi bi-sliders me-2"),
                                     "Tus preferencias"]),
                    dbc.CardBody([
                        dbc.Label("Horizonte (meses)"),
                        dcc.Slider(6, 24, 6, value=12, id="pl-horizonte",
                                   marks={6: "6", 12: "12", 18: "18", 24: "24"}),

                        dbc.Label("Riesgo máximo que aceptas", className="mt-3"),
                        html.P("Probabilidad de quedarte sin dinero que estás "
                               "dispuesto a tolerar.",
                               style={"color": PREFIN_TEXTO_SEC, "fontSize": "0.78rem"}),
                        dcc.Slider(0.05, 0.30, 0.05, value=0.10, id="pl-umbral",
                                   marks={0.05: "5%", 0.10: "10%", 0.20: "20%",
                                          0.30: "30%"}),

                        dbc.Label("¿Tienes un gasto puntual previsto? (€)",
                                  className="mt-3"),
                        dbc.Input(id="pl-gasto", type="number", value=0,
                                  min=0, step=100),
                        html.Small("0 = ninguno. Si lo indicas, el plan elige el "
                                   "mejor mes para afrontarlo.",
                                   style={"color": PREFIN_TEXTO_SEC,
                                          "fontSize": "0.75rem"}),

                        dbc.Button(
                            [html.I(className="bi bi-stars me-2"), "Ver mi plan"],
                            id="btn-plan", color="primary",
                            className="mt-4 w-100", n_clicks=0,
                        ),
                    ]),
                ]),
            ], md=3, className="mb-3"),

            dbc.Col([
                dcc.Loading(
                    html.Div(id="div-plan-resultado", children=dbc.Alert(
                        [html.I(className="bi bi-info-circle me-2"),
                         "Pulsa «Ver mi plan» para que PREFIN calcule tus opciones."],
                        color="light")),
                    type="default",
                ),
            ], md=9),
        ]),
    ], fluid=True, className="px-4")


# ============================================================
# PÁGINA: MICRO-AHORRO
# ============================================================
def layout_ahorro(df):
    if df is None or df.empty:
        return _pantalla_sin_datos()

    return dbc.Container([
        html.Div([
            html.H1("Micro-ahorro automático", className="page-title"),
            html.P("Redondea cada gasto al siguiente euro y acumula ahorro sin esfuerzo.",
                   className="page-subtitle"),
        ], className="pt-4"),

        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader([html.I(className="bi bi-gear me-2"),
                                     "Unidad de redondeo"]),
                    dbc.CardBody([
                        dbc.RadioItems(
                            id="radio-redondeo",
                            options=[{"label": k, "value": v}
                                      for k, v in OPCIONES_REDONDEO.items()],
                            value=1,
                        ),
                    ]),
                ]),
            ], md=3, className="mb-3"),
            dbc.Col(dcc.Loading(html.Div(id="div-ahorro-resultado"),
                                type="default", color=PREFIN_INK), md=9),
        ]),
    ], fluid=True, className="px-4")


# ============================================================
# AUXILIAR: leer DataFrame desde el cache del servidor
# ============================================================
def _df_from_store(_token):
    return _CACHE.get("df", None)


# ============================================================
# CALLBACKS — ROUTING
# ============================================================
@app.callback(Output("page-content", "children"),
              Input("url", "pathname"),
              State("store-df", "data"))
def render_page(pathname, store_data):
    df = _df_from_store(store_data)
    routes = {
        "/":           lambda: layout_dashboard(df),
        "/analisis":   lambda: layout_analisis(df),
        "/riesgo":     lambda: layout_riesgo(df),
        "/simulacion": lambda: layout_simulacion(df),
        "/plan":       lambda: layout_plan(df),
        "/ahorro":     lambda: layout_ahorro(df),
        "/datos":      lambda: layout_datos(),
    }
    fn = routes.get(pathname, lambda: layout_dashboard(df))
    return fn()


# ============================================================
# CALLBACKS — CARGA DE DATOS
# ============================================================
@app.callback(
    Output("store-df", "data"),
    Output("toast-notif", "children"),
    Output("toast-notif", "is_open"),
    Output("toast-notif", "header"),
    Output("datos-feedback", "children"),
    Input("btn-sintetico", "n_clicks"),
    Input("btn-demo", "n_clicks"),
    Input("btn-truelayer", "n_clicks"),
    Input("upload-data", "contents"),
    State("upload-data", "filename"),
    State("sl-meses", "value"),
    State("inp-nomina", "value"),
    State("radio-perfil", "value"),
    State("inp-mes-tl", "value"),
    prevent_initial_call=True,
)
def cargar_datos(n_sint, n_demo, n_tl, contents, filename, meses, nomina, perfil, mes_tl):
    ctx = callback_context
    trigger = ctx.triggered[0]["prop_id"].split(".")[0] if ctx.triggered else ""

    df = None
    mensaje = ""
    error = False

    try:
        if trigger == "btn-demo":
            df = cargar_sinteticos(meses=36, ingreso_base=2000, perfil="medio",
                                    seed=42, realista=True)
            mensaje = f"Modo demo cargado: {len(df):,} transacciones (36 meses, datos realistas)."

        elif trigger == "btn-sintetico":
            df = cargar_sinteticos(meses=meses or 12, ingreso_base=nomina or 1800,
                                    perfil=perfil or "medio", realista=True)
            mensaje = f"Datos sintéticos generados: {len(df):,} transacciones ({meses} meses, perfil {perfil})."

        elif trigger == "upload-data" and contents:
            df = cargar_desde_upload(contents, filename)
            mensaje = f"Archivo cargado: {filename} — {len(df):,} transacciones."

        elif trigger == "btn-truelayer":
            df = cargar_desde_truelayer(month=mes_tl or None)
            mensaje = f"Datos de TrueLayer cargados: {len(df):,} transacciones."

    except Exception as e:
        error = True
        mensaje = f"Error al cargar datos: {str(e)}"

    if df is not None and not error:
        # El modelo de riesgo es poblacional (entrenado una sola vez de forma
        # perezosa); no se reentrena con cada carga de un usuario.
        _CACHE["df"] = df
        feedback = dbc.Alert([html.I(className="bi bi-check-circle me-2"), mensaje],
                              color="success")
        return "loaded", mensaje, True, "PREFIN", feedback
    else:
        feedback = dbc.Alert([html.I(className="bi bi-exclamation-circle me-2"), mensaje],
                              color="danger") if error else dash.no_update
        return dash.no_update, mensaje, error, "PREFIN", feedback


# ============================================================
# CALLBACKS — SIMULACIÓN
# ============================================================
@app.callback(
    Output("store-cambios-cat", "data"),
    Output("lista-cambios-cat", "children"),
    Input("btn-add-cat", "n_clicks"),
    State("sel-cat-sim", "value"),
    State("sl-cat-cambio", "value"),
    State("store-cambios-cat", "data"),
    prevent_initial_call=True,
)
def añadir_cambio_cat(n, cat, pct, cambios_actuales):
    if not cat:
        return cambios_actuales, dash.no_update
    cambios_actuales = cambios_actuales or {}
    cambios_actuales[cat] = pct

    items = []
    for c, p in cambios_actuales.items():
        bg = "#F0FDF4" if p < 0 else "#FEF2F2"
        fg = "#14532D" if p < 0 else "#7F1D1D"
        items.append(html.Span(
            f"{c}: {p:+.0f}%",
            style={
                "backgroundColor": bg, "color": fg,
                "padding": "3px 9px", "borderRadius": "6px",
                "fontSize": "0.78rem", "fontWeight": "500",
                "marginRight": "5px", "marginBottom": "4px",
                "display": "inline-block",
            },
        ))
    return cambios_actuales, html.Div(items)


@app.callback(
    Output("div-simulacion-resultado", "children"),
    Input("btn-simular", "n_clicks"),
    State("store-df", "data"),
    State("sl-horizonte", "value"),
    State("sl-ingreso", "value"),
    State("store-cambios-cat", "data"),
    State("inp-imprevisto", "value"),
    State("inp-meta-ahorro", "value"),
    prevent_initial_call=True,
)
def ejecutar_simulacion(n, store_data, horizonte, cambio_ing, cambios_cat,
                         imprevisto, meta):
    df = _df_from_store(store_data)
    if df is None or df.empty:
        return dbc.Alert("Sin datos. Carga primero un dataset.", color="warning")

    # --- Gemelo digital estocástico (Monte Carlo) ---
    mc = simular_montecarlo(
        df,
        meses=horizonte or 12,
        n_sim=5000,
        cambio_ingreso_pct=cambio_ing or 0,
        cambios_categoria=cambios_cat or {},
        evento_imprevisto=imprevisto or 0,
        mes_evento=1,
        meta_ahorro_mensual=meta if meta and meta > 0 else None,
        umbral_iliquidez=0.0,
    )
    if not mc:
        return dbc.Alert("Sin datos suficientes para simular.", color="warning")

    fig_cono = figura_cono_montecarlo(mc)

    # P(iliquidez) por mes.
    bandas = mc["bandas"]
    fig_iliq = px.area(
        bandas, x="mes", y="prob_iliquidez",
        title="Probabilidad de iliquidez por mes",
        labels={"prob_iliquidez": "P(saldo < 0)", "mes": "Mes"},
        color_discrete_sequence=[PREFIN_ROJO],
    )
    fig_iliq.update_traces(line=dict(width=2), fillcolor="rgba(220,38,38,0.10)")
    fig_iliq.update_yaxes(tickformat=".0%", range=[0, 1])
    aplicar_layout_plotly(fig_iliq)

    prob_horizonte = mc["prob_iliquidez_horizonte"]
    color_prob = (PREFIN_VERDE if prob_horizonte < 0.05
                  else PREFIN_AMBAR if prob_horizonte < 0.20 else PREFIN_ROJO)
    color_var = PREFIN_VERDE if mc["var_95"] >= 0 else PREFIN_ROJO

    return html.Div([
        # KPIs estocásticos.
        dbc.Row([
            dbc.Col(kpi_card("Saldo final esperado",
                             f"{mc['saldo_final_esperado']:,.0f} €",
                             "bi-wallet-fill", PREFIN_INK,
                             "media de 5.000 escenarios",
                             ayuda="Saldo medio al final del horizonte, promediando "
                                   "las 5.000 trayectorias simuladas."),
                    md=3, className="mb-3"),
            dbc.Col(kpi_card("Riesgo de quedarte sin dinero",
                             f"{prob_horizonte:.0%}",
                             "bi-exclamation-triangle", color_prob,
                             "en algún mes del horizonte",
                             ayuda="Porcentaje de escenarios simulados en los que tu "
                                   "saldo baja de 0 € en algún momento."),
                    md=3, className="mb-3"),
            dbc.Col(kpi_card("Peor caso probable (VaR 95%)",
                             f"{mc['var_95']:,.0f} €",
                             "bi-graph-down-arrow", color_var,
                             "saldo mínimo, 95% confianza",
                             ayuda="Value-at-Risk: en el 95% de los escenarios tu "
                                   "saldo no baja de esta cifra. Si es negativa, "
                                   "indica riesgo de números rojos."),
                    md=3, className="mb-3"),
            dbc.Col(kpi_card("Caso extremo medio (CVaR)",
                             f"{mc['cvar_95']:,.0f} €",
                             "bi-arrow-down-circle",
                             PREFIN_VERDE if mc["cvar_95"] >= 0 else PREFIN_ROJO,
                             "media del 5% peor",
                             ayuda="Expected Shortfall: saldo medio en el 5% de "
                                   "escenarios PEORES. Mide cómo de grave sería "
                                   "el mal caso."),
                    md=3, className="mb-3"),
        ]),
        # Cono Monte Carlo (elemento protagonista).
        dbc.Row([
            dbc.Col(dbc.Card(dbc.CardBody(dcc.Graph(
                figure=fig_cono, config={"displayModeBar": False},
                style={"height": "440px"})), className="cono-card"),
                md=8, className="mb-3"),
            dbc.Col(dbc.Card(dbc.CardBody(dcc.Graph(
                figure=fig_iliq, config={"displayModeBar": False},
                style={"height": "440px"}))), md=4, className="mb-3"),
        ]),
    ])


# ============================================================
# CALLBACKS — MI PLAN (motor prescriptivo)
# ============================================================
# Paleta profesional para distinguir las tarjetas de plan (acento, fondo de
# cabecera, texto oscuro). El recomendado va en verde (es el "bueno"); las
# alternativas usan acentos NO semánticos (índigo, cian, violeta, slate) para no
# sugerir falsamente precaución/riesgo.
PALETA_PLANES = [
    ("#16A34A", "#F0FDF4", "#14532D"),   # recomendado · verde
    ("#6366F1", "#EEF2FF", "#3730A3"),   # índigo
    ("#0EA5E9", "#E0F2FE", "#075985"),   # cian
    ("#8B5CF6", "#F5F3FF", "#5B21B6"),   # violeta
    ("#0D9488", "#ECFDF5", "#115E59"),   # teal
]


def _cajetin(label, valor, color, icono):
    """Cajetín de estadística: caja con icono, etiqueta y valor destacado."""
    return html.Div([
        html.Div(html.I(className=f"bi {icono}"),
                 style={"color": color, "fontSize": "1.1rem", "marginBottom": "2px"}),
        html.Div(valor, style={"fontWeight": "700", "fontSize": "1.05rem",
                               "color": color, "fontVariantNumeric": "tabular-nums",
                               "lineHeight": "1.1"}),
        html.Div(label, style={"color": PREFIN_TEXTO_SEC, "fontSize": "0.72rem",
                               "marginTop": "2px"}),
    ], style={
        "flex": "1", "textAlign": "center", "padding": "0.6rem 0.4rem",
        "borderRadius": "10px", "backgroundColor": "#F8FAFC",
        "border": f"1px solid {PREFIN_BORDE}",
    })


def _tarjeta_plan(plan, idx, recomendado=False):
    """Tarjeta de un plan propuesto, con impacto cuantificado y color distintivo."""
    riesgo = plan["prob_iliquidez"]
    color_riesgo = (PREFIN_VERDE if riesgo < 0.05
                    else PREFIN_AMBAR if riesgo < 0.15 else PREFIN_ROJO)
    acento, fondo_hdr, texto = PALETA_PLANES[idx % len(PALETA_PLANES)]
    encabezado = ([html.I(className="bi bi-trophy-fill me-2"), "Plan recomendado"]
                  if recomendado else
                  [html.I(className="bi bi-lightbulb me-2"), f"Alternativa {idx}"])
    acciones = plan.get("acciones") or [plan.get("explicacion", "")]
    return dbc.Card([
        dbc.CardHeader(encabezado,
                       style={"fontWeight": "600", "backgroundColor": fondo_hdr,
                              "color": texto, "borderBottom": f"1px solid {acento}33"}),
        dbc.CardBody([
            # Ahorro acumulado (cifra protagonista).
            html.Div(f"{plan['ahorro_protegido']:,.0f} €", style={
                "fontSize": "2rem", "fontWeight": "700", "color": texto,
                "fontVariantNumeric": "tabular-nums", "lineHeight": "1"}),
            html.Div("ahorro acumulado en el horizonte",
                     style={"color": PREFIN_TEXTO_SEC, "fontSize": "0.82rem",
                            "marginBottom": "0.9rem"}),
            # Acciones como bullet points.
            html.Div("Tu plan", style={
                "fontSize": "0.72rem", "fontWeight": "600", "letterSpacing": "0.05em",
                "textTransform": "uppercase", "color": PREFIN_TEXTO_SEC,
                "marginBottom": "0.35rem"}),
            html.Ul([
                html.Li(a, style={"fontSize": "0.88rem", "marginBottom": "0.25rem"})
                for a in acciones
            ], style={"paddingLeft": "1.15rem", "marginBottom": "0.9rem"}),
            # Cajetines de riesgo y VaR.
            html.Div([
                _cajetin("Riesgo de iliquidez", f"{riesgo:.0%}", color_riesgo,
                         "bi-shield-exclamation"),
                _cajetin("Peor caso (VaR)", f"{plan['var_95']:,.0f} €",
                         PREFIN_INK, "bi-graph-down-arrow"),
            ], style={"display": "flex", "gap": "0.6rem"}),
        ]),
    ], className="mb-3 h-100", style={"borderLeft": f"4px solid {acento}"})


@app.callback(
    Output("div-plan-resultado", "children"),
    Input("btn-plan", "n_clicks"),
    State("store-df", "data"),
    State("pl-horizonte", "value"),
    State("pl-umbral", "value"),
    State("pl-gasto", "value"),
    prevent_initial_call=True,
)
def calcular_plan(n, store_data, horizonte, umbral, gasto):
    df = _df_from_store(store_data)
    if df is None or df.empty:
        return dbc.Alert("Sin datos. Carga primero un dataset.", color="warning")

    res = optimizar_plan(
        df, meses=horizonte or 12,
        umbral_iliquidez_max=umbral or 0.10,
        gasto_puntual=gasto or 0.0,
    )
    planes = res["planes"]
    if not planes:
        return dbc.Alert(
            [html.I(className="bi bi-emoji-frown me-2"),
             "No he encontrado ningún plan de ahorro que mantenga tu riesgo por "
             "debajo del umbral. Prueba a subir el riesgo aceptable o a reducir "
             "el gasto puntual previsto."],
            color="warning")

    cols = []
    for i, plan in enumerate(planes):
        cols.append(dbc.Col(_tarjeta_plan(plan, i, recomendado=(i == 0)),
                            md=6, lg=4))

    return html.Div([
        dbc.Alert(
            [html.I(className="bi bi-lightbulb me-2"),
             f"He evaluado {res['n_evaluados']} combinaciones de acciones y estas "
             f"son las mejores que mantienen tu riesgo por debajo del "
             f"{res['umbral']:.0%}."],
            color="light", className="mb-3"),
        dbc.Row(cols),
    ])


# ============================================================
# CALLBACKS — MICRO-AHORRO
# ============================================================
@app.callback(
    Output("div-ahorro-resultado", "children"),
    Input("radio-redondeo", "value"),
    State("store-df", "data"),
)
def actualizar_microahorro(unidad, store_data):
    df = _df_from_store(store_data)
    if df is None or df.empty:
        return dbc.Alert("Sin datos.", color="warning")

    res = resumen_microahorro(df, unidad_redondeo=float(unidad))
    objetivos = objetivos_ahorro(res["proyeccion_anual"])
    por_cat = microahorro_por_categoria(df, unidad_redondeo=float(unidad))

    # Acumulado
    fig_acum = px.area(
        res["por_mes"], x="mes", y="acumulado",
        title=f"Ahorro acumulado · redondeo a {int(unidad)} €",
        labels={"acumulado": "Ahorro (€)", "mes": ""},
        color_discrete_sequence=[PREFIN_VERDE],
    )
    fig_acum.update_traces(line=dict(width=2), fillcolor="rgba(22, 163, 74, 0.12)")
    aplicar_layout_plotly(fig_acum)

    # Por categoría
    fig_cat = px.bar(
        por_cat, x="total", y="categoria", orientation="h",
        title="Potencial de ahorro por categoría",
        labels={"total": "Ahorro (€)", "categoria": ""},
        color="total", color_continuous_scale=ESCALA_INTENSIDAD,
    )
    aplicar_layout_plotly(fig_cat)
    fig_cat.update_layout(yaxis={"categoryorder": "total ascending"},
                          coloraxis_showscale=False)

    # Metas (tarjetas con icono, estilo "marketiniano")
    def _icono_meta(nombre):
        n = nombre.lower()
        if "emergencia" in n:
            return "bi-shield-check"
        if "vacacion" in n:
            return "bi-airplane"
        if "ordenador" in n or "portátil" in n:
            return "bi-laptop"
        if "coche" in n:
            return "bi-car-front"
        return "bi-bullseye"

    objetivo_cols = []
    for obj in objetivos:
        meses = obj["meses"]
        alcanzable = obj["alcanzable"] and isinstance(meses, (int, float))
        if alcanzable:
            eta = [f"{meses:.0f}", html.Small("meses")]
        else:
            eta = [html.I(className="bi bi-infinity"), html.Small("a este ritmo")]
        objetivo_cols.append(dbc.Col(html.Div([
            html.Div(html.I(className=f"bi {_icono_meta(obj['nombre'])}"),
                     className="meta-icono"),
            html.Div([
                html.Div(obj["nombre"], className="meta-nombre"),
                html.Div(f"{obj['importe']:,} €", className="meta-importe"),
            ]),
            html.Div(eta, className="meta-eta",
                     style={"color": PREFIN_VERDE if alcanzable else PREFIN_TEXTO_MUTED}),
        ], className="meta-card"), md=6, className="mb-3"))

    return html.Div([
        dbc.Row([
            dbc.Col(kpi_card("Ahorro mensual estimado",
                             f"{res['mensual_medio']:.2f} €",
                             "bi-piggy-bank", PREFIN_VERDE), md=3, className="mb-3"),
            dbc.Col(kpi_card("Proyección anual",
                             f"{res['proyeccion_anual']:.2f} €",
                             "bi-calendar-check", PREFIN_INK), md=3, className="mb-3"),
            dbc.Col(kpi_card("Transacciones",
                             f"{res['n_transacciones']:,}",
                             "bi-receipt", PREFIN_AMBAR), md=3, className="mb-3"),
            dbc.Col(kpi_card("Ahorro total histórico",
                             f"{res['total']:.2f} €",
                             "bi-cash-stack", PREFIN_VERDE), md=3, className="mb-3"),
        ]),
        dbc.Row([
            dbc.Col(dbc.Card(dbc.CardBody(dcc.Graph(figure=fig_acum,
                                                    config={"displayModeBar": False}))),
                    md=7, className="mb-3"),
            dbc.Col(dbc.Card(dbc.CardBody(dcc.Graph(figure=fig_cat,
                                                    config={"displayModeBar": False}))),
                    md=5, className="mb-3"),
        ]),
        dbc.Card([
            dbc.CardHeader([html.I(className="bi bi-bullseye me-2"),
                             "Metas alcanzables con micro-ahorro"]),
            dbc.CardBody(dbc.Row(objetivo_cols)),
        ]),
    ])


# ============================================================
# PUNTO DE ENTRADA
# ============================================================
if __name__ == "__main__":
   app.run(debug=False, port=8050)
