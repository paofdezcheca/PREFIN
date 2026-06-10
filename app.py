# app.py — PREFIN: Plataforma Inteligente de Predicción y Prevención Financiera
# Interfaz Dash con estética minimalista

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import dash
from dash import dcc, html, Input, Output, State, callback_context, dash_table
import dash_bootstrap_components as dbc

from config import (
    COLORES_CATEGORIA, COLOR_RIESGO,
    PREFIN_INK, PREFIN_FONDO, PREFIN_SUPERFICIE, PREFIN_BORDE,
    PREFIN_TEXTO_SEC, PREFIN_VERDE, PREFIN_ROJO, PREFIN_AMBAR,
    PLOTLY_LAYOUT,
)
from fuentes.loader import cargar_desde_upload, cargar_sinteticos, cargar_desde_truelayer
from modulos.analyzer import (
    resumen_mensual, gasto_por_categoria_mes,
    kpis_globales, detectar_anomalias, tendencia_gasto, gasto_diario_semana,
)
from modulos.ml_model import ModeloRiesgo
from modulos.digital_twin import estado_actual, simular_escenario, resumen_simulacion
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
modelo = ModeloRiesgo()

# Cache en memoria del servidor (no en el navegador)
_CACHE = {"df": None}


# ============================================================
# COMPONENTES REUTILIZABLES
# ============================================================

def aplicar_layout_plotly(fig):
    """Aplica el layout minimalista por defecto a una figura Plotly."""
    fig.update_layout(**PLOTLY_LAYOUT)
    return fig


def kpi_card(titulo, valor, icono, color=PREFIN_INK, subtitulo=""):
    """Tarjeta KPI minimalista."""
    return html.Div([
        html.Div([
            html.Div(
                html.I(className=f"bi {icono}", style={"color": color}),
                className="kpi-icon",
                style={"backgroundColor": f"{color}14"},  # 8% opacidad
            ),
            html.Div([
                html.Div(titulo, className="kpi-label"),
                html.Div(valor, className="kpi-value"),
                html.Div(subtitulo, className="kpi-sub") if subtitulo else html.Span(),
            ], style={"flex": "1", "marginLeft": "0.85rem"}),
        ], style={"display": "flex", "alignItems": "flex-start"}),
    ], className="kpi-card")


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
                html.Div("P", className="prefin-logo"),
                html.Span("PREFIN", className="navbar-brand mb-0"),
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
    html.Div(id="page-content", style={
        "backgroundColor": PREFIN_FONDO,
        "minHeight": "calc(100vh - 60px)",
        "paddingBottom": "60px",
    }),
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
            html.I(className="bi bi-database-slash",
                   style={"fontSize": "3rem", "color": PREFIN_TEXTO_SEC}),
            html.H4("Sin datos cargados", className="mt-3",
                    style={"color": PREFIN_INK}),
            html.P("Ve a la sección Datos para cargar transacciones o generar un conjunto de prueba.",
                   style={"color": PREFIN_TEXTO_SEC, "fontSize": "0.95rem"}),
            dbc.Button([html.I(className="bi bi-arrow-right me-2"), "Ir a Datos"],
                       href="/datos", color="primary", className="mt-2"),
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
                        dcc.Slider(3, 24, 3, value=12, id="sl-meses",
                                   marks={i: str(i) for i in range(3, 25, 3)}),

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
# PÁGINA: DASHBOARD
# ============================================================
def layout_dashboard(df):
    if df is None or df.empty:
        return _pantalla_sin_datos()

    kpis = kpis_globales(df)
    pred = modelo.predecir(df)
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

    # --- Últimas transacciones ---
    df_tabla = df.tail(10).iloc[::-1].copy()
    df_tabla["fecha"] = df_tabla["fecha"].dt.strftime("%d/%m/%Y")
    df_tabla["importe"] = df_tabla["importe"].apply(lambda x: f"{x:+,.2f} €")

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
                             "Recomendado: >15%"), md=3, className="mb-3"),
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
            dbc.CardBody([
                dash_table.DataTable(
                    data=df_tabla[["fecha", "descripcion", "importe", "categoria"]].to_dict("records"),
                    columns=[
                        {"name": "Fecha", "id": "fecha"},
                        {"name": "Descripción", "id": "descripcion"},
                        {"name": "Importe", "id": "importe"},
                        {"name": "Categoría", "id": "categoria"},
                    ],
                    style_table={"overflowX": "auto"},
                    style_cell={
                        "textAlign": "left", "padding": "10px 14px",
                        "fontFamily": "Inter", "fontSize": "0.88rem",
                        "border": "none",
                        "borderBottom": f"1px solid {PREFIN_BORDE}",
                        "color": PREFIN_INK,
                    },
                    style_header={
                        "backgroundColor": "#FAFAF9",
                        "fontWeight": "600",
                        "color": PREFIN_TEXTO_SEC,
                        "fontSize": "0.78rem",
                        "textTransform": "uppercase",
                        "letterSpacing": "0.04em",
                        "border": "none",
                        "borderBottom": f"1px solid {PREFIN_BORDE}",
                    },
                    style_data_conditional=[
                        {"if": {"filter_query": '{importe} contains "+"', "column_id": "importe"},
                         "color": PREFIN_VERDE, "fontWeight": "500"},
                        {"if": {"filter_query": '{importe} contains "-"', "column_id": "importe"},
                         "color": PREFIN_ROJO, "fontWeight": "500"},
                    ],
                    page_size=10,
                ),
            ]),
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

        dbc.Card([
            dbc.CardHeader([
                html.I(className="bi bi-exclamation-triangle me-2"),
                f"Transacciones inusuales ({len(anom_df)})",
            ]),
            dbc.CardBody([
                html.P(
                    "Transacciones con un importe superior a 2.5σ por encima de la media de su categoría.",
                    style={"color": PREFIN_TEXTO_SEC, "fontSize": "0.85rem"},
                ) if len(anom_df) > 0 else None,
                dash_table.DataTable(
                    data=anom_df[["fecha_str", "descripcion", "categoria",
                                  "importe_str", "z_str"]].to_dict("records"),
                    columns=[
                        {"name": "Fecha", "id": "fecha_str"},
                        {"name": "Descripción", "id": "descripcion"},
                        {"name": "Categoría", "id": "categoria"},
                        {"name": "Importe", "id": "importe_str"},
                        {"name": "Desviación", "id": "z_str"},
                    ],
                    style_cell={
                        "textAlign": "left", "padding": "10px 14px",
                        "fontFamily": "Inter", "fontSize": "0.88rem",
                        "border": "none",
                        "borderBottom": f"1px solid {PREFIN_BORDE}",
                        "color": PREFIN_INK,
                    },
                    style_header={
                        "backgroundColor": "#FAFAF9",
                        "fontWeight": "600", "color": PREFIN_TEXTO_SEC,
                        "fontSize": "0.78rem",
                        "textTransform": "uppercase",
                        "letterSpacing": "0.04em",
                        "border": "none",
                        "borderBottom": f"1px solid {PREFIN_BORDE}",
                    },
                    page_size=8,
                ) if not anom_df.empty else dbc.Alert(
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

    pred = modelo.predecir(df)
    nivel = pred["nivel"]
    score = pred["score"]

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
    }
    if imp:
        imp_df = pd.DataFrame([
            {"feature": nombres_es.get(k, k), "importancia": v}
            for k, v in sorted(imp.items(), key=lambda x: -x[1])
        ])
        fig_imp = px.bar(
            imp_df, x="importancia", y="feature", orientation="h",
            title="Importancia de variables · Random Forest",
            color_discrete_sequence=[PREFIN_INK],
        )
        aplicar_layout_plotly(fig_imp)
        fig_imp.update_layout(yaxis={"categoryorder": "total ascending"})
    else:
        fig_imp = go.Figure()
        fig_imp.add_annotation(text="Datos insuficientes",
                                showarrow=False, font=dict(size=13))
        aplicar_layout_plotly(fig_imp)

    # --- Probabilidades ---
    proba = pred.get("probabilidades", {})
    proba_items = []
    for niv, pct in proba.items():
        color_map = {"Bajo": PREFIN_VERDE, "Medio": PREFIN_AMBAR, "Alto": PREFIN_ROJO}
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

    # --- Gasto futuro ---
    gasto_fut = pred.get("gasto_futuro_est")
    gasto_fut_txt = f"{gasto_fut:,.2f} €" if gasto_fut else "—"

    return dbc.Container([
        html.Div([
            html.H1("Predicción de riesgo", className="page-title"),
            html.P("Modelo Random Forest entrenado sobre tu historial financiero mensual.",
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
                    dbc.CardHeader("Gasto estimado · próximo mes"),
                    dbc.CardBody([
                        html.Div(gasto_fut_txt, style={
                            "fontSize": "1.7rem",
                            "fontWeight": "700",
                            "color": PREFIN_INK,
                            "letterSpacing": "-0.025em",
                        }),
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
                    md=12),
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
            dbc.Col([html.Div(id="div-simulacion-resultado")], md=9),
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
            dbc.Col(html.Div(id="div-ahorro-resultado"), md=9),
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
    Input("btn-truelayer", "n_clicks"),
    Input("upload-data", "contents"),
    State("upload-data", "filename"),
    State("sl-meses", "value"),
    State("inp-nomina", "value"),
    State("radio-perfil", "value"),
    State("inp-mes-tl", "value"),
    prevent_initial_call=True,
)
def cargar_datos(n_sint, n_tl, contents, filename, meses, nomina, perfil, mes_tl):
    ctx = callback_context
    trigger = ctx.triggered[0]["prop_id"].split(".")[0] if ctx.triggered else ""

    df = None
    mensaje = ""
    error = False

    try:
        if trigger == "btn-sintetico":
            df = cargar_sinteticos(meses=meses or 12, ingreso_base=nomina or 1800,
                                    perfil=perfil or "medio")
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
        try:
            modelo.entrenar(df)
        except Exception:
            pass
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

    estado = estado_actual(df)
    sim = simular_escenario(
        estado=estado,
        meses=horizonte or 12,
        cambio_ingreso_pct=cambio_ing or 0,
        cambios_categoria=cambios_cat or {},
        evento_imprevisto=imprevisto or 0,
        meta_ahorro_mensual=meta if meta and meta > 0 else None,
    )
    resumen = resumen_simulacion(sim)

    # Saldo
    fig_saldo = px.line(
        sim, x="mes", y="saldo_acumulado", color="escenario",
        title="Proyección de saldo · Actual vs Simulado",
        labels={"saldo_acumulado": "Saldo (€)", "mes": "Mes"},
        color_discrete_map={"Actual": "#A1A1AA", "Simulado": PREFIN_INK},
        markers=True,
    )
    fig_saldo.update_traces(line=dict(width=2))
    aplicar_layout_plotly(fig_saldo)

    # Ahorro
    fig_ahorro = px.bar(
        sim, x="mes", y="ahorro", color="escenario", barmode="group",
        title="Ahorro mensual · Actual vs Simulado",
        color_discrete_map={"Actual": "#A1A1AA", "Simulado": PREFIN_VERDE},
    )
    aplicar_layout_plotly(fig_ahorro)

    delta_saldo = resumen.get("diferencia_saldo", 0)
    mejora_ahorro = resumen.get("mejora_ahorro", 0)

    return html.Div([
        dbc.Row([
            dbc.Col(kpi_card("Saldo actual final",
                             f"{resumen.get('saldo_final_actual', 0):,.2f} €",
                             "bi-wallet2", PREFIN_TEXTO_SEC), md=3, className="mb-3"),
            dbc.Col(kpi_card("Saldo simulado final",
                             f"{resumen.get('saldo_final_simulado', 0):,.2f} €",
                             "bi-wallet-fill", PREFIN_INK), md=3, className="mb-3"),
            dbc.Col(kpi_card("Diferencia",
                             f"{delta_saldo:+,.2f} €",
                             "bi-arrow-left-right",
                             PREFIN_VERDE if delta_saldo >= 0 else PREFIN_ROJO),
                    md=3, className="mb-3"),
            dbc.Col(kpi_card("Mejora ahorro total",
                             f"{mejora_ahorro:+,.2f} €",
                             "bi-piggy-bank",
                             PREFIN_VERDE if mejora_ahorro >= 0 else PREFIN_ROJO),
                    md=3, className="mb-3"),
        ]),
        dbc.Row([
            dbc.Col(dbc.Card(dbc.CardBody(dcc.Graph(figure=fig_saldo,
                                                    config={"displayModeBar": False}))),
                    md=7, className="mb-3"),
            dbc.Col(dbc.Card(dbc.CardBody(dcc.Graph(figure=fig_ahorro,
                                                    config={"displayModeBar": False}))),
                    md=5, className="mb-3"),
        ]),
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
        color_discrete_sequence=[PREFIN_INK],
    )
    aplicar_layout_plotly(fig_cat)
    fig_cat.update_layout(yaxis={"categoryorder": "total ascending"})

    # Metas
    objetivo_items = []
    for obj in objetivos:
        meses = obj["meses"]
        meses_txt = (f"{meses:.0f} meses" if obj["alcanzable"]
                     and isinstance(meses, (int, float)) else "—")
        objetivo_items.append(html.Div([
            html.Div([
                html.Strong(obj["nombre"], style={"fontSize": "0.9rem"}),
                html.Span(f"{obj['importe']:,} €",
                          style={"color": PREFIN_TEXTO_SEC,
                                  "fontSize": "0.85rem",
                                  "marginLeft": "auto"}),
            ], style={"display": "flex"}),
            html.Div([
                html.I(className="bi bi-clock me-1",
                       style={"color": PREFIN_TEXTO_SEC, "fontSize": "0.75rem"}),
                html.Span(f"Tiempo estimado: {meses_txt}",
                          style={"color": PREFIN_TEXTO_SEC,
                                  "fontSize": "0.8rem"}),
            ], style={"marginTop": "4px"}),
        ], style={
            "padding": "11px 14px",
            "borderBottom": f"1px solid {PREFIN_BORDE}",
        }))

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
            dbc.CardBody(objetivo_items, style={"padding": "0"}),
        ]),
    ])


# ============================================================
# PUNTO DE ENTRADA
# ============================================================
if __name__ == "__main__":
   app.run(debug=False, port=8050)
