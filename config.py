# config.py — Constantes globales de PREFIN
# Paleta minimalista basada en grises neutros + un acento oscuro elegante

# ---------------------------------------------------------------------------
# Categorías
# ---------------------------------------------------------------------------
CATEGORIAS = [
    "Supermercado",
    "Restaurantes y Ocio",
    "Transporte",
    "Servicios del Hogar",
    "Suscripciones",
    "Salud y Farmacia",
    "Ropa y Compras",
    "Educación",
    "Transferencias",
    "Ingresos",
    "Otros",
]

# ---------------------------------------------------------------------------
# Paleta principal — sistema por CAPAS (base fría slate + acento índigo)
# El significado del color semántico (verde/ámbar/rojo) NO cambia.
# ---------------------------------------------------------------------------
PREFIN_INK         = "#0F172A"   # Texto principal (slate-900)
PREFIN_FONDO       = "#E9EEF6"   # Fondo base (slate suave, claramente no blanco)
PREFIN_SUPERFICIE  = "#FFFFFF"   # Tarjetas (un plano por encima del fondo)
PREFIN_NAVBAR      = "#14213D"   # Azul marino de la barra de navegación
PREFIN_SUPERFICIE_ELEV = "#FFFFFF"  # Superficie elevada (se distingue por sombra)
PREFIN_BORDE       = "#E2E8F0"   # Bordes sutiles (slate-200)
PREFIN_BORDE_FUERTE = "#CBD5E1"  # Bordes con más presencia (slate-300)
PREFIN_TEXTO_SEC   = "#64748B"   # Texto secundario (slate-500)
PREFIN_TEXTO_MUTED = "#6B7280"   # Texto tenue — mínimo 4.5:1 sobre blanco (WCAG AA)

# Acento de marca / interacción (índigo), distinto de los semánticos.
PREFIN_ACENTO      = "#6366F1"   # Índigo-500 (marca, elementos interactivos)
PREFIN_ACENTO_OSC  = "#4F46E5"   # Índigo-600 (hover / activo)
PREFIN_ACENTO_SUAVE = "#EEF2FF"  # Índigo-50 (fondos de estado activo)

# Compatibilidad con código antiguo
PREFIN_AZUL    = PREFIN_INK
PREFIN_BLANCO  = PREFIN_SUPERFICIE
PREFIN_GRIS    = "#F1F5F9"

# ---------------------------------------------------------------------------
# Colores semánticos (estado financiero)
# ---------------------------------------------------------------------------
PREFIN_VERDE  = "#16A34A"  # positivo / ingreso / ahorro
PREFIN_ROJO   = "#DC2626"  # negativo / gasto / riesgo alto
PREFIN_AMBAR  = "#B45309"  # advertencia / riesgo medio — amber-700, 5.0:1 sobre blanco (WCAG AA)

# ---------------------------------------------------------------------------
# Colores por categoría — paleta armónica desaturada
# ---------------------------------------------------------------------------
COLORES_CATEGORIA = {
    "Supermercado":         "#10B981",
    "Restaurantes y Ocio":  "#F59E0B",
    "Transporte":           "#3B82F6",
    "Servicios del Hogar":  "#8B5CF6",
    "Suscripciones":        "#EC4899",
    "Salud y Farmacia":     "#06B6D4",
    "Ropa y Compras":       "#F43F5E",
    "Educación":            "#6366F1",
    "Transferencias":       "#64748B",
    "Ingresos":             "#22C55E",
    "Otros":                "#A78BFA",
}

# ---------------------------------------------------------------------------
# Umbrales del modelo de riesgo (% del ingreso mensual)
# ---------------------------------------------------------------------------
UMBRAL_GASTO_TOTAL     = 0.85
UMBRAL_OCIO            = 0.20
UMBRAL_AHORRO_MINIMO   = 0.10

COLOR_RIESGO = {
    "Bajo":   PREFIN_VERDE,
    "Medio":  PREFIN_AMBAR,
    "Alto":   PREFIN_ROJO,
}

# ---------------------------------------------------------------------------
# Backend
# ---------------------------------------------------------------------------
BACKEND_URL = "http://localhost:8000"

# ---------------------------------------------------------------------------
# Plotly: layout por defecto para todas las gráficas
# ---------------------------------------------------------------------------
PLOTLY_LAYOUT = dict(
    paper_bgcolor=PREFIN_SUPERFICIE,
    plot_bgcolor=PREFIN_SUPERFICIE,
    font=dict(family="Inter, -apple-system, BlinkMacSystemFont, sans-serif",
              size=12, color=PREFIN_INK),
    margin=dict(t=50, b=40, l=50, r=30),
    xaxis=dict(showgrid=False, linecolor=PREFIN_BORDE, automargin=True,
               tickfont=dict(color=PREFIN_TEXTO_SEC)),
    yaxis=dict(showgrid=True, gridcolor=PREFIN_BORDE, linecolor=PREFIN_BORDE,
               automargin=True, zeroline=False,
               tickfont=dict(color=PREFIN_TEXTO_SEC)),
    title=dict(font=dict(size=19, color=PREFIN_INK,
                         family="Sora, Inter, sans-serif"),
               x=0.02, xanchor="left", pad=dict(b=10)),
    legend=dict(font=dict(size=11, color=PREFIN_INK)),
    colorway=[PREFIN_ACENTO, PREFIN_INK, PREFIN_VERDE, PREFIN_AMBAR, PREFIN_ROJO,
              "#8B5CF6", "#64748B"],
)

# Escala secuencial de marca (índigo) para rankings con gradación de intensidad.
ESCALA_INTENSIDAD = [
    [0.0, "#E0E7FF"], [0.5, "#818CF8"], [1.0, PREFIN_ACENTO_OSC],
]

# ---------------------------------------------------------------------------
# Tema único de Plotly compartido por TODA la app (frontend y módulos).
# Registrarlo como plantilla por defecto garantiza una estética coherente sin
# tener que aplicarla figura a figura.
# ---------------------------------------------------------------------------
try:
    import plotly.graph_objects as _go
    import plotly.io as _pio
    _pio.templates["prefin"] = _go.layout.Template(layout=PLOTLY_LAYOUT)
    _pio.templates.default = "prefin"
except Exception:  # pragma: no cover - plotly siempre presente en runtime
    pass
