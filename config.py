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
# Paleta principal — minimalista
# ---------------------------------------------------------------------------
PREFIN_INK       = "#0F172A"   # Casi negro azulado (texto principal, acentos)
PREFIN_ACENTO    = "#0F172A"   # Color de marca
PREFIN_FONDO     = "#FAFAF9"   # Fondo general (blanco roto cálido)
PREFIN_SUPERFICIE= "#FFFFFF"   # Tarjetas
PREFIN_BORDE     = "#E7E5E4"   # Bordes sutiles
PREFIN_TEXTO_SEC = "#71717A"   # Texto secundario
PREFIN_TEXTO_MUTED = "#A1A1AA" # Texto muy tenue

# Compatibilidad con código antiguo
PREFIN_AZUL    = PREFIN_INK
PREFIN_BLANCO  = PREFIN_SUPERFICIE
PREFIN_GRIS    = "#F4F4F5"

# ---------------------------------------------------------------------------
# Colores semánticos (estado financiero)
# ---------------------------------------------------------------------------
PREFIN_VERDE  = "#16A34A"  # positivo / ingreso / ahorro
PREFIN_ROJO   = "#DC2626"  # negativo / gasto / riesgo alto
PREFIN_AMBAR  = "#D97706"  # advertencia / riesgo medio

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
    xaxis=dict(showgrid=False, linecolor=PREFIN_BORDE,
               tickfont=dict(color=PREFIN_TEXTO_SEC)),
    yaxis=dict(showgrid=True, gridcolor=PREFIN_BORDE, linecolor=PREFIN_BORDE,
               tickfont=dict(color=PREFIN_TEXTO_SEC)),
    title=dict(font=dict(size=15, color=PREFIN_INK), x=0.02, xanchor="left"),
    legend=dict(font=dict(size=11, color=PREFIN_INK)),
)
