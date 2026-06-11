# data/generator.py — Generador de datos bancarios sintéticos realistas (contexto español)

import random
from datetime import date, timedelta

import numpy as np
import pandas as pd

from config import CATEGORIAS

# ---------------------------------------------------------------------------
# Comercios y montos típicos por categoría
# ---------------------------------------------------------------------------
_COMERCIOS = {
    "Supermercado": [
        ("Mercadona", 35, 110),
        ("Lidl", 20, 70),
        ("Carrefour", 40, 130),
        ("Dia", 15, 60),
        ("Alcampo", 50, 150),
    ],
    "Restaurantes y Ocio": [
        ("Restaurante Casa Pepe", 15, 45),
        ("McDonald's", 6, 14),
        ("Bar El Rincón", 5, 18),
        ("Cine Yelmo", 8, 18),
        ("Spotify", 9.99, 9.99),
        ("Concierto FNAC", 20, 60),
        ("Netflix", 13.99, 17.99),
    ],
    "Transporte": [
        ("Metro Madrid", 1.50, 1.50),
        ("Renfe Cercanías", 2.60, 2.60),
        ("Cabify", 8, 25),
        ("BP Gasolinera", 40, 90),
        ("Repsol", 45, 95),
        ("Parking El Corte Inglés", 5, 18),
    ],
    "Servicios del Hogar": [
        ("Endesa", 60, 130),
        ("Iberdrola", 55, 120),
        ("Canal de Isabel II", 25, 55),
        ("Movistar", 45, 80),
        ("Vodafone", 38, 75),
        ("Gas Natural Fenosa", 40, 90),
    ],
    "Suscripciones": [
        ("Amazon Prime", 4.99, 4.99),
        ("HBO Max", 8.99, 13.99),
        ("Disney+", 8.99, 11.99),
        ("Gym DirectFit", 30, 45),
        ("iCloud", 0.99, 2.99),
        ("Adobe Creative", 24.99, 54.99),
    ],
    "Salud y Farmacia": [
        ("Farmacia Central", 8, 40),
        ("Clínica Sanitas", 30, 90),
        ("Farmacia El Pinar", 5, 25),
        ("Óptica 2000", 50, 200),
        ("Fisioterapia Reyes", 35, 60),
    ],
    "Ropa y Compras": [
        ("Zara", 30, 120),
        ("H&M", 20, 80),
        ("El Corte Inglés", 50, 200),
        ("Decathlon", 25, 100),
        ("FNAC", 20, 80),
        ("Amazon.es", 15, 150),
    ],
    "Educación": [
        ("Udemy", 9.99, 19.99),
        ("Coursera", 39.99, 49.99),
        ("Librería Crisol", 12, 45),
        ("Campus Virtual UC3M", 0, 0),
        ("Clases Academia Oxford", 60, 120),
    ],
    "Transferencias": [
        ("Bizum Familia", 20, 200),
        ("Transferencia Compañera", 50, 300),
        ("Alquiler Piso", 500, 900),
    ],
}

_INGRESOS = [
    ("NÓMINA EMPRESA SA", 1400, 2500),
    ("NÓMINA AUTÓNOMO", 1000, 2000),
    ("Transferencia Padres", 100, 400),
    ("Devolución Hacienda", 80, 350),
    ("Ingreso Bizum", 20, 150),
]

# ---------------------------------------------------------------------------
# Enriquecimiento realista (modo `realista=True`)
# ---------------------------------------------------------------------------
# Estacionalidad anual del gasto variable: multiplicador por mes del año.
# Recoge picos de verano (vacaciones) y diciembre (navidad), y enero más flojo.
_ESTAC_MENSUAL = {
    1: 0.92, 2: 0.95, 3: 1.00, 4: 1.00, 5: 1.03, 6: 1.08,
    7: 1.15, 8: 1.12, 9: 1.00, 10: 1.00, 11: 1.05, 12: 1.30,
}

# Recibos anuales/periódicos grandes (mes_del_año, descripción, categoría, min, max).
_RECIBOS_ANUALES = [
    (6,  "IBI Ayuntamiento",        "Servicios del Hogar", 250, 480),
    (3,  "Seguro Coche",            "Transporte",          300, 520),
    (10, "Seguro Hogar",            "Servicios del Hogar", 150, 280),
]

# Shocks (gastos imprevistos): comercio, categoría, min, max.
_SHOCKS = [
    ("Reparación Avería",   "Servicios del Hogar", 150, 900),
    ("Multa Tráfico",       "Transporte",          100, 500),
    ("Urgencia Dental",     "Salud y Farmacia",    120, 700),
    ("Electrodoméstico",    "Ropa y Compras",      200, 1200),
]


def generar_transacciones(
    meses: int = 12,
    fecha_fin: date = None,
    ingreso_base: float = 1800.0,
    perfil_riesgo: str = "medio",   # "bajo" | "medio" | "alto"
    seed: int = 42,
    realista: bool = False,
    estacionalidad: bool = None,
    shocks: bool = None,
    ingreso_irregular: bool = None,
    saldo_inicial: float = None,
) -> pd.DataFrame:
    """
    Genera un DataFrame con transacciones bancarias sintéticas realistas.

    Parámetros
    ----------
    meses          : número de meses de historia a generar
    fecha_fin      : último día del período (por defecto hoy)
    ingreso_base   : nómina mensual aproximada en €
    perfil_riesgo  : nivel de gasto / riesgo del usuario simulado
    seed           : semilla aleatoria para reproducibilidad
    realista       : si True, activa el enriquecimiento (estacionalidad anual,
                     recibos periódicos, shocks e ingreso irregular). Por
                     defecto False para preservar la reproducibilidad de las
                     fases anteriores y los tests. Las banderas individuales,
                     si se dejan en None, heredan el valor de `realista`.
    estacionalidad : modula el gasto variable por mes del año (navidad, verano).
    shocks         : añade gastos imprevistos puntuales con baja probabilidad.
    ingreso_irregular : varía más la nómina y simula meses de ingreso bajo
                     (típico de autónomos). Útil para generar casos de iliquidez.
    saldo_inicial  : saldo de partida; si None, se usa un offset aleatorio.

    NOTA DE DISEÑO (des-circularización): el enriquecimiento hace que la
    iliquidez emerja del PROCESO (ingresos, gastos, shocks), no de una regla.
    Así la etiqueta de riesgo del modelo (¿iliquidez futura?) es un resultado
    observado, independiente de las features, y no una tautología.
    """
    random.seed(seed)
    np.random.seed(seed)

    # Resolver banderas de enriquecimiento.
    estacionalidad = realista if estacionalidad is None else estacionalidad
    shocks = realista if shocks is None else shocks
    if ingreso_irregular is None:
        ingreso_irregular = False

    fecha_fin = fecha_fin or date.today()
    fecha_inicio = date(fecha_fin.year, fecha_fin.month, 1) - pd.DateOffset(months=meses - 1)
    fecha_inicio = fecha_inicio.date()

    # Multiplicador de gasto según perfil
    mult = {"bajo": 0.70, "medio": 1.00, "alto": 1.35}.get(perfil_riesgo, 1.00)

    filas = []

    # Iterar mes a mes
    cur = fecha_inicio.replace(day=1)
    while cur <= fecha_fin:
        factor_estac = _ESTAC_MENSUAL.get(cur.month, 1.0) if estacionalidad else 1.0

        # ---- Ingresos del mes ----
        nombre_ingreso, min_i, max_i = random.choice(_INGRESOS[:2])
        if ingreso_irregular:
            # Más varianza y, con baja probabilidad, un mes de ingreso bajo.
            importe_nomina = round(random.uniform(ingreso_base * 0.80, ingreso_base * 1.20), 2)
            if random.random() < 0.12:
                importe_nomina = round(importe_nomina * random.uniform(0.4, 0.65), 2)
        else:
            importe_nomina = round(random.uniform(ingreso_base * 0.95, ingreso_base * 1.05), 2)
        dia_nomina = random.randint(25, 28)
        # Día de nómina: ajustar al mes correcto
        try:
            fecha_nomina = cur.replace(day=min(dia_nomina, _dias_en_mes(cur.year, cur.month)))
        except Exception:
            fecha_nomina = cur.replace(day=25)

        filas.append({
            "fecha":       fecha_nomina,
            "descripcion": nombre_ingreso,
            "importe":     +importe_nomina,
            "divisa":      "EUR",
            "categoria":   "Ingresos",
        })

        # ---- Gastos fijos del mes ----
        # Alquiler (si perfil medio/alto)
        if perfil_riesgo != "bajo":
            alquiler = round(random.uniform(500, 850) * mult, 2)
            filas.append({
                "fecha":       cur.replace(day=1),
                "descripcion": "Transferencia Alquiler",
                "importe":     -alquiler,
                "divisa":      "EUR",
                "categoria":   "Transferencias",
            })

        # Servicios del hogar (luz, agua, internet)
        for servicio, comercios in [("Servicios del Hogar", _COMERCIOS["Servicios del Hogar"])]:
            nombre, mn, mx = random.choice(comercios)
            importe = round(random.uniform(mn, mx) * mult, 2)
            filas.append({
                "fecha":       cur.replace(day=random.randint(2, 10)),
                "descripcion": nombre,
                "importe":     -importe,
                "divisa":      "EUR",
                "categoria":   servicio,
            })

        # Suscripciones (1-3 por mes)
        for _ in range(random.randint(1, 3)):
            nombre, mn, mx = random.choice(_COMERCIOS["Suscripciones"])
            importe = round(random.uniform(mn if mn > 0 else 4.99, mx), 2)
            filas.append({
                "fecha":       cur.replace(day=random.randint(1, 28)),
                "descripcion": nombre,
                "importe":     -importe,
                "divisa":      "EUR",
                "categoria":   "Suscripciones",
            })

        # ---- Gastos variables del mes ----
        cat_variable = {
            "Supermercado":        random.randint(4, 8),
            "Restaurantes y Ocio": random.randint(2, 6),
            "Transporte":          random.randint(5, 12),
            "Salud y Farmacia":    random.randint(0, 2),
            "Ropa y Compras":      random.randint(0, 3),
            "Educación":           random.randint(0, 2),
        }

        dias_mes = _dias_en_mes(cur.year, cur.month)
        for cat, n_transacciones in cat_variable.items():
            for _ in range(n_transacciones):
                nombre, mn, mx = random.choice(_COMERCIOS[cat])
                if mn == mx == 0:
                    continue
                importe = round(random.uniform(mn, mx) * mult * factor_estac, 2)
                filas.append({
                    "fecha":       cur.replace(day=random.randint(1, dias_mes)),
                    "descripcion": nombre,
                    "importe":     -importe,
                    "divisa":      "EUR",
                    "categoria":   cat,
                })

        # ---- Recibos anuales/periódicos grandes (modo realista) ----
        if estacionalidad:
            for mes_recibo, desc, categoria, mn, mx in _RECIBOS_ANUALES:
                if cur.month == mes_recibo:
                    filas.append({
                        "fecha":       cur.replace(day=random.randint(5, 20)),
                        "descripcion": desc,
                        "importe":     -round(random.uniform(mn, mx) * mult, 2),
                        "divisa":      "EUR",
                        "categoria":   categoria,
                    })

        # ---- Shocks: gastos imprevistos puntuales ----
        if shocks and random.random() < 0.07:
            desc, categoria, mn, mx = random.choice(_SHOCKS)
            filas.append({
                "fecha":       cur.replace(day=random.randint(1, dias_mes)),
                "descripcion": desc,
                "importe":     -round(random.uniform(mn, mx), 2),
                "divisa":      "EUR",
                "categoria":   categoria,
            })

        # Avanzar al siguiente mes
        if cur.month == 12:
            cur = cur.replace(year=cur.year + 1, month=1)
        else:
            cur = cur.replace(month=cur.month + 1)

    df = pd.DataFrame(filas)
    df["fecha"] = pd.to_datetime(df["fecha"])
    df = df.sort_values("fecha").reset_index(drop=True)
    offset = saldo_inicial if saldo_inicial is not None else round(random.uniform(800, 2500), 2)
    df["saldo_acumulado"] = df["importe"].cumsum() + offset
    return df


def generar_multiusuario(
    n_usuarios: int = 40,
    meses: int = 36,
    seed: int = 0,
    realista: bool = True,
) -> pd.DataFrame:
    """Genera un panel de varios usuarios sintéticos (multi-seed).

    Cada usuario tiene su propio ingreso, perfil, saldo de partida y, con cierta
    probabilidad, ingreso irregular. La variación ENTRE individuos —y los shocks
    dentro de cada uno— hacen que algunos lleguen a iliquidez y otros no, lo que
    genera un problema de clasificación genuino (no trivial) para el modelo de
    riesgo des-circularizado.

    Devuelve un único DataFrame en el esquema canónico más una columna `usuario`
    (entero) que identifica a cada individuo. El resto del pipeline ignora esa
    columna; el entrenamiento del modelo de riesgo itera por usuario.
    """
    rng = random.Random(seed)
    perfiles = ["bajo", "medio", "medio", "alto"]
    dfs = []
    for i in range(n_usuarios):
        ingreso = rng.uniform(1100, 2600)
        perfil = rng.choice(perfiles)
        # Saldo de partida a veces ajustado para crear casos de tensión.
        saldo0 = rng.uniform(150, 3000)
        irregular = rng.random() < 0.30
        df_u = generar_transacciones(
            meses=meses, ingreso_base=ingreso, perfil_riesgo=perfil,
            seed=seed * 1000 + i, realista=realista,
            ingreso_irregular=irregular, saldo_inicial=saldo0,
        )
        df_u["usuario"] = i
        dfs.append(df_u)
    return pd.concat(dfs, ignore_index=True)


def _dias_en_mes(year: int, month: int) -> int:
    import calendar
    return calendar.monthrange(year, month)[1]
