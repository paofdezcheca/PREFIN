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


def generar_transacciones(
    meses: int = 12,
    fecha_fin: date = None,
    ingreso_base: float = 1800.0,
    perfil_riesgo: str = "medio",   # "bajo" | "medio" | "alto"
    seed: int = 42,
) -> pd.DataFrame:
    """
    Genera un DataFrame con transacciones bancarias sintéticas realistas.

    Parámetros
    ----------
    meses         : número de meses de historia a generar
    fecha_fin     : último día del período (por defecto hoy)
    ingreso_base  : nómina mensual aproximada en €
    perfil_riesgo : nivel de gasto / riesgo del usuario simulado
    seed          : semilla aleatoria para reproducibilidad
    """
    random.seed(seed)
    np.random.seed(seed)

    fecha_fin = fecha_fin or date.today()
    fecha_inicio = date(fecha_fin.year, fecha_fin.month, 1) - pd.DateOffset(months=meses - 1)
    fecha_inicio = fecha_inicio.date()

    # Multiplicador de gasto según perfil
    mult = {"bajo": 0.70, "medio": 1.00, "alto": 1.35}.get(perfil_riesgo, 1.00)

    filas = []

    # Iterar mes a mes
    cur = fecha_inicio.replace(day=1)
    while cur <= fecha_fin:
        # ---- Ingresos del mes ----
        nombre_ingreso, min_i, max_i = random.choice(_INGRESOS[:2])
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
                importe = round(random.uniform(mn, mx) * mult, 2)
                filas.append({
                    "fecha":       cur.replace(day=random.randint(1, dias_mes)),
                    "descripcion": nombre,
                    "importe":     -importe,
                    "divisa":      "EUR",
                    "categoria":   cat,
                })

        # Avanzar al siguiente mes
        if cur.month == 12:
            cur = cur.replace(year=cur.year + 1, month=1)
        else:
            cur = cur.replace(month=cur.month + 1)

    df = pd.DataFrame(filas)
    df["fecha"] = pd.to_datetime(df["fecha"])
    df = df.sort_values("fecha").reset_index(drop=True)
    df["saldo_acumulado"] = df["importe"].cumsum() + round(random.uniform(800, 2500), 2)
    return df


def _dias_en_mes(year: int, month: int) -> int:
    import calendar
    return calendar.monthrange(year, month)[1]
