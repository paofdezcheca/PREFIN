# modules/categorizer.py — Categorizador de transacciones por palabras clave

import re
import pandas as pd

# ---------------------------------------------------------------------------
# Reglas de categorización: lista ordenada de (patrón_regex, categoría)
# Se aplica el primer patrón que coincide (case-insensitive)
# ---------------------------------------------------------------------------
_REGLAS = [
    # INGRESOS
    (r"nómina|nomina|salario|sueldo|ingreso empresa|haberes", "Ingresos"),
    (r"devolución hacienda|devolucion hacienda|agencia tributaria", "Ingresos"),

    # SUPERMERCADO
    (r"mercadona|lidl|carrefour|dia\b|alcampo|eroski|hipercor|aldi|consum|ahorra?más", "Supermercado"),

    # RESTAURANTES Y OCIO
    (r"mcdonald|burger king|kfc|telepizza|domino|glovo|just eat|uber eat", "Restaurantes y Ocio"),
    (r"restaurante|cafetería|cafeteria|bar\b|taberna|cervecería|pizz|sushi|kebab", "Restaurantes y Ocio"),
    (r"cine\b|yelmo|cinesa|kinepolis|teatro|museo|concert|festival|ocio", "Restaurantes y Ocio"),
    (r"netflix|hbo|disney\+|amazon prime video|apple tv", "Restaurantes y Ocio"),

    # TRANSPORTE
    (r"metro|renfe|cercanías|cercanias|autobús|autobus|emt\b|bus\b", "Transporte"),
    (r"cabify|uber\b|bolt|taxi|blablacar|mietwagen", "Transporte"),
    (r"gasolinera|gasolina|bp\b|repsol|cepsa|shell|galp", "Transporte"),
    (r"parking|aparcamiento|autopista|toll|peaje", "Transporte"),
    (r"renfe|ave\b|alvia|intercity", "Transporte"),

    # SERVICIOS DEL HOGAR
    (r"endesa|iberdrola|fenosa|naturgy|gas natural|viesgo", "Servicios del Hogar"),
    (r"canal de isabel|aguas de|suministros de agua", "Servicios del Hogar"),
    (r"movistar|vodafone|orange\b|yoigo|masmovil|pepephone|jazztel", "Servicios del Hogar"),
    (r"alquiler|arrendamiento|comunidad de propietarios|hipoteca", "Servicios del Hogar"),

    # SUSCRIPCIONES
    (r"spotify|apple music|youtube premium|deezer|tidal", "Suscripciones"),
    (r"amazon prime\b|prime video|kindle|audible", "Suscripciones"),
    (r"gym|gimnasio|fitness|holmes place|mcfit|diverxo", "Suscripciones"),
    (r"icloud|google one|dropbox|adobe|microsoft 365|office 365", "Suscripciones"),
    (r"hbo max|disney\+|paramount\+|crunchyroll", "Suscripciones"),

    # SALUD Y FARMACIA
    (r"farmacia|parafarmacia|sanitas|adeslas|mapfre salud|quirón|vithas", "Salud y Farmacia"),
    (r"médico|medico|dentista|clinica|hospital|laboratorio análisis", "Salud y Farmacia"),
    (r"óptica|optica|optician|lentes|gafa", "Salud y Farmacia"),
    (r"fisioter|psicólog|psicolog|nutricionista", "Salud y Farmacia"),

    # ROPA Y COMPRAS
    (r"zara|h&m|mango|bershka|pull.bear|stradivarius|primark", "Ropa y Compras"),
    (r"el corte inglés|corte ingles|fnac|media markt|leroy merlin|ikea", "Ropa y Compras"),
    (r"amazon\.es|amazon.com|aliexpress|shein|zalando|ebay", "Ropa y Compras"),
    (r"decathlon|sport|deporte|running", "Ropa y Compras"),

    # EDUCACIÓN
    (r"universidad|colegio|academia|clases|matrícula|matricula|librería|libreria", "Educación"),
    (r"udemy|coursera|linkedin learning|edx|uned|openwebinars", "Educación"),

    # TRANSFERENCIAS
    (r"bizum|transferencia|traspaso|envío|paypal|wise|revolut", "Transferencias"),
]

_COMPILED = [(re.compile(p, re.IGNORECASE), cat) for p, cat in _REGLAS]


def categorizar_descripcion(descripcion: str) -> str:
    """Devuelve la categoría para una descripción de transacción."""
    if not isinstance(descripcion, str) or not descripcion.strip():
        return "Otros"
    for patron, categoria in _COMPILED:
        if patron.search(descripcion):
            return categoria
    return "Otros"


def categorizar_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Añade/sobreescribe la columna 'categoria' en el DataFrame.
    Si el importe es positivo y no hay categoría clara → Ingresos.
    """
    df = df.copy()
    df["categoria"] = df["descripcion"].apply(categorizar_descripcion)

    # Corregir: importes positivos sin categoría clara → Ingresos
    mask_ingreso = (df["importe"] > 0) & (~df["categoria"].isin(["Ingresos", "Transferencias"]))
    df.loc[mask_ingreso, "categoria"] = "Ingresos"

    return df
