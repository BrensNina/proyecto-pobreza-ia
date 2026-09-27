"""
Lectura de los archivos CSV de la ENAHO.

El INEI no publica todos los años con el mismo formato:
- ENAHO 2024: separador "," y decimales con punto.
- ENAHO 2025: separador ";" y decimales con coma.
Por eso detectamos el formato leyendo la primera línea del archivo.
"""

import os

import pandas as pd

# Nombre de cada archivo CSV dentro de su módulo (el año se completa después)
ARCHIVOS_MODULO = {
    "vivienda": ("01", "Enaho01-{anio}-100.csv"),
    "miembros": ("02", "Enaho01-{anio}-200.csv"),
    "educacion": ("03", "Enaho01A-{anio}-300.csv"),
    "equipamiento": ("18", "Enaho01-{anio}-612.csv"),
    "sumaria": ("34", "Sumaria-{anio}.csv"),
    "programas": ("37", "Enaho01-{anio}-700B.csv"),
}


def leer_csv_enaho(ruta, columnas=None):
    """Lee un CSV de la ENAHO detectando el separador y el tipo de decimal."""
    with open(ruta, encoding="latin-1") as archivo:
        encabezado = archivo.readline()

    if ";" in encabezado:
        separador, decimal = ";", ","
    else:
        separador, decimal = ",", "."

    # Las columnas se comparan en mayúsculas porque el INEI mezcla mayúsculas y minúsculas
    usecols = None
    if columnas is not None:
        columnas_mayus = {c.upper() for c in columnas}
        usecols = lambda c: c.upper() in columnas_mayus

    df = pd.read_csv(
        ruta,
        sep=separador,
        decimal=decimal,
        encoding="latin-1",
        usecols=usecols,
        dtype={"UBIGEO": str, "ubigeo": str},
        low_memory=False,
    )
    df.columns = df.columns.str.upper()
    return df


def ruta_modulo(anio, nombre, carpeta_raw="data/raw"):
    """Construye la ruta del CSV de un módulo a partir del año."""
    from src.data.descargar import CODIGOS_ENAHO

    modulo, archivo = ARCHIVOS_MODULO[nombre]
    codigo = CODIGOS_ENAHO[anio]
    return os.path.join(carpeta_raw, f"enaho_{anio}", f"{codigo}-Modulo{modulo}", archivo.format(anio=anio))


def cargar_modulos(anio=2025, carpeta_raw="data/raw", nombres=None):
    """Carga en un diccionario los DataFrames de los módulos indicados."""
    if nombres is None:
        nombres = list(ARCHIVOS_MODULO.keys())
    return {nombre: leer_csv_enaho(ruta_modulo(anio, nombre, carpeta_raw)) for nombre in nombres}
