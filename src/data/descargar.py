"""
Descarga de los módulos de la ENAHO desde el portal de microdatos del INEI.

Fuente oficial: https://proyectos.inei.gob.pe/microdatos/
Cada módulo se publica como un archivo ZIP que contiene el CSV de datos,
el diccionario de variables (PDF) y la ficha técnica de la encuesta.
"""

import os
import urllib.request
import zipfile

# URL base de descarga en formato CSV del portal de microdatos del INEI
URL_BASE = "https://proyectos.inei.gob.pe/iinei/srienaho/descarga/CSV/{codigo}-Modulo{modulo}.zip"

# Código interno que el INEI asigna a cada ENAHO anual (metodología actualizada)
CODIGOS_ENAHO = {
    2024: 966,
    2025: 1031,
}

# Módulos que utiliza el proyecto
MODULOS = {
    "01": "Características de la vivienda y del hogar",
    "02": "Características de los miembros del hogar",
    "03": "Educación",
    "18": "Equipamiento del hogar",
    "34": "Sumarias (variables calculadas, incluye la condición de pobreza)",
    "37": "Programas sociales (solo para el análisis del problema, no como predictora)",
}

# El servidor del INEI rechaza descargas sin un navegador identificado
CABECERAS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}


def descargar_modulo(anio, modulo, carpeta_destino):
    """Descarga y descomprime un módulo de la ENAHO. Devuelve la carpeta con los archivos."""
    codigo = CODIGOS_ENAHO[anio]
    url = URL_BASE.format(codigo=codigo, modulo=modulo)
    carpeta_anio = os.path.join(carpeta_destino, f"enaho_{anio}")
    os.makedirs(carpeta_anio, exist_ok=True)

    ruta_zip = os.path.join(carpeta_anio, f"{codigo}-Modulo{modulo}.zip")

    # Si el ZIP ya existe, no se vuelve a descargar
    if not os.path.exists(ruta_zip):
        print(f"Descargando módulo {modulo} de la ENAHO {anio}...")
        solicitud = urllib.request.Request(url, headers=CABECERAS)
        with urllib.request.urlopen(solicitud, timeout=300) as respuesta, open(ruta_zip, "wb") as archivo:
            archivo.write(respuesta.read())

    # Descomprimimos el contenido dentro de la carpeta del año
    with zipfile.ZipFile(ruta_zip) as zip_modulo:
        zip_modulo.extractall(carpeta_anio)

    return os.path.join(carpeta_anio, f"{codigo}-Modulo{modulo}")


def descargar_enaho(anio=2025, carpeta_destino="data/raw", modulos=None):
    """Descarga todos los módulos que utiliza el proyecto para un año de la ENAHO."""
    if modulos is None:
        modulos = list(MODULOS.keys())

    carpetas = {}
    for modulo in modulos:
        carpetas[modulo] = descargar_modulo(anio, modulo, carpeta_destino)
    return carpetas


if __name__ == "__main__":
    # Permite ejecutar la descarga desde la terminal: python src/data/descargar.py
    rutas = descargar_enaho(2025)
    for modulo, ruta in rutas.items():
        print(f"Módulo {modulo}: {ruta}")
