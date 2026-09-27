"""
Uso del modelo final: carga, preparación de los hogares y predicción.

Es el puente entre el modelo entrenado (models/modelo_final.joblib) y el prototipo
(app/app.py). El prototipo no calcula nada por su cuenta: todo pasa por estas
funciones, que también se prueban en tests/.
"""

import json
import os

import joblib
import pandas as pd

from src.config import CARPETA_MODELOS
from src.features.construir import (
    ARTICULOS_EQUIPAMIENTO,
    VARIABLES_PREDICTORAS,
    agregar_variables_derivadas,
    anular_codigos_invalidos,
)

ARCHIVO_MODELO = "modelo_final.joblib"
ARCHIVO_METADATOS = "modelo_final_metadatos.json"


def cargar_modelo_final(carpeta=CARPETA_MODELOS):
    """Devuelve el Pipeline del modelo final y sus metadatos (umbral, métricas, referencia)."""
    modelo = joblib.load(os.path.join(carpeta, ARCHIVO_MODELO))
    with open(os.path.join(carpeta, ARCHIVO_METADATOS), encoding="utf-8") as archivo:
        metadatos = json.load(archivo)
    return modelo, metadatos


def preparar_hogar(respuestas):
    """
    Convierte las respuestas de la ficha de un hogar en la fila que espera el modelo.

    La ficha pide solo datos observables. Las variables derivadas se calculan aquí,
    con las mismas funciones usadas para construir el dataset de entrenamiento:
    - n_equipos: artículos marcados en la ficha + "otros_equipos" (otros artefactos o vehículos);
    - personas_por_habitacion y tasa_dependencia (hacinamiento y dependencia).
    """
    datos = dict(respuestas)
    if datos["n_menores_14"] + datos["n_mayores_65"] > datos["n_miembros"]:
        raise ValueError("Los menores de 14 y los mayores de 65 no pueden superar al total de miembros del hogar.")

    otros_equipos = datos.pop("otros_equipos", 0)
    articulos = pd.Series([datos.get(articulo) for articulo in ARTICULOS_EQUIPAMIENTO.values()], dtype=float)
    datos["n_equipos"] = articulos.sum() + otros_equipos
    datos["n_edad_activa"] = datos["n_miembros"] - datos["n_menores_14"] - datos["n_mayores_65"]

    hogar = agregar_variables_derivadas(pd.DataFrame([datos]))
    return validar_hogares(hogar)


def validar_hogares(hogares):
    """
    Revisa una tabla de hogares antes de predecir.

    - Verifica que estén las 36 variables predictoras.
    - Convierte los valores a número y deja como faltante (NaN) lo que no sea numérico
      o no sea un código válido del INEI. El pipeline imputará esos faltantes.
    """
    faltantes = [variable for variable in VARIABLES_PREDICTORAS if variable not in hogares.columns]
    if faltantes:
        raise ValueError(f"Faltan columnas: {', '.join(faltantes)}")
    datos = hogares[VARIABLES_PREDICTORAS].apply(pd.to_numeric, errors="coerce")
    return anular_codigos_invalidos(datos)


def predecir(modelo, hogares, umbral):
    """Probabilidad de pobreza de cada hogar y decisión de priorizarlo (1) o no (0) según el umbral."""
    probabilidad = modelo.predict_proba(hogares[VARIABLES_PREDICTORAS])[:, 1]
    return pd.DataFrame({
        "probabilidad_pobreza": probabilidad,
        "priorizar": (probabilidad >= umbral).astype(int),
    }, index=hogares.index)
