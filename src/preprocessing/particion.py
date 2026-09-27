"""
División de los datos en entrenamiento y prueba, e identificación de los hogares
del panel para la validación temporal.

La partición se guarda en un archivo (llaves de hogar + conjunto) para que todos los
cuadernos usen exactamente los mismos hogares de entrenamiento y de prueba.
"""

import pandas as pd
from sklearn.model_selection import train_test_split

from src.config import PROPORCION_PRUEBA, SEMILLA
from src.data.cargar import leer_csv_enaho, ruta_modulo
from src.features.construir import CLAVES_HOGAR, VARIABLE_OBJETIVO


def dividir_entrenamiento_prueba(hogares):
    """División estratificada: la proporción de hogares pobres se mantiene en ambos conjuntos."""
    return train_test_split(
        hogares,
        test_size=PROPORCION_PRUEBA,
        random_state=SEMILLA,
        stratify=hogares[VARIABLE_OBJETIVO],
    )


def guardar_particion(entrenamiento, prueba, ruta):
    """Guarda las llaves de los hogares de cada conjunto."""
    particion = pd.concat([
        entrenamiento[CLAVES_HOGAR].assign(conjunto="entrenamiento"),
        prueba[CLAVES_HOGAR].assign(conjunto="prueba"),
    ])
    particion.to_csv(ruta, index=False)
    return particion


def aplicar_particion(hogares, ruta):
    """Separa el dataset en entrenamiento y prueba usando la partición guardada."""
    particion = pd.read_csv(ruta)
    datos = hogares.merge(particion, on=CLAVES_HOGAR, how="inner")
    entrenamiento = datos[datos["conjunto"] == "entrenamiento"].drop(columns="conjunto")
    prueba = datos[datos["conjunto"] == "prueba"].drop(columns="conjunto")
    return entrenamiento.reset_index(drop=True), prueba.reset_index(drop=True)


def identificar_hogares_panel(anio=2025, anio_anterior=2024, carpeta_raw="data/raw"):
    """
    Marca los hogares de `anio` que también fueron entrevistados en `anio_anterior`.

    La ENAHO vuelve a entrevistar a una parte de los hogares cada año (muestra panel).
    Un hogar se considera del panel si declara haber sido entrevistado el año anterior
    (PANEL = 1) o si su llave aparece en la encuesta del año anterior.
    """
    vivienda = leer_csv_enaho(ruta_modulo(anio, "vivienda", carpeta_raw), columnas=CLAVES_HOGAR + ["PANEL"])
    sumaria = leer_csv_enaho(ruta_modulo(anio, "sumaria", carpeta_raw), columnas=CLAVES_HOGAR)
    sumaria_anterior = leer_csv_enaho(ruta_modulo(anio_anterior, "sumaria", carpeta_raw), columnas=CLAVES_HOGAR)

    hogares = sumaria.merge(vivienda, on=CLAVES_HOGAR, how="left")
    declara_panel = pd.to_numeric(hogares["PANEL"].astype(str).str.strip(), errors="coerce") == 1

    llaves_anteriores = sumaria_anterior.assign(en_anio_anterior=True)
    hogares = hogares.merge(llaves_anteriores, on=CLAVES_HOGAR, how="left")
    en_anio_anterior = hogares["en_anio_anterior"].eq(True)

    hogares["es_panel"] = (declara_panel | en_anio_anterior).astype(int)
    return hogares[CLAVES_HOGAR + ["es_panel"]]
