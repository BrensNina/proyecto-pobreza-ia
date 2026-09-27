"""Pruebas unitarias del preprocesamiento y de la partición de datos (src/preprocessing/)."""

import numpy as np
import pandas as pd

from src.config import PROPORCION_PRUEBA
from src.features.construir import CLAVES_HOGAR, VARIABLE_OBJETIVO, VARIABLES_PREDICTORAS
from src.preprocessing.particion import aplicar_particion, dividir_entrenamiento_prueba, guardar_particion
from src.preprocessing.preprocesador import crear_preprocesador


def test_preprocesador_aprende_solo_del_entrenamiento(hogares_ficticios):
    """Prevención de data leakage: la mediana para imputar sale del entrenamiento, no de la prueba."""
    entrenamiento = hogares_ficticios.iloc[:150].copy()
    prueba = hogares_ficticios.iloc[150:].copy()
    prueba["jefe_edad"] = 99.0   # valores de prueba muy distintos

    preprocesador = crear_preprocesador().fit(entrenamiento[VARIABLES_PREDICTORAS])

    imputador = preprocesador.named_transformers_["numericas"].named_steps["imputador"]
    posicion = list(imputador.feature_names_in_).index("jefe_edad")
    todos = pd.concat([entrenamiento, prueba])
    assert imputador.statistics_[posicion] == entrenamiento["jefe_edad"].median()
    assert imputador.statistics_[posicion] != todos["jefe_edad"].median()


def test_preprocesador_imputa_faltantes(hogares_ficticios):
    datos = hogares_ficticios[VARIABLES_PREDICTORAS].copy()
    preprocesador = crear_preprocesador().fit(datos)

    con_faltantes = datos.head(3).copy()
    con_faltantes[["n_habitaciones", "material_piso", "tiene_refrigeradora"]] = np.nan

    transformado = preprocesador.transform(con_faltantes)
    assert not np.isnan(transformado).any()


def test_preprocesador_tolera_categorias_nuevas(hogares_ficticios):
    datos = hogares_ficticios[VARIABLES_PREDICTORAS].copy()
    datos = datos[datos["material_piso"] != 7]   # el código 7 ("Otro") no aparece en el entrenamiento
    preprocesador = crear_preprocesador().fit(datos)

    hogar = datos.head(1).copy()
    hogar["material_piso"] = 7.0
    transformado = preprocesador.transform(hogar)

    columnas_piso = [i for i, nombre in enumerate(preprocesador.get_feature_names_out())
                     if nombre.startswith("categoricas__material_piso_")]
    assert transformado[0, columnas_piso].sum() == 0


def test_particion_estratificada_y_disjunta(hogares_ficticios):
    entrenamiento, prueba = dividir_entrenamiento_prueba(hogares_ficticios)

    assert len(prueba) == int(len(hogares_ficticios) * PROPORCION_PRUEBA)
    # La proporción de hogares pobres (20 %) se mantiene en ambos conjuntos
    assert entrenamiento[VARIABLE_OBJETIVO].mean() == prueba[VARIABLE_OBJETIVO].mean() == 0.2
    # Ningún hogar está en los dos conjuntos
    llaves_entrenamiento = set(map(tuple, entrenamiento[CLAVES_HOGAR].values))
    llaves_prueba = set(map(tuple, prueba[CLAVES_HOGAR].values))
    assert not llaves_entrenamiento & llaves_prueba


def test_particion_reproducible(hogares_ficticios):
    _, prueba_1 = dividir_entrenamiento_prueba(hogares_ficticios)
    _, prueba_2 = dividir_entrenamiento_prueba(hogares_ficticios)
    assert prueba_1.index.tolist() == prueba_2.index.tolist()


def test_particion_guardada_se_recupera_igual(hogares_ficticios, tmp_path):
    entrenamiento, prueba = dividir_entrenamiento_prueba(hogares_ficticios)
    ruta = tmp_path / "particion.csv"
    guardar_particion(entrenamiento, prueba, ruta)

    entrenamiento_2, prueba_2 = aplicar_particion(hogares_ficticios, ruta)

    ordenar = lambda df: df.sort_values(CLAVES_HOGAR).reset_index(drop=True)
    pd.testing.assert_frame_equal(ordenar(prueba), ordenar(prueba_2))
    assert len(entrenamiento_2) == len(entrenamiento)
