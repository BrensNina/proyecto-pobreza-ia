"""
Pruebas de validación del modelo con los datos reales de la ENAHO 2025.

Verifican que el modelo guardado reproduce los resultados reportados en el informe,
que supera al baseline sin IA y que se cumplen los requisitos no funcionales
(tiempo de respuesta y tolerancia a datos faltantes).
"""

import time

import numpy as np
import pytest

from src.evaluation.metricas import calcular_metricas
from src.features.construir import CLAVES_HOGAR, VARIABLE_OBJETIVO, VARIABLES_PREDICTORAS
from src.models.modelos import regla_rural


@pytest.fixture(scope="module")
def evaluacion_prueba(modelo_y_metadatos, particion_2025):
    modelo, metadatos = modelo_y_metadatos
    _, prueba = particion_2025
    X, y = prueba[VARIABLES_PREDICTORAS], prueba[VARIABLE_OBJETIVO]
    probabilidad = modelo.predict_proba(X)[:, 1]
    prediccion = (probabilidad >= metadatos["umbral_decision"]).astype(int)
    return {
        "modelo": calcular_metricas(y, prediccion, probabilidad),
        "regla": calcular_metricas(y, regla_rural(X)),
    }


def test_particion_real_es_la_reportada(particion_2025):
    entrenamiento, prueba = particion_2025
    assert (len(entrenamiento), len(prueba)) == (26961, 6741)
    llaves_entrenamiento = set(map(tuple, entrenamiento[CLAVES_HOGAR].values))
    llaves_prueba = set(map(tuple, prueba[CLAVES_HOGAR].values))
    assert not llaves_entrenamiento & llaves_prueba
    # Estratificación: 18,1 % de hogares pobres en ambos conjuntos
    assert abs(entrenamiento[VARIABLE_OBJETIVO].mean() - prueba[VARIABLE_OBJETIVO].mean()) < 0.001


def test_modelo_reproduce_las_metricas_reportadas(evaluacion_prueba, modelo_y_metadatos):
    _, metadatos = modelo_y_metadatos
    for metrica, valor_reportado in metadatos["metricas_prueba"].items():
        assert evaluacion_prueba["modelo"][metrica] == pytest.approx(valor_reportado, abs=1e-4), metrica


def test_modelo_supera_al_baseline_sin_ia(evaluacion_prueba):
    """El modelo detecta más hogares pobres y, a la vez, incluye menos hogares no pobres que la regla rural."""
    modelo, regla = evaluacion_prueba["modelo"], evaluacion_prueba["regla"]
    assert modelo["f1"] > regla["f1"] + 0.15
    assert modelo["recall"] > regla["recall"]
    assert modelo["precision"] > regla["precision"]


def test_modelo_ordena_bien_a_los_hogares(evaluacion_prueba):
    assert evaluacion_prueba["modelo"]["roc_auc"] > 0.8


def test_tiempo_de_respuesta_por_hogar(modelo_y_metadatos, particion_2025):
    """Requisito no funcional: responder en menos de 1 segundo por hogar."""
    modelo, _ = modelo_y_metadatos
    _, prueba = particion_2025
    hogar = prueba[VARIABLES_PREDICTORAS].iloc[[0]]

    inicio = time.perf_counter()
    modelo.predict_proba(hogar)
    assert time.perf_counter() - inicio < 1.0


def test_responde_con_datos_faltantes(modelo_y_metadatos, particion_2025):
    """Requisito no funcional: tolerar respuestas vacías en la ficha."""
    modelo, _ = modelo_y_metadatos
    _, prueba = particion_2025
    hogar = prueba[VARIABLES_PREDICTORAS].iloc[[0]].copy()
    hogar[["n_habitaciones", "material_piso", "tiene_refrigeradora", "educacion_jefe"]] = np.nan

    probabilidad = modelo.predict_proba(hogar)[0, 1]
    assert 0 <= probabilidad <= 1
