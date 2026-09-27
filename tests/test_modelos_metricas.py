"""Pruebas unitarias de los modelos, el baseline y las métricas (src/models/modelos.py y src/evaluation/metricas.py)."""

import pandas as pd
import pytest

from src.evaluation.metricas import calcular_metricas
from src.models.modelos import GRILLAS, crear_modelos, crear_pipeline, regla_rural


def test_metricas_con_valores_conocidos():
    metricas = calcular_metricas([1, 1, 0, 0], [1, 0, 1, 0])
    assert metricas["accuracy"] == metricas["precision"] == metricas["recall"] == metricas["f1"] == 0.5
    # Sin probabilidades no se calcula el ROC-AUC (caso de la regla rural)
    assert "roc_auc" not in metricas


def test_metricas_con_probabilidades_incluyen_roc_auc():
    metricas = calcular_metricas([1, 1, 0, 0], [1, 1, 0, 0], [0.9, 0.8, 0.3, 0.1])
    assert metricas["f1"] == 1.0
    assert metricas["roc_auc"] == 1.0


def test_precision_sin_predicciones_positivas_es_cero():
    """Un modelo que no marca a ningún hogar como pobre no debe generar error."""
    metricas = calcular_metricas([1, 0, 0], [0, 0, 0])
    assert metricas["precision"] == 0
    assert metricas["recall"] == 0


def test_regla_rural_marca_a_los_hogares_rurales():
    hogares = pd.DataFrame({"area_rural": [0, 1, 1, 0]})
    assert regla_rural(hogares).tolist() == [0, 1, 1, 0]


def test_pipeline_une_preprocesamiento_y_modelo():
    pipeline = crear_pipeline(crear_modelos()["Regresión logística"])
    assert [nombre for nombre, _ in pipeline.steps] == ["preprocesamiento", "modelo"]


@pytest.mark.parametrize("nombre", list(GRILLAS))
def test_grillas_usan_hiperparametros_validos(nombre):
    """Cada hiperparámetro de GridSearchCV existe en el pipeline (evita errores de escritura)."""
    parametros = crear_pipeline(crear_modelos()[nombre]).get_params()
    assert set(GRILLAS[nombre]) <= set(parametros)


def test_hay_una_grilla_por_modelo():
    assert set(GRILLAS) == set(crear_modelos())
