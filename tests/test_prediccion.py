"""
Pruebas de integración: modelo final + preparación del hogar + predicción + explicación
(src/models/prediccion.py y src/evaluation/interpretacion.py).
"""

import numpy as np
import pandas as pd
import pytest

from src.evaluation.interpretacion import (
    GRUPO_DE_VARIABLE,
    calcular_referencia,
    explicar_hogar,
    puntaje_referencia,
    resumir_por_grupo,
)
from src.features.construir import VARIABLES_PREDICTORAS
from src.models.prediccion import predecir, preparar_hogar, validar_hogares


def test_modelo_final_espera_las_36_variables(modelo_y_metadatos):
    modelo, metadatos = modelo_y_metadatos
    assert list(modelo.feature_names_in_) == VARIABLES_PREDICTORAS
    assert metadatos["variables_predictoras"] == VARIABLES_PREDICTORAS
    assert metadatos["umbral_decision"] == 0.5


def test_preparar_hogar_calcula_variables_derivadas(ficha_hogar):
    hogar = preparar_hogar(ficha_hogar)

    assert list(hogar.columns) == VARIABLES_PREDICTORAS
    fila = hogar.iloc[0]
    # 5 artículos marcados (TV, computadora, cocina a gas, refrigeradora, lavadora) + 2 otros
    assert fila["n_equipos"] == 7
    assert fila["personas_por_habitacion"] == 1.0
    assert fila["tasa_dependencia"] == 0.0


def test_preparar_hogar_rechaza_composicion_imposible(ficha_hogar):
    ficha_hogar.update({"n_miembros": 2, "n_menores_14": 2, "n_mayores_65": 1})
    with pytest.raises(ValueError, match="no pueden superar"):
        preparar_hogar(ficha_hogar)


def test_prediccion_es_probabilidad_y_respeta_el_umbral(modelo_y_metadatos, ficha_hogar):
    modelo, metadatos = modelo_y_metadatos
    resultado = predecir(modelo, preparar_hogar(ficha_hogar), metadatos["umbral_decision"]).iloc[0]

    assert 0 <= resultado["probabilidad_pobreza"] <= 1
    assert resultado["priorizar"] == int(resultado["probabilidad_pobreza"] >= metadatos["umbral_decision"])


def test_hogar_con_carencias_tiene_mayor_probabilidad(modelo_y_metadatos, ficha_hogar):
    """Prueba de coherencia: más miembros, piso de tierra, leña y sin bienes elevan el riesgo."""
    modelo, metadatos = modelo_y_metadatos
    carencias = dict(ficha_hogar)
    carencias.update({
        "n_miembros": 7, "n_menores_14": 4, "n_habitaciones": 1, "material_piso": 6, "combustible_cocina": 6,
        "fuente_agua": 8, "desague": 9, "tiene_internet": 0, "tiene_computadora": 0, "tiene_refrigeradora": 0,
        "tiene_lavadora": 0, "tiene_cocina_gas": 0, "otros_equipos": 0,
    })

    probabilidad_tipica = predecir(modelo, preparar_hogar(ficha_hogar), 0.5)["probabilidad_pobreza"].iloc[0]
    probabilidad_carencias = predecir(modelo, preparar_hogar(carencias), 0.5)["probabilidad_pobreza"].iloc[0]

    assert probabilidad_carencias > probabilidad_tipica
    assert probabilidad_carencias >= metadatos["umbral_decision"]


def test_explicacion_suma_exactamente_la_prediccion(modelo_y_metadatos, ficha_hogar):
    """Puntaje del hogar = puntaje del hogar promedio + suma de los aportes."""
    modelo, metadatos = modelo_y_metadatos
    referencia = pd.Series(metadatos["referencia_explicacion"])
    hogar = preparar_hogar(ficha_hogar)

    probabilidad = modelo.predict_proba(hogar)[0, 1]
    explicacion = explicar_hogar(modelo, hogar, referencia)

    puntaje = np.log(probabilidad / (1 - probabilidad))
    assert np.isclose(puntaje, puntaje_referencia(modelo, referencia) + explicacion["aporte"].sum())
    # Una fila por variable original, y los grupos suman lo mismo que el detalle
    assert sorted(explicacion["variable"]) == sorted(VARIABLES_PREDICTORAS)
    assert np.isclose(resumir_por_grupo(explicacion).sum(), explicacion["aporte"].sum())


def test_grupos_de_la_explicacion_cubren_todas_las_variables():
    assert set(GRUPO_DE_VARIABLE) == set(VARIABLES_PREDICTORAS)


def test_referencia_guardada_es_el_promedio_del_entrenamiento(modelo_y_metadatos, particion_2025):
    modelo, metadatos = modelo_y_metadatos
    entrenamiento, _ = particion_2025
    referencia = calcular_referencia(modelo, entrenamiento[VARIABLES_PREDICTORAS])
    guardada = pd.Series(metadatos["referencia_explicacion"])[referencia.index]
    assert np.allclose(referencia.values, guardada.values)


def test_validar_hogares_detecta_columnas_faltantes(ficha_hogar):
    hogares = preparar_hogar(ficha_hogar).drop(columns=["material_piso", "jefe_edad"])
    with pytest.raises(ValueError, match="material_piso"):
        validar_hogares(hogares)


def test_validar_hogares_limpia_valores_invalidos(modelo_y_metadatos, ficha_hogar):
    """Robustez: textos y códigos inexistentes pasan a faltantes y el modelo igual responde."""
    modelo, _ = modelo_y_metadatos
    hogares = preparar_hogar(ficha_hogar).astype(object)
    hogares.loc[0, "material_piso"] = 99      # código que no existe
    hogares.loc[0, "jefe_edad"] = "no sabe"   # texto en una variable numérica

    limpios = validar_hogares(hogares)

    assert np.isnan(limpios.loc[0, "material_piso"])
    assert np.isnan(limpios.loc[0, "jefe_edad"])
    assert 0 <= predecir(modelo, limpios, 0.5)["probabilidad_pobreza"].iloc[0] <= 1
