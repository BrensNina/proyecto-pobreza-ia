"""
Datos y recursos compartidos por las pruebas (fixtures de pytest).

Las pruebas unitarias usan tablas pequeñas y ficticias, construidas aquí, porque
solo verifican que el código haga lo que debe (formas, reglas, tipos). No son
evidencia del desempeño del modelo: esa evidencia sale de la ENAHO real y se
verifica en test_validacion_modelo.py.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.config import RUTA_HOGARES, RUTA_PARTICION
from src.features.construir import (
    CLAVES_HOGAR,
    ETIQUETAS,
    VARIABLE_OBJETIVO,
    VARIABLES_BINARIAS,
    VARIABLES_CATEGORICAS,
    VARIABLES_NUMERICAS,
)
from src.models.prediccion import cargar_modelo_final
from src.preprocessing.particion import aplicar_particion

RAIZ = Path(__file__).resolve().parents[1]


@pytest.fixture
def hogares_ficticios():
    """200 hogares ficticios (40 pobres) con las 36 variables predictoras y valores válidos."""
    generador = np.random.default_rng(0)
    n = 200
    datos = {"CONGLOME": np.arange(1, n + 1), "VIVIENDA": np.ones(n, dtype=int), "HOGAR": np.ones(n, dtype=int)}
    for variable in VARIABLES_NUMERICAS:
        datos[variable] = generador.integers(0, 10, n).astype(float)
    for variable in VARIABLES_CATEGORICAS:
        datos[variable] = generador.choice(list(ETIQUETAS[variable].keys()), n).astype(float)
    for variable in VARIABLES_BINARIAS:
        datos[variable] = generador.integers(0, 2, n)
    datos[VARIABLE_OBJETIVO] = np.r_[np.ones(40, dtype=int), np.zeros(160, dtype=int)]
    return pd.DataFrame(datos)[CLAVES_HOGAR + VARIABLES_NUMERICAS + VARIABLES_CATEGORICAS + VARIABLES_BINARIAS
                               + [VARIABLE_OBJETIVO]]


@pytest.fixture
def ficha_hogar():
    """Respuestas de la ficha para un hogar típico de Lima Metropolitana (el que abre el prototipo)."""
    return {
        "departamento": 15, "dominio": 8, "area_rural": 0,
        "n_miembros": 3, "n_menores_14": 0, "n_mayores_65": 0,
        "jefe_edad": 55, "jefe_mujer": 0, "educacion_jefe": 6, "max_educacion_adultos": 8,
        "tipo_vivienda": 1, "material_paredes": 1, "material_piso": 5, "material_techo": 1,
        "tenencia_vivienda": 2, "n_habitaciones": 3,
        "fuente_agua": 1, "desague": 1, "combustible_cocina": 2, "tiene_electricidad": 1,
        "tiene_telefono_fijo": 0, "tiene_celular": 1, "tiene_tv_cable": 0, "tiene_internet": 1,
        "tiene_radio": 0, "tiene_tv_color": 1, "tiene_computadora": 1, "tiene_cocina_gas": 1,
        "tiene_refrigeradora": 1, "tiene_lavadora": 1, "tiene_microondas": 0, "tiene_auto": 0,
        "tiene_motocicleta": 0, "otros_equipos": 2,
    }


@pytest.fixture(scope="session")
def modelo_y_metadatos():
    """Modelo final guardado por notebooks/05_evaluation.ipynb y sus metadatos."""
    return cargar_modelo_final(RAIZ / "models")


@pytest.fixture(scope="session")
def particion_2025():
    """Conjuntos de entrenamiento y prueba reales de la ENAHO 2025 (partición guardada)."""
    hogares = pd.read_csv(RAIZ / RUTA_HOGARES[2025])
    return aplicar_particion(hogares, RAIZ / RUTA_PARTICION)
