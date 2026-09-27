"""Pruebas unitarias de la construcción de variables (src/features/construir.py)."""

import numpy as np
import pandas as pd

from src.features.construir import (
    CLAVES_HOGAR,
    DESCRIPCIONES,
    VARIABLE_OBJETIVO,
    VARIABLES_PREDICTORAS,
    agregar_variables_derivadas,
    anular_codigos_invalidos,
    diccionario_de_datos,
    variables_contexto,
    variables_equipamiento,
    variables_miembros,
)

LLAVE = {"CONGLOME": 1, "VIVIENDA": 1, "HOGAR": 1}


def test_contexto_define_objetivo_y_area_segun_el_inei():
    sumaria = pd.DataFrame({
        "CONGLOME": [1, 2, 3], "VIVIENDA": [1, 1, 1], "HOGAR": [1, 1, 1],
        "MIEPERHO": [4, 2, 5], "DOMINIO": [8, 4, 6], "ESTRATO": [1, 5, 6],
        "UBIGEO": ["150101", "60101", "210101"], "FACTOR07": [100.0, 80.0, 50.0],
        "POBREZA": [1, 2, 3],
    })

    contexto = variables_contexto(sumaria)

    # POBREZA 1 (extremo) y 2 (no extremo) son pobres; 3 es no pobre
    assert contexto[VARIABLE_OBJETIVO].tolist() == [1, 1, 0]
    # Estratos 1 a 5 son urbanos y 6 a 8 rurales
    assert contexto["area_rural"].tolist() == [0, 0, 1]
    # El ubigeo recupera su cero inicial y da el departamento
    assert contexto["ubigeo"].tolist() == ["150101", "060101", "210101"]
    assert contexto["departamento"].tolist() == [15, 6, 21]


def test_contexto_no_deja_pasar_variables_monetarias():
    """Prevención de data leakage: gasto, ingreso y líneas de pobreza no llegan a las predictoras."""
    sumaria = pd.DataFrame({
        "CONGLOME": [1], "VIVIENDA": [1], "HOGAR": [1], "MIEPERHO": [3], "DOMINIO": [8], "ESTRATO": [1],
        "UBIGEO": ["150101"], "FACTOR07": [100.0], "POBREZA": [3],
        "GASHOG2D": [30000.0], "INGHOG2D": [40000.0], "LINEA": [500.0], "LINPE": [250.0],
    })

    contexto = variables_contexto(sumaria)

    predictoras = set(contexto.columns) & set(VARIABLES_PREDICTORAS)
    assert predictoras == {"n_miembros", "dominio", "area_rural", "departamento"}
    assert not {"GASHOG2D", "INGHOG2D", "LINEA", "LINPE"} & set(contexto.columns)


def test_miembros_excluye_trabajadores_del_hogar_y_no_residentes():
    miembros = pd.DataFrame([
        {**LLAVE, "P203": 1, "P204": 1, "P207": 2, "P208A": 40},   # jefa del hogar
        {**LLAVE, "P203": 2, "P204": 1, "P207": 1, "P208A": 42},   # cónyuge
        {**LLAVE, "P203": 3, "P204": 1, "P207": 1, "P208A": 10},   # hijo
        {**LLAVE, "P203": 7, "P204": 1, "P207": 2, "P208A": 70},   # abuela
        {**LLAVE, "P203": 8, "P204": 1, "P207": 2, "P208A": 30},   # trabajadora del hogar: se excluye
        {**LLAVE, "P203": 3, "P204": 2, "P207": 1, "P208A": 5},    # no es miembro del hogar: se excluye
    ])

    fila = variables_miembros(miembros).iloc[0]

    assert fila["n_menores_14"] == 1
    assert fila["n_mayores_65"] == 1
    assert fila["n_edad_activa"] == 2
    assert fila["jefe_edad"] == 40
    assert fila["jefe_mujer"] == 1


def test_equipamiento_convierte_respuestas_y_cuenta_articulos():
    otro_hogar = {"CONGLOME": 2, "VIVIENDA": 1, "HOGAR": 1}
    equipamiento = pd.DataFrame([
        {**LLAVE, "P612N": 1, "P612": 1},    # radio: sí
        {**LLAVE, "P612N": 2, "P612": 2},    # TV a color: no
        {**LLAVE, "P612N": 7, "P612": 9},    # computadora: sin respuesta
        {**LLAVE, "P612N": 3, "P612": 1},    # TV blanco y negro (no seleccionado): sí, cuenta en el total
        {**otro_hogar, "P612N": 7, "P612": 1},
    ])

    fila = variables_equipamiento(equipamiento).set_index("CONGLOME").loc[1]

    assert fila["tiene_radio"] == 1
    assert fila["tiene_tv_color"] == 0
    assert np.isnan(fila["tiene_computadora"])
    assert fila["n_equipos"] == 2


def test_variables_derivadas():
    hogares = pd.DataFrame({
        "n_miembros": [4, 3, 2], "n_habitaciones": [2, 0, 1],
        "n_menores_14": [2, 0, 0], "n_mayores_65": [0, 1, 2], "n_edad_activa": [2, 2, 0],
    })

    derivadas = agregar_variables_derivadas(hogares)

    assert derivadas["personas_por_habitacion"].iloc[0] == 2.0
    # Con 0 habitaciones el hacinamiento queda como faltante (no se divide entre cero)
    assert np.isnan(derivadas["personas_por_habitacion"].iloc[1])
    assert derivadas["tasa_dependencia"].tolist()[:2] == [1.0, 0.5]
    # Si nadie está en edad activa, la dependencia es el número de dependientes
    assert derivadas["tasa_dependencia"].iloc[2] == 2.0


def test_codigo_invalido_pasa_a_faltante(hogares_ficticios):
    hogares = hogares_ficticios.head(2).copy()
    hogares["combustible_cocina"] = [9.0, 2.0]   # 9 no existe en el diccionario del INEI

    limpios = anular_codigos_invalidos(hogares)

    assert np.isnan(limpios["combustible_cocina"].iloc[0])
    assert limpios["combustible_cocina"].iloc[1] == 2.0


def test_predictoras_sin_identificadores_ni_objetivo():
    prohibidas = set(CLAVES_HOGAR) | {"ubigeo", "factor07", "pobreza_inei", VARIABLE_OBJETIVO}
    assert not prohibidas & set(VARIABLES_PREDICTORAS)
    assert len(VARIABLES_PREDICTORAS) == len(set(VARIABLES_PREDICTORAS)) == 36


def test_diccionario_describe_todas_las_columnas():
    diccionario = diccionario_de_datos()
    assert set(VARIABLES_PREDICTORAS) <= set(diccionario["variable"])
    assert set(diccionario["variable"]) == set(DESCRIPCIONES)
