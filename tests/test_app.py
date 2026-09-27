"""
Pruebas funcionales del prototipo (app/app.py).

Usan AppTest de Streamlit, que ejecuta la app sin abrir el navegador y permite
llenar la ficha, pulsar botones y leer lo que se muestra, como lo haría un usuario.
"""

import pytest
from streamlit.testing.v1 import AppTest

from conftest import RAIZ

RUTA_APP = str(RAIZ / "app" / "app.py")


@pytest.fixture
def app():
    return AppTest.from_file(RUTA_APP, default_timeout=60).run()


def texto_de_la_pagina(app):
    return " ".join(elemento.value for elemento in app.markdown)


def test_la_app_abre_sin_errores(app):
    assert not app.exception
    assert app.metric[0].label == "Probabilidad estimada de pobreza"
    # El hogar típico de Lima Metropolitana no se prioriza
    assert "No priorizar" in texto_de_la_pagina(app)


def test_hogar_con_carencias_se_prioriza(app):
    app.number_input(key="n_miembros").set_value(7)
    app.number_input(key="n_menores_14").set_value(4)
    app.number_input(key="n_habitaciones").set_value(1)
    app.selectbox(key="material_piso").set_value(6)            # tierra
    app.selectbox(key="combustible_cocina").set_value(6)       # leña
    for articulo in ["tiene_computadora", "tiene_refrigeradora", "tiene_lavadora", "tiene_internet"]:
        app.checkbox(key=articulo).uncheck()
    app.run()

    assert not app.exception
    assert "Priorizar para verificación" in texto_de_la_pagina(app)


def test_composicion_imposible_muestra_un_error(app):
    app.number_input(key="n_menores_14").set_value(5)   # el hogar típico tiene 3 miembros
    app.run()

    assert "no pueden superar" in app.error[0].value


def test_el_dominio_depende_del_departamento(app):
    app.selectbox(key="departamento").set_value(21).run()   # Puno: Sierra Sur o Selva

    assert app.selectbox(key="dominio").value == 6
    assert app.selectbox(key="dominio").options == ["Sierra Sur", "Selva"]


def test_cargar_un_hogar_real_muestra_su_condicion_oficial(app):
    boton = next(b for b in app.button if b.label.startswith("Cargar un hogar real"))
    boton.click().run()

    assert not app.exception
    assert "Condición oficial según el INEI" in app.info[0].value


def test_lista_de_hogares_evalua_la_muestra_de_la_enaho(app):
    metricas = {metrica.label: metrica.value for metrica in app.metric}
    assert metricas["Hogares evaluados"] == "300"
    assert "Hogares pobres detectados (INEI)" in metricas
