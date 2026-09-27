"""
Prototipo: apoyo a la focalización de hogares en situación de pobreza monetaria.

Aplicación web hecha con Streamlit que usa el modelo final del proyecto
(regresión logística, models/modelo_final.joblib). Tiene tres pestañas:

1. Evaluar un hogar: ficha con las características observables de un hogar.
   Devuelve la probabilidad de pobreza, la decisión de priorizarlo y por qué.
2. Evaluar una lista de hogares: ordena una lista (CSV o muestra real de la ENAHO)
   de mayor a menor probabilidad, como haría un programa social al focalizar.
3. Acerca del modelo: desempeño, equidad, limitaciones y uso responsable.

Toda la lógica (preparar el hogar, predecir y explicar) está en src/ y se prueba
en tests/. Esta app solo organiza la interfaz.

Ejecutar desde la carpeta raíz del proyecto:

    streamlit run app/app.py
"""

import os
import sys

import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import numpy as np
import pandas as pd
import streamlit as st

# La app está en app/; el código del proyecto (src/) está en la carpeta raíz
RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if RAIZ not in sys.path:
    sys.path.insert(0, RAIZ)

from src.config import CARPETA_MODELOS, CARPETA_RESULTADOS, RUTA_HOGARES, RUTA_PARTICION, SEMILLA
from src.evaluation.interpretacion import explicar_hogar, puntaje_referencia, resumir_por_grupo
from src.features.construir import ARTICULOS_EQUIPAMIENTO, CLAVES_HOGAR, ETIQUETAS, VARIABLES_PREDICTORAS
from src.models.prediccion import cargar_modelo_final, predecir, preparar_hogar, validar_hogares
from src.preprocessing.particion import aplicar_particion
from src.visualizacion import COLOR_NO_POBRE, COLOR_POBRE, TEXTO_SECUNDARIO, aplicar_estilo

# ---------------------------------------------------------------------------
# Datos fijos de la ficha
# ---------------------------------------------------------------------------

# Dominios geográficos posibles en cada departamento (observados en la ENAHO 2024 y 2025)
DOMINIOS_POR_DEPARTAMENTO = {
    1: [4, 7], 2: [2, 5], 3: [6], 4: [3, 6], 5: [5, 7], 6: [1, 4, 7], 7: [8], 8: [6, 7], 9: [5],
    10: [5, 7], 11: [2, 5], 12: [5, 7], 13: [1, 4], 14: [1, 4], 15: [2, 5, 8], 16: [7], 17: [7],
    18: [3, 6], 19: [5, 7], 20: [1, 4], 21: [6, 7], 22: [7], 23: [3, 6], 24: [1], 25: [7],
}
LIMA_METROPOLITANA = 8

AYUDA_PROBABILIDAD = ("Por el balanceo de clases, el modelo entrega probabilidades más altas que la frecuencia real "
                      "de pobreza (en promedio 0,37 frente a 0,18). Deben leerse como un puntaje de riesgo para "
                      "ordenar hogares y compararlos con el umbral.")

# Hogar con el que se abre la ficha: valores más frecuentes (o medianas) de los hogares de
# Lima Metropolitana en el entrenamiento
HOGAR_TIPICO = {
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

# Variables Sí/No que se marcan con una casilla (el área y el sexo del jefe(a) usan botones de opción)
CASILLAS = ["tiene_electricidad", "tiene_telefono_fijo", "tiene_celular", "tiene_tv_cable", "tiene_internet"] + \
    list(ARTICULOS_EQUIPAMIENTO.values())

NOMBRES_ARTICULOS = {
    "tiene_radio": "Radio", "tiene_tv_color": "TV a color", "tiene_computadora": "Computadora o laptop",
    "tiene_cocina_gas": "Cocina a gas", "tiene_refrigeradora": "Refrigeradora", "tiene_lavadora": "Lavadora",
    "tiene_microondas": "Microondas", "tiene_auto": "Auto o camioneta", "tiene_motocicleta": "Motocicleta",
}


# ---------------------------------------------------------------------------
# Carga de recursos (se hace una sola vez y queda en memoria)
# ---------------------------------------------------------------------------

@st.cache_resource
def cargar_recursos():
    modelo, metadatos = cargar_modelo_final(os.path.join(RAIZ, CARPETA_MODELOS))
    referencia = pd.Series(metadatos["referencia_explicacion"])
    return modelo, metadatos, referencia


@st.cache_data
def hogares_de_prueba():
    """Hogares del conjunto de prueba de la ENAHO 2025 (nunca usados para entrenar)."""
    hogares = pd.read_csv(os.path.join(RAIZ, RUTA_HOGARES[2025]))
    _, prueba = aplicar_particion(hogares, os.path.join(RAIZ, RUTA_PARTICION))
    return prueba


@st.cache_data
def tabla_resultados(nombre):
    return pd.read_csv(os.path.join(RAIZ, CARPETA_RESULTADOS, "tablas", nombre), index_col=0)


# ---------------------------------------------------------------------------
# Utilidades
# ---------------------------------------------------------------------------

def numero(valor, decimales=2, signo=False):
    """Número con coma decimal y signo menos tipográfico (formato del informe)."""
    texto = f"{valor:+.{decimales}f}" if signo else f"{valor:.{decimales}f}"
    return texto.replace(".", ",").replace("-", "−")


def porcentaje(valor, decimales=1):
    return numero(valor * 100, decimales) + " %"


def cargar_en_ficha(valores):
    """Escribe los valores de un hogar en los campos de la ficha."""
    for variable, valor in valores.items():
        st.session_state[variable] = bool(valor) if variable in CASILLAS else int(valor)


def leer_ficha():
    """Respuestas actuales de la ficha, como números enteros."""
    return {variable: int(st.session_state[variable]) for variable in HOGAR_TIPICO}


def cargar_hogar_tipico():
    cargar_en_ficha(HOGAR_TIPICO)
    st.session_state.pop("hogar_enaho", None)


def cargar_hogar_enaho():
    """Carga en la ficha un hogar real, al azar, del conjunto de prueba de la ENAHO 2025."""
    prueba = hogares_de_prueba()
    # Solo hogares sin datos faltantes y con composición consistente, que la ficha pueda representar
    completos = prueba[prueba[VARIABLES_PREDICTORAS].notna().all(axis=1)
                       & (prueba["n_menores_14"] + prueba["n_mayores_65"] <= prueba["n_miembros"])]
    fila = completos.iloc[np.random.default_rng().integers(len(completos))]

    valores = {variable: fila[variable] for variable in HOGAR_TIPICO if variable != "otros_equipos"}
    marcados = sum(fila[articulo] for articulo in ARTICULOS_EQUIPAMIENTO.values())
    valores["otros_equipos"] = max(0, fila["n_equipos"] - marcados)
    cargar_en_ficha(valores)
    st.session_state["hogar_enaho"] = {"pobre": int(fila["pobre"]), "ficha": leer_ficha()}


def selector(variable, etiqueta, opciones=None, etiquetas=None, ayuda=None):
    """Lista desplegable que guarda el código del INEI y muestra su etiqueta."""
    etiquetas = etiquetas or ETIQUETAS[variable]
    opciones = opciones or list(etiquetas.keys())
    st.selectbox(etiqueta, opciones, format_func=lambda codigo: etiquetas[codigo], key=variable, help=ayuda)


# ---------------------------------------------------------------------------
# Pestaña 1: evaluar un hogar
# ---------------------------------------------------------------------------

def ficha_del_hogar():
    """Campos de la ficha, agrupados en cuatro secciones."""
    ubicacion, vivienda, servicios, bienes = st.tabs(
        ["Ubicación y composición", "Vivienda", "Servicios", "Artefactos y vehículos"])

    with ubicacion:
        col1, col2 = st.columns(2)
        with col1:
            selector("departamento", "Departamento")
            # El dominio depende del departamento; si el elegido no corresponde, se toma el primero posible
            opciones = DOMINIOS_POR_DEPARTAMENTO[st.session_state["departamento"]]
            if st.session_state["dominio"] not in opciones:
                st.session_state["dominio"] = opciones[0]
            selector("dominio", "Dominio geográfico", opciones=opciones,
                     ayuda="Región natural de la ENAHO. Solo se muestran las que existen en el departamento elegido.")
            es_lima = st.session_state["dominio"] == LIMA_METROPOLITANA
            if es_lima:
                st.session_state["area_rural"] = 0
            st.radio("Área", [0, 1], format_func=lambda x: "Rural" if x else "Urbana", key="area_rural",
                     horizontal=True, disabled=es_lima,
                     help="Lima Metropolitana es totalmente urbana en la ENAHO." if es_lima else None)
        with col2:
            st.number_input("Miembros del hogar", 1, 20, step=1, key="n_miembros")
            st.number_input("Menores de 14 años", 0, 15, step=1, key="n_menores_14")
            st.number_input("Mayores de 65 años", 0, 10, step=1, key="n_mayores_65")
        st.markdown("**Jefe(a) del hogar**")
        col1, col2 = st.columns(2)
        with col1:
            st.number_input("Edad del jefe(a)", 14, 100, step=1, key="jefe_edad")
            st.radio("Sexo del jefe(a)", [0, 1], format_func=lambda x: "Mujer" if x else "Hombre",
                     key="jefe_mujer", horizontal=True)
        with col2:
            selector("educacion_jefe", "Nivel educativo del jefe(a)")
            selector("max_educacion_adultos", "Mayor nivel educativo entre los adultos (18+)",
                     etiquetas=ETIQUETAS["educacion_jefe"])

    with vivienda:
        col1, col2 = st.columns(2)
        with col1:
            selector("tipo_vivienda", "Tipo de vivienda")
            selector("material_paredes", "Material de las paredes")
            selector("material_piso", "Material del piso")
        with col2:
            selector("material_techo", "Material del techo")
            selector("tenencia_vivienda", "Tenencia de la vivienda")
            st.number_input("Habitaciones (sin contar baño, cocina ni pasadizos)", 1, 15, step=1,
                            key="n_habitaciones")

    with servicios:
        col1, col2 = st.columns(2)
        with col1:
            selector("fuente_agua", "Procedencia del agua")
            selector("desague", "Servicio higiénico conectado a")
            selector("combustible_cocina", "Combustible que más usan para cocinar")
        with col2:
            st.checkbox("Alumbrado eléctrico", key="tiene_electricidad")
            st.checkbox("Teléfono fijo", key="tiene_telefono_fijo")
            st.checkbox("Teléfono celular", key="tiene_celular")
            st.checkbox("TV por cable o satelital", key="tiene_tv_cable")
            st.checkbox("Internet", key="tiene_internet")

    with bienes:
        st.caption("Marca los artefactos y vehículos que tiene el hogar.")
        columnas = st.columns(3)
        for posicion, articulo in enumerate(ARTICULOS_EQUIPAMIENTO.values()):
            columnas[posicion % 3].checkbox(NOMBRES_ARTICULOS[articulo], key=articulo)
        st.number_input("Otros artefactos o vehículos (plancha, licuadora, equipo de sonido, bicicleta, etc.)",
                        0, 19, step=1, key="otros_equipos")


def grafico_aportes(grupos):
    """Barras horizontales: aporte de cada grupo de características frente al hogar promedio."""
    aplicar_estilo()
    datos = grupos.sort_values()
    fig, ax = plt.subplots(figsize=(5.2, 3.8))
    colores = [COLOR_POBRE if valor > 0 else COLOR_NO_POBRE for valor in datos.values]
    ax.barh(datos.index, datos.values, color=colores, height=0.6)
    ax.axvline(0, color=TEXTO_SECUNDARIO, linewidth=1)

    limite = max(np.abs(datos.values).max(), 0.5) * 1.4
    ax.set_xlim(-limite, limite)
    for posicion, valor in enumerate(datos.values):
        desplazamiento = 0.03 * limite if valor >= 0 else -0.03 * limite
        ax.text(valor + desplazamiento, posicion, numero(valor, signo=True), va="center",
                ha="left" if valor >= 0 else "right", fontsize=10, color=TEXTO_SECUNDARIO)
    ax.text(0, -0.12, "← reduce", transform=ax.transAxes, ha="left", va="top", fontsize=10,
            color=TEXTO_SECUNDARIO)
    ax.text(1, -0.12, "eleva →", transform=ax.transAxes, ha="right", va="top", fontsize=10,
            color=TEXTO_SECUNDARIO)
    ax.tick_params(axis="y", labelsize=10, length=0)
    ax.tick_params(axis="x", labelsize=9)
    ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: numero(x, 0 if float(x).is_integer() else 1)))
    ax.spines["left"].set_visible(False)
    ax.grid(axis="y", visible=False)
    fig.tight_layout()
    return fig


def tarjeta_decision(priorizar, umbral):
    color = COLOR_POBRE if priorizar else COLOR_NO_POBRE
    titulo = "Priorizar para verificación" if priorizar else "No priorizar"
    detalle = (f"La probabilidad supera el umbral de {porcentaje(umbral, 0)}." if priorizar
               else f"La probabilidad está por debajo del umbral de {porcentaje(umbral, 0)}.")
    st.markdown(
        f"<div style='border-left: 6px solid {color}; background: rgba(128, 128, 128, 0.10);"
        f" padding: 0.6rem 1rem; border-radius: 0.4rem; margin-bottom: 0.8rem'>"
        f"<div style='font-size: 1.15rem; font-weight: 700'>{titulo}</div>"
        f"<div style='font-size: 0.9rem'>{detalle}</div></div>",
        unsafe_allow_html=True)


def resultado_del_hogar(modelo, metadatos, referencia):
    """Probabilidad, decisión y explicación del hogar de la ficha."""
    umbral = metadatos["umbral_decision"]
    try:
        hogar = preparar_hogar(leer_ficha())
    except ValueError as error:
        st.error(str(error))
        return

    probabilidad = float(predecir(modelo, hogar, umbral)["probabilidad_pobreza"].iloc[0])
    explicacion = explicar_hogar(modelo, hogar, referencia)
    base = puntaje_referencia(modelo, referencia)
    probabilidad_promedio = 1 / (1 + np.exp(-base))

    st.metric("Probabilidad estimada de pobreza", porcentaje(probabilidad), help=AYUDA_PROBABILIDAD)
    st.progress(probabilidad)
    tarjeta_decision(probabilidad >= umbral, umbral)

    # Si la ficha tiene un hogar real de la ENAHO sin cambios, se muestra su condición oficial
    enaho = st.session_state.get("hogar_enaho")
    if enaho is not None:
        if enaho["ficha"] == leer_ficha():
            condicion = "Pobre" if enaho["pobre"] else "No pobre"
            st.info(f"Hogar real de la ENAHO 2025 (conjunto de prueba). Condición oficial según el INEI: **{condicion}**.")
        else:
            st.caption("Modificaste la ficha del hogar cargado de la ENAHO: su condición oficial ya no aplica.")

    st.markdown("**¿Por qué?** Aporte de cada grupo de características, frente al hogar promedio "
                f"(probabilidad de {porcentaje(probabilidad_promedio)}):")
    figura = grafico_aportes(resumir_por_grupo(explicacion))
    st.pyplot(figura)
    plt.close(figura)

    fila = hogar.iloc[0]
    st.caption(f"Variables calculadas por el sistema: {numero(fila['personas_por_habitacion'])} personas por "
               f"habitación · tasa de dependencia {numero(fila['tasa_dependencia'])} · "
               f"{int(fila['n_equipos'])} artefactos y vehículos.")
    return explicacion


def detalle_de_aportes(explicacion):
    """Tabla con el aporte de cada una de las 36 características."""
    with st.expander("Ver el aporte de cada característica"):
        st.caption("Las características de un mismo grupo están relacionadas entre sí (por ejemplo, el número de "
                   "miembros y el de menores), así que su aporte individual debe leerse en conjunto.")
        detalle = explicacion.rename(columns={"grupo": "Grupo", "nombre": "Característica",
                                              "valor": "Valor del hogar", "aporte": "Aporte"})
        detalle["Aporte"] = detalle["Aporte"].map(lambda valor: numero(valor, 3, signo=True))
        st.dataframe(detalle[["Grupo", "Característica", "Valor del hogar", "Aporte"]], hide_index=True)


def pestana_hogar(modelo, metadatos, referencia):
    st.caption("Registra las características observables del hogar. El resultado se actualiza al instante.")
    col1, col2 = st.columns(2)
    col1.button("Cargar un hogar real al azar (ENAHO 2025)", on_click=cargar_hogar_enaho, width="stretch")
    col2.button("Volver al hogar típico", on_click=cargar_hogar_tipico, width="stretch")

    izquierda, derecha = st.columns([3, 2], gap="large")
    with izquierda:
        ficha_del_hogar()
    with derecha:
        with st.container(border=True):
            explicacion = resultado_del_hogar(modelo, metadatos, referencia)
    # El detalle va debajo de la ficha, donde hay más ancho para la tabla
    if explicacion is not None:
        with izquierda:
            detalle_de_aportes(explicacion)


# ---------------------------------------------------------------------------
# Pestaña 2: evaluar una lista de hogares
# ---------------------------------------------------------------------------

def plantilla_csv():
    """Archivo de ejemplo con las 36 variables predictoras (el hogar típico)."""
    plantilla = preparar_hogar(HOGAR_TIPICO)
    plantilla.insert(0, "id_hogar", "hogar_001")
    return plantilla.to_csv(index=False).encode("utf-8-sig")


def pestana_lista(modelo, metadatos):
    umbral = metadatos["umbral_decision"]
    st.markdown(
        "Un programa social suele partir de una **lista de hogares candidatos**. El prototipo calcula la "
        "probabilidad de pobreza de cada uno y los **ordena de mayor a menor**, para verificar primero a los de "
        "mayor riesgo.")
    origen = st.radio("Datos a evaluar", ["muestra", "archivo"], horizontal=True, key="origen_lista",
                      format_func=lambda x: ("Muestra real: 300 hogares de prueba de la ENAHO 2025" if x == "muestra"
                                             else "Subir un archivo CSV"))

    if origen == "muestra":
        tabla = hogares_de_prueba().sample(300, random_state=SEMILLA)
        tabla = tabla[CLAVES_HOGAR + VARIABLES_PREDICTORAS + ["pobre"]].reset_index(drop=True)
    else:
        st.caption("Una fila por hogar, con las 36 variables predictoras y los códigos del INEI "
                   "(ver docs/diccionario_datos.csv). Las columnas adicionales se conservan como identificación.")
        st.download_button("Descargar plantilla CSV", plantilla_csv(), "plantilla_hogares.csv", "text/csv")
        archivo = st.file_uploader("Archivo CSV de hogares", type="csv")
        if archivo is None:
            return
        try:
            tabla = pd.read_csv(archivo, sep=None, engine="python")
        except Exception as error:
            st.error(f"No se pudo leer el archivo: {error}")
            return

    try:
        hogares = validar_hogares(tabla)
    except ValueError as error:
        st.error(str(error))
        return
    resultado = predecir(modelo, hogares, umbral)

    identificacion = [c for c in tabla.columns if c not in VARIABLES_PREDICTORAS + ["pobre"]]
    salida = tabla[identificacion].copy()
    salida["Departamento"] = hogares["departamento"].map(ETIQUETAS["departamento"])
    salida["Área"] = hogares["area_rural"].map({0: "Urbana", 1: "Rural"})
    salida["Probabilidad de pobreza"] = resultado["probabilidad_pobreza"]
    salida["Priorizar"] = resultado["priorizar"].map({1: "Sí", 0: "No"})
    if "pobre" in tabla.columns:
        salida["Condición INEI"] = tabla["pobre"].map({1: "Pobre", 0: "No pobre"})
    salida = salida.sort_values("Probabilidad de pobreza", ascending=False)
    salida.insert(0, "Orden", range(1, len(salida) + 1))

    total, priorizados = len(salida), int(resultado["priorizar"].sum())
    col1, col2, col3 = st.columns(3)
    col1.metric("Hogares evaluados", total)
    col2.metric(f"Priorizados (probabilidad ≥ {porcentaje(umbral, 0)})",
                f"{priorizados} ({porcentaje(priorizados / total)})")
    if "pobre" in tabla.columns:
        pobres = tabla["pobre"] == 1
        detectados = int(((resultado["priorizar"] == 1) & pobres).sum())
        col3.metric("Hogares pobres detectados (INEI)", f"{detectados} de {int(pobres.sum())}")
        st.caption(f"De los {priorizados} hogares priorizados, {detectados} son pobres según el INEI. Los hogares "
                   "pobres no priorizados son errores de exclusión: deben cubrirse con otros mecanismos, como la "
                   "solicitud directa del hogar.")

    st.dataframe(salida, hide_index=True, height=420, column_config={
        "Probabilidad de pobreza": st.column_config.ProgressColumn(format="percent", min_value=0, max_value=1)})
    st.download_button("Descargar resultados (CSV)", salida.to_csv(index=False).encode("utf-8-sig"),
                       "hogares_priorizados.csv", "text/csv")


# ---------------------------------------------------------------------------
# Pestaña 3: acerca del modelo
# ---------------------------------------------------------------------------

def pestana_modelo(metadatos):
    st.subheader("¿Qué hace este prototipo?")
    st.markdown(
        "Estima la **probabilidad de que un hogar sea pobre** según la definición oficial del INEI (pobreza "
        "monetaria), usando **solo características observables**: vivienda, servicios, bienes, composición del "
        "hogar, educación y ubicación. Así apoya la **focalización de programas sociales** cuando no se conoce "
        "el gasto del hogar, que es costoso de medir.")

    st.subheader("Desempeño")
    # Tablas del cuaderno 05 con precisión completa, para que el redondeo coincida con el informe
    metricas_prueba = tabla_resultados("05_metricas_prueba.csv")
    regla = metricas_prueba.loc["E0 Regla 'rural = pobre'"]
    prueba = metricas_prueba.loc["E2 Regresión logística"]
    temporal = tabla_resultados("05_validacion_temporal.csv").loc["E2 Regresión logística"]
    filas = [("F1 (métrica principal)", "f1"),
             ("Recall: hogares pobres detectados", "recall"),
             ("Precision: priorizados que son pobres", "precision"),
             ("ROC-AUC: capacidad de ordenar por riesgo", "roc_auc"),
             ("Accuracy", "accuracy")]
    desempeno = pd.DataFrame({
        "Métrica": [nombre for nombre, _ in filas],
        "Modelo · prueba ENAHO 2025": [numero(prueba[clave], 3) for _, clave in filas],
        "Modelo · entrenado 2024, evaluado 2025": [numero(temporal[clave], 3) for _, clave in filas],
        "Regla «rural = pobre» · prueba": [numero(regla[clave], 3) if pd.notna(regla[clave]) else "—"
                                           for _, clave in filas],
    })
    st.dataframe(desempeno, hide_index=True)
    st.caption("Frente a la regla sin IA, el modelo detecta más hogares pobres y, a la vez, comete menos errores de "
               "inclusión. Entrenado con un año, mantiene su desempeño al año siguiente.")

    st.subheader("Equidad: ¿a quiénes deja fuera?")
    sesgo = tabla_resultados("05_analisis_sesgo.csv").reset_index()
    sesgo = sesgo.rename(columns={"index": "Dimensión", "grupo": "Grupo", "hogares_pobres": "Hogares pobres en la prueba",
                                  "recall_logistica": "Pobres detectados"})
    sesgo["Pobres detectados"] = sesgo["Pobres detectados"].map(porcentaje)
    # Alto suficiente para mostrar todas las filas sin barra de desplazamiento
    st.dataframe(sesgo[["Dimensión", "Grupo", "Hogares pobres en la prueba", "Pobres detectados"]], hide_index=True,
                 height=35 * (len(sesgo) + 1) + 3)
    st.caption("El modelo deja fuera a más hogares pobres en zonas urbanas, en hogares con jefa mujer y en la costa. "
               "En esos grupos se recomienda una verificación adicional.")

    st.subheader("Uso responsable y limitaciones")
    st.markdown(
        "- Es una herramienta de **apoyo a la decisión**: prioriza hogares para una verificación. **No reemplaza** "
        "la clasificación socioeconómica oficial ni decide por sí sola quién recibe un programa.\n"
        "- Aproximadamente **2 de cada 10 hogares pobres no son priorizados** (errores de exclusión) y más de la "
        "mitad de los priorizados no son pobres (errores de inclusión). Por eso la verificación es indispensable.\n"
        "- La probabilidad es un **puntaje de riesgo**: por el balanceo de clases, el modelo entrega valores más "
        "altos que la frecuencia real de pobreza (en promedio 0,37 frente a 0,18). Sirve para ordenar hogares y "
        "compararlos con el umbral, no como la proporción exacta de hogares pobres.\n"
        "- La ubicación influye en la predicción porque el INEI usa líneas de pobreza distintas según el costo de "
        "vida de cada zona. El mismo hogar puede tener distinta probabilidad en Lima que en otra región.\n"
        "- El modelo aprende de la ENAHO; si las condiciones del país cambian, debe **reentrenarse** con datos "
        "recientes.\n"
        "- La ficha no guarda ni envía datos de los hogares: todo se calcula en el momento.")

    st.subheader("Ficha técnica")
    st.markdown(
        f"- **Modelo:** {metadatos['modelo']}, `C={metadatos['hiperparametros']['C']}`, "
        f"`class_weight=\"{metadatos['hiperparametros']['class_weight']}\"`.\n"
        f"- **Umbral de decisión:** {numero(metadatos['umbral_decision'], 1)}.\n"
        f"- **Datos de entrenamiento:** {metadatos['datos_entrenamiento']}.\n"
        f"- **Variables predictoras:** {len(metadatos['variables_predictoras'])} características observables.\n"
        f"- **scikit-learn:** {metadatos['version_scikit_learn']} · **fecha del modelo:** {metadatos['fecha']}.\n"
        "- **Autor:** Nina Anchapuri Brens Fabrizio Leandro. Curso de Inteligencia Artificial, Ingeniería "
        "Informática, Universidad Nacional Federico Villarreal.")


# ---------------------------------------------------------------------------
# Página
# ---------------------------------------------------------------------------

def main():
    st.set_page_config(page_title="Focalización de hogares pobres", layout="wide")
    modelo, metadatos, referencia = cargar_recursos()

    # Al abrir la app, la ficha empieza con el hogar típico
    if "ficha_iniciada" not in st.session_state:
        cargar_en_ficha(HOGAR_TIPICO)
        st.session_state["ficha_iniciada"] = True

    st.title("Focalización de hogares en situación de pobreza")
    st.caption("Prototipo de apoyo a la decisión · Regresión logística entrenada con la ENAHO 2025 (INEI)")

    hogar, lista, acerca = st.tabs(["Evaluar un hogar", "Evaluar una lista de hogares", "Acerca del modelo"])
    with hogar:
        pestana_hogar(modelo, metadatos, referencia)
    with lista:
        pestana_lista(modelo, metadatos)
    with acerca:
        pestana_modelo(metadatos)


main()
