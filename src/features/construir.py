"""
Construcción del dataset a nivel de hogar a partir de los módulos de la ENAHO.

Unidad de análisis: el hogar (llave CONGLOME + VIVIENDA + HOGAR).
Variable objetivo: pobre = 1 si el INEI clasifica al hogar como pobre extremo o
pobre no extremo (POBREZA 1 o 2), y 0 si es no pobre (POBREZA 3).

Solo se usan características OBSERVABLES del hogar: vivienda, servicios,
equipamiento, composición y educación. No se usan ingresos, gastos ni programas
sociales, porque son la base del cálculo de la pobreza o una consecuencia de ella
(ver la sección de data leakage del informe).
"""

import numpy as np
import pandas as pd

CLAVES_HOGAR = ["CONGLOME", "VIVIENDA", "HOGAR"]

# ---------------------------------------------------------------------------
# Variables de la vivienda (módulo 100): código ENAHO -> nombre en el proyecto
# ---------------------------------------------------------------------------
VARIABLES_VIVIENDA = {
    "P101": "tipo_vivienda",
    "P102": "material_paredes",
    "P103": "material_piso",
    "P103A": "material_techo",
    "P104": "n_habitaciones",
    "P105A": "tenencia_vivienda",
    "P110": "fuente_agua",
    "P111A": "desague",
    "P1121": "tiene_electricidad",
    "P113A": "combustible_cocina",
    "P1141": "tiene_telefono_fijo",
    "P1142": "tiene_celular",
    "P1143": "tiene_tv_cable",
    "P1144": "tiene_internet",
}

# Equipamiento del hogar (módulo 612): código de artículo -> nombre
ARTICULOS_EQUIPAMIENTO = {
    1: "tiene_radio",
    2: "tiene_tv_color",
    7: "tiene_computadora",
    10: "tiene_cocina_gas",
    12: "tiene_refrigeradora",
    13: "tiene_lavadora",
    14: "tiene_microondas",
    17: "tiene_auto",
    18: "tiene_motocicleta",
}

# ---------------------------------------------------------------------------
# Listas de variables predictoras según su tipo (las usa el preprocesamiento)
# ---------------------------------------------------------------------------
VARIABLES_NUMERICAS = [
    "n_miembros",
    "n_habitaciones",
    "personas_por_habitacion",
    "jefe_edad",
    "n_menores_14",
    "n_mayores_65",
    "tasa_dependencia",
    "educacion_jefe",
    "max_educacion_adultos",
    "n_equipos",
]

VARIABLES_CATEGORICAS = [
    "tipo_vivienda",
    "material_paredes",
    "material_piso",
    "material_techo",
    "tenencia_vivienda",
    "fuente_agua",
    "desague",
    "combustible_cocina",
    "dominio",
    "departamento",
]

VARIABLES_BINARIAS = [
    "area_rural",
    "jefe_mujer",
    "tiene_electricidad",
    "tiene_telefono_fijo",
    "tiene_celular",
    "tiene_tv_cable",
    "tiene_internet",
] + list(ARTICULOS_EQUIPAMIENTO.values())

VARIABLES_PREDICTORAS = VARIABLES_NUMERICAS + VARIABLES_CATEGORICAS + VARIABLES_BINARIAS

# Columnas que se conservan para análisis, pero que NUNCA entran al modelo
COLUMNAS_IDENTIFICACION = CLAVES_HOGAR + ["ubigeo", "factor07", "pobreza_inei"]
VARIABLE_OBJETIVO = "pobre"

# ---------------------------------------------------------------------------
# Etiquetas legibles de los códigos del INEI (para gráficos y el prototipo)
# ---------------------------------------------------------------------------
ETIQUETAS = {
    "tipo_vivienda": {1: "Casa independiente", 2: "Departamento en edificio", 3: "Vivienda en quinta",
                      4: "Casa de vecindad", 5: "Choza o cabaña", 6: "Vivienda improvisada",
                      7: "Local no destinado a vivienda", 8: "Otro"},
    "material_paredes": {1: "Ladrillo o bloque de cemento", 2: "Piedra o sillar con cal o cemento", 3: "Adobe",
                         4: "Tapia", 5: "Quincha", 6: "Piedra con barro", 7: "Madera",
                         8: "Triplay/calamina/estera", 9: "Otro"},
    "material_piso": {1: "Parquet o madera pulida", 2: "Láminas asfálticas o vinílicos", 3: "Losetas o terrazos",
                      4: "Madera", 5: "Cemento", 6: "Tierra", 7: "Otro"},
    "material_techo": {1: "Concreto armado", 2: "Madera", 3: "Tejas", 4: "Calamina o fibra de cemento",
                       5: "Caña o estera con torta de barro", 6: "Triplay/estera/carrizo",
                       7: "Paja u hojas de palmera", 8: "Otro"},
    "tenencia_vivienda": {1: "Alquilada", 2: "Propia, totalmente pagada", 3: "Propia, por invasión",
                          4: "Propia, comprándola a plazos", 5: "Cedida por el trabajo",
                          6: "Cedida por otro hogar o institución", 7: "Otra forma"},
    "fuente_agua": {1: "Red pública dentro de la vivienda", 2: "Red pública fuera de la vivienda",
                    3: "Pilón de uso público", 4: "Camión cisterna", 5: "Pozo", 6: "Manantial o puquio",
                    7: "Otra", 8: "Río, acequia o laguna"},
    "desague": {1: "Red pública dentro de la vivienda", 2: "Red pública fuera de la vivienda",
                3: "Letrina", 4: "Pozo séptico o biodigestor", 5: "Pozo ciego o negro",
                6: "Río, acequia o canal", 7: "Otra", 9: "Campo abierto"},
    "combustible_cocina": {1: "Electricidad", 2: "Gas (balón GLP)", 3: "Gas natural", 5: "Carbón",
                           6: "Leña", 7: "Otro", 8: "No cocinan"},
    "dominio": {1: "Costa Norte", 2: "Costa Centro", 3: "Costa Sur", 4: "Sierra Norte", 5: "Sierra Centro",
                6: "Sierra Sur", 7: "Selva", 8: "Lima Metropolitana"},
    "departamento": {1: "Amazonas", 2: "Áncash", 3: "Apurímac", 4: "Arequipa", 5: "Ayacucho",
                     6: "Cajamarca", 7: "Callao", 8: "Cusco", 9: "Huancavelica", 10: "Huánuco",
                     11: "Ica", 12: "Junín", 13: "La Libertad", 14: "Lambayeque", 15: "Lima",
                     16: "Loreto", 17: "Madre de Dios", 18: "Moquegua", 19: "Pasco", 20: "Piura",
                     21: "Puno", 22: "San Martín", 23: "Tacna", 24: "Tumbes", 25: "Ucayali"},
    "educacion_jefe": {1: "Sin nivel", 2: "Inicial", 3: "Primaria incompleta", 4: "Primaria completa",
                       5: "Secundaria incompleta", 6: "Secundaria completa",
                       7: "Superior no universitaria incompleta", 8: "Superior no universitaria completa",
                       9: "Superior universitaria incompleta", 10: "Superior universitaria completa",
                       11: "Maestría/Doctorado"},
}


def a_numero(serie):
    """Convierte una columna del INEI a número. Los espacios en blanco pasan a NaN."""
    return pd.to_numeric(serie.astype(str).str.strip().replace({"": np.nan, "nan": np.nan}), errors="coerce")


def variables_contexto(sumaria):
    """Tamaño del hogar, ubicación geográfica, objetivo y factor de expansión (módulo 34)."""
    df = sumaria[CLAVES_HOGAR].copy()
    df["n_miembros"] = sumaria["MIEPERHO"]
    df["dominio"] = sumaria["DOMINIO"]
    # Estratos 1 a 5 son urbanos y 6 a 8 rurales (convención del INEI)
    df["area_rural"] = (sumaria["ESTRATO"] >= 6).astype(int)
    df["ubigeo"] = sumaria["UBIGEO"].astype(str).str.zfill(6)
    df["departamento"] = df["ubigeo"].str[:2].astype(int)
    df["factor07"] = sumaria["FACTOR07"]
    df["pobreza_inei"] = sumaria["POBREZA"]
    df[VARIABLE_OBJETIVO] = (sumaria["POBREZA"] <= 2).astype(int)
    return df


def variables_vivienda(vivienda):
    """Materiales, servicios y comunicaciones de la vivienda (módulo 100)."""
    df = vivienda[CLAVES_HOGAR].copy()
    for codigo, nombre in VARIABLES_VIVIENDA.items():
        df[nombre] = a_numero(vivienda[codigo])
    # 99 es el código de valor faltante en el número de habitaciones
    df.loc[df["n_habitaciones"] >= 99, "n_habitaciones"] = np.nan
    return df


def variables_miembros(miembros):
    """Composición del hogar y datos del jefe(a) (módulo 200)."""
    m = miembros.copy()
    m["P203"] = a_numero(m["P203"])
    m["P204"] = a_numero(m["P204"])
    m["P207"] = a_numero(m["P207"])
    m["P208A"] = a_numero(m["P208A"])
    m.loc[m["P208A"] >= 99, "P208A"] = np.nan

    # Solo miembros del hogar, sin trabajadores del hogar (8) ni pensionistas (9)
    m = m[(m["P204"] == 1) & (~m["P203"].isin([8, 9]))]

    agrupado = m.groupby(CLAVES_HOGAR)
    df = pd.DataFrame({
        "n_menores_14": agrupado["P208A"].apply(lambda edades: (edades < 14).sum()),
        "n_mayores_65": agrupado["P208A"].apply(lambda edades: (edades >= 65).sum()),
        "n_edad_activa": agrupado["P208A"].apply(lambda edades: edades.between(14, 64).sum()),
    }).reset_index()

    jefes = m[m["P203"] == 1][CLAVES_HOGAR + ["P207", "P208A"]].drop_duplicates(CLAVES_HOGAR)
    jefes = jefes.rename(columns={"P208A": "jefe_edad"})
    jefes["jefe_mujer"] = (jefes["P207"] == 2).astype(int)

    return df.merge(jefes.drop(columns="P207"), on=CLAVES_HOGAR, how="left")


def variables_educacion(educacion, miembros):
    """Nivel educativo del jefe(a) y máximo nivel entre los adultos del hogar (módulo 300)."""
    e = educacion[CLAVES_HOGAR + ["CODPERSO", "P301A"]].copy()
    e["P301A"] = a_numero(e["P301A"])
    # 12 = Básica especial y 99 = faltante: no encajan en la escala ordinal, se tratan como faltantes
    e.loc[e["P301A"].isin([12, 99]), "P301A"] = np.nan

    m = miembros[CLAVES_HOGAR + ["CODPERSO", "P203", "P208A"]].copy()
    m["P203"] = a_numero(m["P203"])
    m["P208A"] = a_numero(m["P208A"])
    e = e.merge(m, on=CLAVES_HOGAR + ["CODPERSO"], how="left")

    jefe = e[e["P203"] == 1].groupby(CLAVES_HOGAR)["P301A"].first().rename("educacion_jefe")
    adultos = e[e["P208A"] >= 18].groupby(CLAVES_HOGAR)["P301A"].max().rename("max_educacion_adultos")
    return pd.concat([jefe, adultos], axis=1).reset_index()


def variables_equipamiento(equipamiento):
    """Tenencia de artefactos y vehículos del hogar (módulo 612)."""
    e = equipamiento[CLAVES_HOGAR + ["P612N", "P612"]].copy()
    e["P612N"] = a_numero(e["P612N"])
    e["P612"] = a_numero(e["P612"])
    e["tiene"] = np.where(e["P612"] == 1, 1, np.where(e["P612"] == 2, 0, np.nan))

    # Total de artículos que posee el hogar (de la lista completa del módulo)
    total = e.groupby(CLAVES_HOGAR)["tiene"].sum(min_count=1).rename("n_equipos")

    # Una columna 0/1 por cada artículo seleccionado
    seleccion = e[e["P612N"].isin(ARTICULOS_EQUIPAMIENTO.keys())]
    tabla = seleccion.pivot_table(index=CLAVES_HOGAR, columns="P612N", values="tiene", aggfunc="max")
    tabla = tabla.rename(columns=ARTICULOS_EQUIPAMIENTO)
    tabla.columns.name = None

    return pd.concat([tabla, total], axis=1).reset_index()


def agregar_variables_derivadas(df):
    """
    Variables derivadas (feature engineering): hacinamiento y tasa de dependencia.

    Se usa al construir el dataset y también en el prototipo, para que el modelo
    reciba las variables calculadas exactamente igual en el entrenamiento y en el uso.
    Requiere las columnas n_miembros, n_habitaciones, n_menores_14, n_mayores_65 y n_edad_activa.
    """
    df = df.copy()
    habitaciones = df["n_habitaciones"].where(df["n_habitaciones"] > 0)
    df["personas_por_habitacion"] = df["n_miembros"] / habitaciones
    dependientes = df["n_menores_14"] + df["n_mayores_65"]
    # Si nadie está en edad activa, la dependencia se iguala al número de dependientes
    df["tasa_dependencia"] = dependientes / df["n_edad_activa"].where(df["n_edad_activa"] > 0, 1)
    return df


def anular_codigos_invalidos(df):
    """Validez: un código categórico que no figura en el diccionario del INEI se trata como faltante."""
    df = df.copy()
    for variable in VARIABLES_CATEGORICAS:
        codigo_invalido = df[variable].notna() & ~df[variable].isin(ETIQUETAS[variable].keys())
        df.loc[codigo_invalido, variable] = np.nan
    return df


def construir_dataset_hogares(modulos):
    """Une todos los módulos y devuelve un DataFrame con una fila por hogar."""
    df = variables_contexto(modulos["sumaria"])
    df = df.merge(variables_vivienda(modulos["vivienda"]), on=CLAVES_HOGAR, how="left")
    df = df.merge(variables_miembros(modulos["miembros"]), on=CLAVES_HOGAR, how="left")
    df = df.merge(variables_educacion(modulos["educacion"], modulos["miembros"]), on=CLAVES_HOGAR, how="left")
    df = df.merge(variables_equipamiento(modulos["equipamiento"]), on=CLAVES_HOGAR, how="left")
    df = agregar_variables_derivadas(df)
    df = anular_codigos_invalidos(df)

    columnas = COLUMNAS_IDENTIFICACION + VARIABLES_PREDICTORAS + [VARIABLE_OBJETIVO]
    return df[columnas]


def diccionario_de_datos():
    """Tabla con el origen y la descripción de cada columna del dataset de hogares."""
    filas = []
    for variable in COLUMNAS_IDENTIFICACION + VARIABLES_PREDICTORAS + [VARIABLE_OBJETIVO]:
        origen, descripcion = DESCRIPCIONES[variable]
        if variable in VARIABLES_NUMERICAS:
            tipo = "Numérica"
        elif variable in VARIABLES_CATEGORICAS:
            tipo = "Categórica"
        elif variable in VARIABLES_BINARIAS or variable == VARIABLE_OBJETIVO:
            tipo = "Binaria (0/1)"
        else:
            tipo = "Identificación (no entra al modelo)"
        filas.append({"variable": variable, "tipo": tipo, "origen_enaho": origen, "descripcion": descripcion})
    return pd.DataFrame(filas)


# Origen en la ENAHO y descripción de cada columna (Anexo C del informe)
DESCRIPCIONES = {
    "CONGLOME": ("Todas", "Número de conglomerado (parte de la llave del hogar)"),
    "VIVIENDA": ("Todas", "Número de vivienda (parte de la llave del hogar)"),
    "HOGAR": ("Todas", "Número de hogar (parte de la llave del hogar)"),
    "ubigeo": ("Sumaria", "Código de ubicación geográfica del INEI (departamento, provincia, distrito)"),
    "factor07": ("Sumaria", "Factor de expansión del hogar; solo para estimaciones poblacionales"),
    "pobreza_inei": ("Sumaria POBREZA", "Condición de pobreza oficial: 1 pobre extremo, 2 pobre no extremo, 3 no pobre"),
    "n_miembros": ("Sumaria MIEPERHO", "Número de miembros del hogar"),
    "n_habitaciones": ("Mód. 100 P104", "Habitaciones de la vivienda, sin contar baño, cocina ni pasadizos"),
    "personas_por_habitacion": ("Derivada", "n_miembros / n_habitaciones (hacinamiento)"),
    "jefe_edad": ("Mód. 200 P208A", "Edad del jefe(a) del hogar en años"),
    "n_menores_14": ("Mód. 200 P208A", "Miembros menores de 14 años"),
    "n_mayores_65": ("Mód. 200 P208A", "Miembros de 65 años o más"),
    "tasa_dependencia": ("Derivada", "(menores de 14 + mayores de 65) / miembros de 14 a 64 años"),
    "educacion_jefe": ("Mód. 300 P301A", "Nivel educativo del jefe(a), escala ordinal de 1 (sin nivel) a 11 (posgrado)"),
    "max_educacion_adultos": ("Mód. 300 P301A", "Máximo nivel educativo entre los miembros de 18 años o más"),
    "n_equipos": ("Mód. 612 P612", "Número de artefactos y vehículos que posee el hogar"),
    "tipo_vivienda": ("Mód. 100 P101", "Tipo de vivienda"),
    "material_paredes": ("Mód. 100 P102", "Material predominante de las paredes exteriores"),
    "material_piso": ("Mód. 100 P103", "Material predominante de los pisos"),
    "material_techo": ("Mód. 100 P103A", "Material predominante de los techos"),
    "tenencia_vivienda": ("Mód. 100 P105A", "Régimen de tenencia de la vivienda"),
    "fuente_agua": ("Mód. 100 P110", "Procedencia principal del agua"),
    "desague": ("Mód. 100 P111A", "Conexión del servicio higiénico"),
    "combustible_cocina": ("Mód. 100 P113A", "Combustible usado con mayor frecuencia para cocinar"),
    "dominio": ("Sumaria DOMINIO", "Dominio geográfico de la ENAHO (costa, sierra, selva y Lima Metropolitana)"),
    "departamento": ("Sumaria UBIGEO", "Departamento (dos primeros dígitos del ubigeo)"),
    "area_rural": ("Sumaria ESTRATO", "1 si el hogar está en área rural (estratos 6 a 8)"),
    "jefe_mujer": ("Mód. 200 P207", "1 si el jefe del hogar es mujer"),
    "tiene_electricidad": ("Mód. 100 P1121", "1 si el hogar tiene alumbrado eléctrico"),
    "tiene_telefono_fijo": ("Mód. 100 P1141", "1 si el hogar tiene teléfono fijo"),
    "tiene_celular": ("Mód. 100 P1142", "1 si el hogar tiene teléfono celular"),
    "tiene_tv_cable": ("Mód. 100 P1143", "1 si el hogar tiene TV por cable o satelital"),
    "tiene_internet": ("Mód. 100 P1144", "1 si el hogar tiene conexión a internet (fija o móvil)"),
    "tiene_radio": ("Mód. 612", "1 si el hogar tiene radio"),
    "tiene_tv_color": ("Mód. 612", "1 si el hogar tiene TV a color"),
    "tiene_computadora": ("Mód. 612", "1 si el hogar tiene computadora o laptop"),
    "tiene_cocina_gas": ("Mód. 612", "1 si el hogar tiene cocina a gas"),
    "tiene_refrigeradora": ("Mód. 612", "1 si el hogar tiene refrigeradora o congeladora"),
    "tiene_lavadora": ("Mód. 612", "1 si el hogar tiene lavadora de ropa"),
    "tiene_microondas": ("Mód. 612", "1 si el hogar tiene horno microondas"),
    "tiene_auto": ("Mód. 612", "1 si el hogar tiene auto o camioneta"),
    "tiene_motocicleta": ("Mód. 612", "1 si el hogar tiene motocicleta"),
    "pobre": ("Sumaria POBREZA", "VARIABLE OBJETIVO: 1 si el hogar es pobre (POBREZA 1 o 2), 0 si no es pobre"),
}
