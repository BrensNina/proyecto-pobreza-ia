"""
Interpretación de la regresión logística: nombres legibles y explicación por hogar.

En la regresión logística, el puntaje de un hogar (log-odds de ser pobre) es el
intercepto más la suma de coeficiente × valor de cada columna transformada. Por eso
el aporte de cada característica puede calcularse de forma exacta.

Para que la explicación se lea "frente a algo", se compara al hogar con el hogar
promedio del entrenamiento:

    aporte = coeficiente × (valor del hogar − valor promedio del entrenamiento)

- Aporte positivo: esa característica hace que el hogar sea MÁS probablemente pobre
  que el hogar promedio.
- Aporte negativo: lo hace MENOS probablemente pobre.

La suma de todos los aportes es exactamente la diferencia de puntaje entre el hogar
y el hogar promedio.

Varias características están muy relacionadas entre sí (por ejemplo, el número de
miembros, el de menores de 14 y la tasa de dependencia). Su efecto se reparte entre
ellas, así que el signo de una sola puede confundir. Por eso la explicación principal
suma los aportes por grupos temáticos, lo que sigue siendo exacto.
"""

import numpy as np
import pandas as pd

from src.features.construir import ETIQUETAS, VARIABLES_BINARIAS, VARIABLES_CATEGORICAS

# Nombre corto de cada variable, para mostrarlo en el prototipo
NOMBRES_VARIABLES = {
    "n_miembros": "Miembros del hogar",
    "n_habitaciones": "Habitaciones",
    "personas_por_habitacion": "Personas por habitación",
    "jefe_edad": "Edad del jefe(a) del hogar",
    "n_menores_14": "Menores de 14 años",
    "n_mayores_65": "Mayores de 65 años",
    "tasa_dependencia": "Tasa de dependencia",
    "educacion_jefe": "Educación del jefe(a)",
    "max_educacion_adultos": "Mayor educación entre los adultos",
    "n_equipos": "Artefactos y vehículos",
    "tipo_vivienda": "Tipo de vivienda",
    "material_paredes": "Paredes",
    "material_piso": "Piso",
    "material_techo": "Techo",
    "tenencia_vivienda": "Tenencia de la vivienda",
    "fuente_agua": "Agua",
    "desague": "Servicio higiénico",
    "combustible_cocina": "Combustible para cocinar",
    "dominio": "Dominio geográfico",
    "departamento": "Departamento",
    "area_rural": "Área rural",
    "jefe_mujer": "Jefa del hogar mujer",
    "tiene_electricidad": "Electricidad",
    "tiene_telefono_fijo": "Teléfono fijo",
    "tiene_celular": "Celular",
    "tiene_tv_cable": "TV por cable",
    "tiene_internet": "Internet",
    "tiene_radio": "Radio",
    "tiene_tv_color": "TV a color",
    "tiene_computadora": "Computadora",
    "tiene_cocina_gas": "Cocina a gas",
    "tiene_refrigeradora": "Refrigeradora",
    "tiene_lavadora": "Lavadora",
    "tiene_microondas": "Microondas",
    "tiene_auto": "Auto o camioneta",
    "tiene_motocicleta": "Motocicleta",
}

# Grupos temáticos de variables para la explicación de cada hogar
GRUPOS_EXPLICACION = {
    "Tamaño y composición del hogar": ["n_miembros", "n_menores_14", "n_mayores_65", "tasa_dependencia"],
    "Hacinamiento": ["personas_por_habitacion", "n_habitaciones"],
    "Jefe(a) del hogar": ["jefe_edad", "jefe_mujer"],
    "Educación": ["educacion_jefe", "max_educacion_adultos"],
    "Artefactos y vehículos": ["n_equipos", "tiene_radio", "tiene_tv_color", "tiene_computadora", "tiene_cocina_gas",
                               "tiene_refrigeradora", "tiene_lavadora", "tiene_microondas", "tiene_auto",
                               "tiene_motocicleta"],
    "Comunicaciones": ["tiene_telefono_fijo", "tiene_celular", "tiene_tv_cable", "tiene_internet"],
    "Vivienda: tipo, materiales y tenencia": ["tipo_vivienda", "material_paredes", "material_piso", "material_techo",
                                              "tenencia_vivienda"],
    "Servicios básicos": ["fuente_agua", "desague", "tiene_electricidad", "combustible_cocina"],
    "Ubicación": ["dominio", "departamento", "area_rural"],
}
GRUPO_DE_VARIABLE = {variable: grupo for grupo, variables in GRUPOS_EXPLICACION.items() for variable in variables}

# Las dos variables de educación usan la misma escala ordinal del INEI
VARIABLES_EDUCACION = ["educacion_jefe", "max_educacion_adultos"]


def variable_original(nombre_tecnico):
    """Convierte 'categoricas__material_piso_6.0' en 'material_piso'."""
    grupo, variable = nombre_tecnico.split("__", 1)
    if grupo == "categoricas":
        for original in VARIABLES_CATEGORICAS:
            if variable.startswith(original + "_"):
                return original
    return variable


def nombre_legible(nombre_tecnico):
    """Convierte 'categoricas__material_piso_6.0' en 'material_piso: Tierra'."""
    grupo, variable = nombre_tecnico.split("__", 1)
    if grupo == "categoricas":
        for original in VARIABLES_CATEGORICAS:
            if variable.startswith(original + "_"):
                codigo = int(float(variable[len(original) + 1:]))
                return f"{original}: {ETIQUETAS[original].get(codigo, codigo)}"
    return variable


def describir_valor(variable, valor):
    """Valor de una variable del hogar en palabras (p. ej., material_piso 6 -> 'Tierra')."""
    if pd.isna(valor):
        return "Sin dato"
    if variable in VARIABLES_CATEGORICAS:
        return ETIQUETAS[variable].get(int(valor), str(valor))
    if variable in VARIABLES_EDUCACION:
        return ETIQUETAS["educacion_jefe"].get(int(valor), str(valor))
    if variable in VARIABLES_BINARIAS:
        return "Sí" if valor == 1 else "No"
    if float(valor).is_integer():
        return str(int(valor))
    return f"{valor:.2f}"


def calcular_referencia(modelo, X_entrenamiento):
    """Valor promedio de cada columna transformada en el entrenamiento (el "hogar promedio")."""
    preprocesador = modelo.named_steps["preprocesamiento"]
    transformado = preprocesador.transform(X_entrenamiento)
    return pd.Series(transformado.mean(axis=0), index=preprocesador.get_feature_names_out())


def puntaje_referencia(modelo, referencia):
    """Puntaje (log-odds) del hogar promedio: intercepto + coeficientes × valores promedio."""
    logistica = modelo.named_steps["modelo"]
    return float(logistica.intercept_[0] + np.dot(logistica.coef_[0], referencia.values))


def explicar_hogar(modelo, hogar, referencia):
    """
    Aporte de cada variable original a la predicción de UN hogar, frente al hogar promedio.

    Las columnas One-Hot de una misma variable se suman, para que "material_piso"
    tenga un solo aporte. Devuelve una tabla ordenada de mayor a menor aporte.
    """
    preprocesador = modelo.named_steps["preprocesamiento"]
    logistica = modelo.named_steps["modelo"]
    nombres_tecnicos = preprocesador.get_feature_names_out()

    valores = preprocesador.transform(hogar)[0]
    aportes = pd.Series(logistica.coef_[0] * (valores - referencia[nombres_tecnicos].values), index=nombres_tecnicos)
    aportes = aportes.groupby([variable_original(n) for n in nombres_tecnicos]).sum()

    fila = hogar.iloc[0]
    tabla = pd.DataFrame({
        "grupo": [GRUPO_DE_VARIABLE[v] for v in aportes.index],
        "variable": aportes.index,
        "nombre": [NOMBRES_VARIABLES[v] for v in aportes.index],
        "valor": [describir_valor(v, fila[v]) for v in aportes.index],
        "aporte": aportes.values,
    })
    return tabla.sort_values("aporte", ascending=False).reset_index(drop=True)


def resumir_por_grupo(explicacion):
    """Suma los aportes de cada grupo temático (tabla de explicar_hogar), de mayor a menor."""
    return explicacion.groupby("grupo")["aporte"].sum().sort_values(ascending=False)
