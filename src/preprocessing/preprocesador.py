"""
Pipeline de preprocesamiento de las variables predictoras.

Cada grupo de variables recibe el tratamiento que corresponde a su tipo:
- numéricas: imputación con la mediana y estandarización (media 0, desviación 1);
- binarias (0/1): imputación con el valor más frecuente; ya están en una escala común;
- categóricas (códigos del INEI): imputación con el valor más frecuente y One-Hot Encoding.

El preprocesador se integra en un Pipeline junto con el modelo. Así, al entrenar,
las medianas, modas, medias y categorías se aprenden SOLO con los datos de
entrenamiento, y en la validación cruzada se vuelven a aprender en cada partición.
Esto evita el data leakage.
"""

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.features.construir import VARIABLES_BINARIAS, VARIABLES_CATEGORICAS, VARIABLES_NUMERICAS


def crear_preprocesador():
    """Devuelve el ColumnTransformer (sin entrenar) que prepara las variables predictoras."""
    transformador_numerico = Pipeline(steps=[
        ("imputador", SimpleImputer(strategy="median")),
        ("escalador", StandardScaler()),
    ])

    transformador_binario = Pipeline(steps=[
        ("imputador", SimpleImputer(strategy="most_frequent")),
    ])

    transformador_categorico = Pipeline(steps=[
        ("imputador", SimpleImputer(strategy="most_frequent")),
        # handle_unknown="ignore": una categoría que no apareció en entrenamiento no genera error
        ("codificador", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])

    return ColumnTransformer(transformers=[
        ("numericas", transformador_numerico, VARIABLES_NUMERICAS),
        ("binarias", transformador_binario, VARIABLES_BINARIAS),
        ("categoricas", transformador_categorico, VARIABLES_CATEGORICAS),
    ])
