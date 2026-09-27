"""
Definición de los modelos, sus grillas de hiperparámetros y el baseline sin IA.

Todos los modelos se construyen dentro de un Pipeline junto con el mismo
preprocesador. Así se comparan en igualdad de condiciones: mismas variables,
misma preparación y misma partición de datos; solo cambia el algoritmo.
"""

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier

from src.config import SEMILLA
from src.preprocessing.preprocesador import crear_preprocesador


def crear_pipeline(modelo):
    """Une el preprocesador y un modelo en un solo Pipeline."""
    return Pipeline(steps=[
        ("preprocesamiento", crear_preprocesador()),
        ("modelo", modelo),
    ])


def crear_modelos():
    """Modelos base (hiperparámetros iniciales) de cada algoritmo."""
    return {
        "Regresión logística": LogisticRegression(max_iter=3000, random_state=SEMILLA),
        "Árbol de decisión": DecisionTreeClassifier(random_state=SEMILLA),
        "KNN": KNeighborsClassifier(),
        "Random Forest": RandomForestClassifier(n_estimators=300, random_state=SEMILLA, n_jobs=-1),
    }


# Hiperparámetros que se exploran con GridSearchCV.
# El prefijo "modelo__" indica que el parámetro pertenece al paso "modelo" del Pipeline.
GRILLAS = {
    "Regresión logística": {
        "modelo__C": [0.01, 0.1, 1, 10],
        "modelo__class_weight": [None, "balanced"],
    },
    "Árbol de decisión": {
        "modelo__max_depth": [4, 6, 8, 10, 12],
        "modelo__min_samples_leaf": [1, 20, 50, 100],
        "modelo__class_weight": [None, "balanced"],
    },
    "KNN": {
        "modelo__n_neighbors": [3, 5, 15, 25, 51],
        "modelo__weights": ["uniform", "distance"],
    },
    "Random Forest": {
        "modelo__max_depth": [12, 20, None],
        "modelo__min_samples_leaf": [2, 5, 10],
        "modelo__class_weight": [None, "balanced"],
    },
}


def regla_rural(X):
    """Baseline sin IA: clasifica como pobre a todo hogar ubicado en área rural."""
    return np.asarray(X["area_rural"]).astype(int)
