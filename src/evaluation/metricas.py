"""
Cálculo de las métricas de evaluación del proyecto.

La métrica principal es F1 para la clase "pobre", porque en focalización importan
los dos tipos de error:
- recall bajo: hogares pobres que quedan fuera de los programas (error de exclusión);
- precision baja: hogares no pobres que reciben ayuda (error de inclusión o filtración).
"""

import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import cross_val_score

from src.config import FOLDS_VALIDACION_CRUZADA

METRICAS_CV = ["f1", "precision", "recall", "roc_auc", "accuracy"]


def calcular_metricas(y_real, y_predicho, y_probabilidad=None):
    """Devuelve un diccionario con las métricas de clasificación para la clase pobre (1)."""
    metricas = {
        "accuracy": accuracy_score(y_real, y_predicho),
        "precision": precision_score(y_real, y_predicho, zero_division=0),
        "recall": recall_score(y_real, y_predicho),
        "f1": f1_score(y_real, y_predicho),
    }
    # El ROC-AUC necesita probabilidades; la regla rural no las tiene
    if y_probabilidad is not None:
        metricas["roc_auc"] = roc_auc_score(y_real, y_probabilidad)
    return metricas


def evaluar_validacion_cruzada(pipeline, X, y, n_jobs=None):
    """
    Media y desviación estándar de cada métrica en la validación cruzada.

    Con cv=5, scikit-learn usa particiones estratificadas y siempre las mismas para
    los mismos datos, así que todos los modelos se evalúan sobre las mismas particiones.
    """
    resultados = {}
    for metrica in METRICAS_CV:
        puntajes = cross_val_score(pipeline, X, y, cv=FOLDS_VALIDACION_CRUZADA, scoring=metrica, n_jobs=n_jobs)
        resultados[f"{metrica}_media"] = puntajes.mean()
        resultados[f"{metrica}_desv"] = puntajes.std()
    return pd.Series(resultados)
