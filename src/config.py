"""
Parámetros globales del proyecto.

Centralizarlos aquí garantiza que todos los cuadernos y scripts usen la misma
semilla, la misma proporción de prueba y las mismas rutas (reproducibilidad).
"""

# Semilla para todo proceso aleatorio (división de datos, modelos, validación cruzada)
SEMILLA = 42

# Proporción del dataset reservada como conjunto de prueba
PROPORCION_PRUEBA = 0.20

# Número de particiones (folds) de la validación cruzada
FOLDS_VALIDACION_CRUZADA = 5

# Rutas de los datasets procesados
RUTA_HOGARES = {
    2024: "data/processed/hogares_enaho_2024.csv",
    2025: "data/processed/hogares_enaho_2025.csv",
}
RUTA_PARTICION = "data/processed/particion_2025.csv"
# Marca de los hogares de 2025 que también fueron entrevistados en 2024 (es_panel = 1)
RUTA_PANEL_2025 = "data/processed/panel_2025.csv"

# Carpetas de salida
CARPETA_MODELOS = "models"
CARPETA_RESULTADOS = "results"
