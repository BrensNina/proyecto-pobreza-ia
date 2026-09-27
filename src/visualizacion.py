"""
Estilo común para todos los gráficos del proyecto.

Criterios: un solo color para series únicas, colores fijos para "No pobre" y "Pobre"
(nunca cambian de un gráfico a otro), líneas de guía finas y sólidas, y texto en gris
oscuro en lugar del color de la serie.
"""

import os

import matplotlib.pyplot as plt
import seaborn as sns

# Paleta categórica validada para daltonismo (el orden importa)
AZUL = "#2a78d6"
NARANJA = "#eb6834"
VERDE_AGUA = "#1baf7a"
AMARILLO = "#eda100"

# Colores fijos por clase de la variable objetivo
COLOR_NO_POBRE = AZUL
COLOR_POBRE = NARANJA
COLORES_CLASE = {0: COLOR_NO_POBRE, 1: COLOR_POBRE}
NOMBRES_CLASE = {0: "No pobre", 1: "Pobre"}

# Tinta y elementos del gráfico
TEXTO = "#0b0b0b"
TEXTO_SECUNDARIO = "#52514e"
GRILLA = "#e1e0d9"
EJE = "#c3c2b7"
FONDO = "#fcfcfb"

CARPETA_FIGURAS = os.path.join("results", "figuras")


def aplicar_estilo():
    """Configura matplotlib y seaborn con el estilo del proyecto."""
    sns.set_theme(style="whitegrid")
    plt.rcParams.update({
        "figure.facecolor": FONDO,
        "axes.facecolor": FONDO,
        "axes.edgecolor": EJE,
        "axes.labelcolor": TEXTO_SECUNDARIO,
        "axes.titlecolor": TEXTO,
        "axes.titlesize": 13,
        "axes.titleweight": "bold",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "grid.color": GRILLA,
        "grid.linestyle": "-",
        "grid.linewidth": 0.8,
        "xtick.color": TEXTO_SECUNDARIO,
        "ytick.color": TEXTO_SECUNDARIO,
        "legend.frameon": False,
        "font.family": "sans-serif",
        "savefig.dpi": 200,
        "savefig.bbox": "tight",
    })


def guardar_figura(nombre):
    """Guarda la figura actual en results/figuras para usarla en el informe."""
    os.makedirs(CARPETA_FIGURAS, exist_ok=True)
    ruta = os.path.join(CARPETA_FIGURAS, f"{nombre}.png")
    plt.savefig(ruta)
    return ruta
