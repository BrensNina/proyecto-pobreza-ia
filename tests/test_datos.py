"""Pruebas unitarias de la lectura de archivos de la ENAHO (src/data/cargar.py)."""

import os

from src.data.cargar import leer_csv_enaho, ruta_modulo


def test_lee_formato_2025_punto_y_coma_y_coma_decimal(tmp_path):
    ruta = tmp_path / "enaho_2025.csv"
    ruta.write_text("conglome;ubigeo;factor07\n005001;010101;123,45\n", encoding="latin-1")

    datos = leer_csv_enaho(ruta)

    assert list(datos.columns) == ["CONGLOME", "UBIGEO", "FACTOR07"]
    assert datos.loc[0, "FACTOR07"] == 123.45
    # El ubigeo se conserva como texto para no perder el cero inicial del departamento
    assert datos.loc[0, "UBIGEO"] == "010101"


def test_lee_formato_2024_coma_y_punto_decimal(tmp_path):
    ruta = tmp_path / "enaho_2024.csv"
    ruta.write_text("CONGLOME,UBIGEO,FACTOR07\n005001,150101,98.5\n", encoding="latin-1")

    datos = leer_csv_enaho(ruta)

    assert datos.loc[0, "FACTOR07"] == 98.5
    assert datos.loc[0, "UBIGEO"] == "150101"


def test_filtra_columnas_sin_importar_mayusculas(tmp_path):
    ruta = tmp_path / "modulo.csv"
    ruta.write_text("Conglome,P101,P102,GASHOG2D\n1,1,3,999.9\n", encoding="latin-1")

    datos = leer_csv_enaho(ruta, columnas=["CONGLOME", "p101"])

    assert list(datos.columns) == ["CONGLOME", "P101"]


def test_lee_caracteres_del_espanol(tmp_path):
    ruta = tmp_path / "etiquetas.csv"
    ruta.write_text("CONGLOME,NOMBRE\n1,Áncash\n", encoding="latin-1")

    assert leer_csv_enaho(ruta).loc[0, "NOMBRE"] == "Áncash"


def test_ruta_de_un_modulo():
    ruta = ruta_modulo(2025, "sumaria", "data/raw")
    assert ruta == os.path.join("data/raw", "enaho_2025", "1031-Modulo34", "Sumaria-2025.csv")
