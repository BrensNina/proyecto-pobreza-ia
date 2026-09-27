# Datos del proyecto

## Origen

Los datos provienen de la **Encuesta Nacional de Hogares (ENAHO) 2025**, metodología actualizada, del **Instituto Nacional de Estadística e Informática (INEI)**.

- Portal oficial: https://proyectos.inei.gob.pe/microdatos/
- Código interno de la ENAHO 2025 anual en el portal: `1031` (la ENAHO 2024 es `966`)
- Fecha de descarga: 24 de septiembre de 2026

Los microdatos son de acceso libre y anónimos: no contienen nombres ni direcciones. Están protegidos por el secreto estadístico (D. Leg. 604 y D. S. 043-2001-PCM). Su uso está permitido citando la fuente.

## Cómo obtener los datos

Los archivos originales **no se versionan** en el repositorio por su tamaño (más de 300 MB descomprimidos). Se descargan automáticamente desde el INEI:

```bash
python src/data/descargar.py
```

El cuaderno `notebooks/01_data_understanding.ipynb` también descarga los dos años si los archivos no existen, verifica sus cifras oficiales y construye ambos datasets procesados. Los cuadernos 02 y 03 descargan, si faltan, los pocos módulos originales que usan (sumaria, vivienda y programas sociales).

## Estructura

```
data/
├── raw/
│   ├── enaho_2025/
│       ├── 1031-Modulo01/   Vivienda y hogar (Enaho01-2025-100.csv)
│       ├── 1031-Modulo02/   Miembros del hogar (Enaho01-2025-200.csv)
│       ├── 1031-Modulo03/   Educación (Enaho01A-2025-300.csv)
│       ├── 1031-Modulo18/   Equipamiento del hogar (Enaho01-2025-612.csv)
│       ├── 1031-Modulo34/   Sumaria, con la condición de pobreza (Sumaria-2025.csv)
│       └── 1031-Modulo37/   Programas sociales (Enaho01-2025-700B.csv)
│   └── enaho_2024/              Mismos módulos (01, 02, 03, 18 y 34) de la ENAHO 2024 (código 966)
└── processed/
    ├── hogares_enaho_2025.csv   Dataset principal a nivel de hogar (una fila por hogar)
    ├── hogares_enaho_2024.csv   Dataset de otro año, para la validación temporal
    ├── particion_2025.csv       Qué hogar está en entrenamiento y cuál en prueba (80/20, estratificada, semilla 42)
    └── panel_2025.csv           Marca de los hogares de 2025 también entrevistados en 2024 (se excluyen de la validación temporal)
```

Para descargar también la ENAHO 2024:

```bash
python -c "from src.data.descargar import descargar_enaho; descargar_enaho(2024, modulos=['01','02','03','18','34'])"
```

Cada carpeta de módulo incluye el diccionario de variables (`Diccionario_2025.pdf`) y la ficha técnica oficial del INEI.

## Formato de los CSV

El INEI cambió el formato entre años. La función `src/data/cargar.py::leer_csv_enaho` lo detecta automáticamente.

| Año | Separador | Decimal | Codificación |
|---|---|---|---|
| 2024 | `,` | `.` | latin-1 |
| 2025 | `;` | `,` | latin-1 |

## Dataset procesado

`processed/hogares_enaho_2025.csv` contiene **33 702 hogares** y **43 columnas**: 36 variables predictoras, la variable objetivo `pobre` y 6 columnas de identificación y análisis. `processed/hogares_enaho_2024.csv` tiene la misma estructura, con **33 691 hogares**. El diccionario completo está en `docs/diccionario_datos.csv`.

Ambos años reproducen exactamente las cifras oficiales de pobreza del INEI:

| Año | Nacional | Urbana | Rural |
|---|---|---|---|
| 2024 | 27,6 % | 24,8 % | 39,3 % |
| 2025 | 25,7 % | 23,4 % | 35,5 % |
