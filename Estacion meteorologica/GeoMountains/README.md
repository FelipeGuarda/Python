# GeoMountains — entregable estación meteorológica WS-01

Entregable para la contraparte técnica del proyecto GEO Mountains (LNAS / UACh),
sobre la estación meteorológica de Bosque Pehuén.

| Archivo | Qué es |
|---|---|
| `FICHA-TECNICA-WS01.md` | El documento para la contraparte: lo que se entrega, lo que no, y lo que falta |
| `FICHA-TECNICA-WS01.docx` | Lo mismo en Word, para circular. Exportado el 2026-09-10 |
| `data/weather_data_WS-01.csv` | El registro completo, un archivo, 265.038 filas |
| `build_registry.py` | Construye el CSV y el reporte desde las fuentes primarias |
| `registry_report.json` | Todas las cifras de la ficha, regenerables |

## Regenerar

```
python build_registry.py
```

Lee los siete volcados TOA5 de `../Linea de tiempo/` y la copia de base de datos
en `../../data-pipeline/data/recovery/weather_station/` (ubicación redefinible
con `WS01_PARQUET_DIR`). Requiere `pandas` y `pyarrow`; en Windows corre bajo
`miniforge3/envs/plataforma-territorial`.

Cuando se descargue `Table1` del datalogger en la próxima visita a terreno, el
tramo posterior al 2025-07-23 pasa a tener `RECORD` y el script deja de depender
de la copia de base de datos para esa parte. El contador nunca se perdió en el
instrumento: lo descartaba la ingesta, corregido en `data-pipeline` el 2026-09-10
(`src/cr800_columns.py`). El anillo guarda 495,8 días, así que la fila más antigua
de ese tramo se sobrescribe hacia **≈2026-12-01** si el logger siguió escribiendo.

## Para circular el documento

```
python build_registry.py                                                    # 1. regenera
pandoc FICHA-TECNICA-WS01.md -o FICHA-TECNICA-WS01.docx --toc --toc-depth=3  # 2. exporta
```

**Ese orden, no el inverso.** Pandoc no regenera los bloques `<!-- GENERADO:… -->`, así que
exportar antes de construir produce un `.docx` con cifras viejas y una fecha que miente.

Después de exportar, actualizar a mano la fila «Versión Word» de la tabla de cabecera de la ficha
con la fecha de la exportación. Es la única cifra del documento que no sale del script, y existe
para que la contraparte pueda saber de qué versión salió el archivo que recibió.

Última exportación: **2026-09-10**, pandoc 3.11.

## Contexto

El análisis del que sale este entregable está en `../ANALISIS-ESTACION-WS01-ES.md`
(y su versión en inglés). Ese documento es una auditoría interna; esta carpeta es
lo que sale hacia afuera. Cuatro afirmaciones de la auditoría quedaron
desactualizadas por lo medido aquí: no existe un hueco de 12 horas en 2023, el
contador `RECORD` no tiene faltantes, el desvío de reloj fue de +11:45 y no de
−2 h, y la copia de base de datos pierde 8 registros por año en las transiciones
de horario de verano — 4 en abril y 4 en septiembre, por mecanismos distintos:

- **Septiembre** — `tz_utils.py` localiza con `nonexistent='shift_forward'`, así
  que las estampas inexistentes 00:00–00:45 y la real de 01:00 caen sobre un mismo
  instante UTC y la clave primaria deja una de las cinco. Mecanismo del código
  actual, reproducido sobre 2019-09-08 (`RECORD 44863-44867 → 04:00Z`).
- **Abril** — las cuatro estampas ambiguas 23:00–23:45 están ausentes del parquet
  (hueco real de 2:15 en UTC). El `tz_utils.py` actual **no** las pierde: las
  ubica en 03:00–03:45 UTC, verificado. Lo más probable es que esas filas se
  ingirieran antes de que `tz_utils.py` existiera. Una re-ingesta las recupera.
