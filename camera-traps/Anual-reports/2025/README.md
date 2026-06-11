# Informe Anual 2025 — Fotomonitoreo Bosque Pehuén

Carpeta autocontenida con los datos, código, figuras y narrativa del informe anual del Programa de Fotomonitoreo bajo la nueva metodología CONAF.

## Estructura

```
2025/
├── informe_anual_2025.md      ← fuente del informe (Markdown editable)
├── render.sh                  ← convierte a DOCX (y opcionalmente PDF)
├── README.md                  ← este archivo
├── py/
│   ├── 00_prepare_basemap.py           ← (one-shot) extrae shapefiles de los ZIPs y genera los GeoJSONs de contexto geográfico (caminos, esteros, contorno y zona sobre 1000 m)
│   ├── 01_data_prep.py                 ← limpia y corrige los datos crudos
│   ├── 02_figures_tables.py            ← genera todas las figuras + tabla resumen (stdout)
│   ├── 03_compute_proximity.py         ← (diagnóstico) reporta distancias CT→camino/agua y candidatos de umbral
│   ├── list_ciervo_guina_images.py     ← lista cada imagen tagged ciervo/güiña con su ruta (revisión manual)
│   └── apply_verdicts.py               ← aplica los veredictos de revisión visual y regenera records/events
├── data/
│   ├── records_clean.parquet                          ← una fila por imagen (canónico, post-correcciones)
│   ├── events_clean.parquet                           ← una fila por episodio de 30 min (canónico)
│   ├── records_baseline.parquet                       ← baseline pre-veredicto (escrito por 01_data_prep.py, leído por apply_verdicts.py)
│   ├── events_baseline.parquet                        ← baseline pre-veredicto (escrito por 01_data_prep.py, leído por apply_verdicts.py)
│   ├── manual_review_ciervo_guina.csv / .md           ← listado de imágenes etiquetadas ciervo/güiña con rutas
│   ├── manual_review_verdicts_2026-06-02.csv          ← veredictos imagen-por-imagen (Felipe, 2026-06-02)
│   ├── corrections_report.md                          ← deltas pre-vs-post revisión (regenerado por apply_verdicts.py)
│   └── prep_log.txt                                   ← auditoría completa del prep automático
├── figures/                              ← figuras canónicas (post-correcciones)
│   ├── 01_top_species.png
│   ├── 02_native_introduced.png
│   ├── 03_richness_total.png
│   ├── 04_richness_nativas.png
│   ├── 05_richness_introducidas.png
│   ├── 06_panel_por_especie.png
│   └── 07_zonas_por_especie.png
└── figures_pre_correction/               ← snapshot de figuras antes de la revisión visual (referencia)
    └── …
```

## Cómo editar y compartir el informe

El archivo `informe_anual_2025.md` es la **fuente única de la narrativa**: se edita en cualquier editor de texto (VS Code, Obsidian, vim, etc.) y se renderiza al formato que necesite la audiencia.

### Para revisión en Word (caso jefe)

```bash
# Primera vez en una máquina nueva:
sudo apt install pandoc

# Cada vez que se quiera generar un Word:
bash render.sh
```

Genera `informe_anual_2025.docx` con las figuras embebidas. Listo para abrir en Word, hacer "Tracked changes" y devolver. Los cambios después se vuelcan manualmente al `.md`.

### Para PDF

```bash
sudo apt install texlive-xetex   # primera vez
bash render.sh pdf
```

## Cómo regenerar los datos y figuras

Si llega más data o se ajusta el código:

```bash
# Activar un entorno con pandas, openpyxl, pyarrow, yaml
conda activate data-pipeline
python py/01_data_prep.py        # escribe records_baseline.parquet + events_baseline.parquet
python py/apply_verdicts.py      # aplica los veredictos manuales → records_clean.parquet + events_clean.parquet canónicos

# Cambiar al entorno con matplotlib
conda activate fire_risk_dashboard
python py/02_figures_tables.py   # actualiza figures/*.png + imprime la Tabla 3 (zonas) en stdout
```

> **Capas de contexto (one-shot).** El script `py/00_prepare_basemap.py` se corre **una sola vez** para generar `plataforma-territorial/data/basemap/*.geojson` a partir de los ZIPs de origen (`Anual-reports/Curvas de nivel_BP-*.zip`, `Anual-reports/Figura 5_Sistema hídrico SN BP-*.zip` y `Anual-reports/Red senderos y Caminos-*.zip`). Requiere `shapely`, `pyproj` y `pyshp` en el entorno. No es necesario volver a correrlo a menos que cambien los shapefiles de origen.

> **Nota.** Tras la primera publicación se incorporó una revisión visual de las detecciones de Ciervo rojo y Güiña (sec. 1.6 del informe). El flujo canónico ahora es: `01_data_prep` produce el snapshot pre-revisión, `apply_verdicts` aplica los veredictos del archivo `data/manual_review_verdicts_2026-06-02.csv` para escribir los parquets canónicos, y `02_figures_tables` los lee. Si en el futuro se etiqueta correctamente al nivel de la revisión humana (etapa 3 del pipeline) y se confirma que los veredictos del archivo ya están reflejados en los CSV de campaña, el paso `apply_verdicts` puede eliminarse del flujo.

## Fuentes de datos canónicas

- **Imágenes etiquetadas** — `camera-traps/data/campaigns/{otono_2025, primavera_2025, pv_2025_2026}/new_labeled_data_reviewed.csv`
- **Registro de instalación de CTs** — `camera-traps/Anual-reports/Registro de monitoreo CT.xlsx`, hoja *Registro de instalacion*
- **Ubicaciones de CTs** — `plataforma-territorial/data/camera_trap_stations.geojson` (26 puntos)
- **Polígono de Bosque Pehuén** — `plataforma-territorial/data/boundary.geojson` (versión canónica vigente desde 2026-05-12)
- **Catálogo de especies** — `data-pipeline/species.yaml`
- **Capas de contexto geográfico** — `plataforma-territorial/data/basemap/` (generadas por `py/00_prepare_basemap.py` a partir de los ZIPs entregados en `Anual-reports/Curvas de nivel_BP-*.zip`, `Anual-reports/Figura 5_Sistema hídrico SN BP-*.zip` y `Anual-reports/Red senderos y Caminos-*.zip`)
- **Documentos de metodología** —
  - `../REVISIÓN DISEÑO METODOLÓGICO DE CONAF.pdf`
  - `../Resultados de evaluación Megadetector.docx.pdf`

## Reglas aplicadas en el pipeline

Documentadas con detalle en el docstring de `py/01_data_prep.py`. Resumen:

1. **Corte temporal CONAF:** descartar registros con timestamp corregido < 2024-10-01.
2. **Filtro de especies:** se descartan todas las aves, invertebrados y micromamíferos (Monito del monte, Ratón cola larga, Rata negra). Quedan solo mamíferos medianos y grandes.
3. **Episodio independiente:** 30 minutos por (cámara, especie).
4. **Correcciones de fecha** aplicadas a CT15 / CT16 / CT19 (Otoño 2025); 71 registros pre-redespliegue de TC16 (Primavera + PV) excluidos por imposibilidad de reconstrucción confiable.
5. **Despliegues no mapeables** (`100EK113`, 252 registros) descartados.
