# Informe Anual 2025 — Fotomonitoreo Bosque Pehuén

Carpeta autocontenida con los datos, código, figuras y narrativa del informe anual del Programa de Fotomonitoreo bajo la nueva metodología CONAF.

## Estructura

```
2025/
├── informe_anual_2025.md      ← fuente del informe (Markdown editable)
├── render.sh                  ← convierte a DOCX (y opcionalmente PDF)
├── README.md                  ← este archivo
├── py/
│   ├── 01_data_prep.py        ← limpia y corrige los datos crudos
│   └── 02_figures_tables.py   ← genera todas las figuras
├── data/
│   ├── records_clean.parquet  ← una fila por imagen (post-filtros)
│   ├── events_clean.parquet   ← una fila por episodio de 30 min
│   └── prep_log.txt           ← auditoría completa del prep
└── figures/
    ├── 01_top_species.png
    ├── 02_native_introduced.png
    ├── 03_richness_total.png
    ├── 04_richness_nativas.png
    ├── 05_richness_introducidas.png
    └── 06_panel_por_especie.png
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
python py/01_data_prep.py        # actualiza records_clean.parquet + events_clean.parquet

# Cambiar al entorno con matplotlib
conda activate fire_risk_dashboard
python py/02_figures_tables.py   # actualiza figures/*.png
```

## Fuentes de datos canónicas

- **Imágenes etiquetadas** — `camera-traps/data/campaigns/{otono_2025, primavera_2025, pv_2025_2026}/new_labeled_data_reviewed.csv`
- **Registro de instalación de CTs** — `camera-traps/Anual-reports/Registro de monitoreo CT.xlsx`, hoja *Registro de instalacion*
- **Ubicaciones de CTs** — `plataforma-territorial/data/camera_trap_stations.geojson` (26 puntos)
- **Polígono de Bosque Pehuén** — `plataforma-territorial/data/boundary.geojson` (versión canónica vigente desde 2026-05-12)
- **Catálogo de especies** — `data-pipeline/species.yaml`
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
