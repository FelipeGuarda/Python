# Guía de replicación — Informe Anual 2025 (Fotomonitoreo Bosque Pehuén)

Esta guía permite a una persona sin contexto previo del proyecto **regenerar el
informe completo** (parquets, figuras y documento Word/PDF) a partir
exclusivamente del contenido de esta carpeta `source_code_CT_2025/`.

> **Importante.** Esta carpeta es **autocontenida**. No depende de ningún
> repositorio externo. Todos los archivos de entrada (CSVs de campañas,
> registro de instalación, catálogo de especies, polígonos de Bosque Pehuén)
> están dentro de `inputs/`. Las versiones canónicas de los parquets y las
> figuras también están incluidas — el pipeline sólo necesita re-correrse si
> se quiere validar la reproducibilidad.

---

## 0. Qué obtienes al terminar

- `data/records_clean.parquet` — una fila por imagen (canónico)
- `data/events_clean.parquet` — una fila por episodio de 30 min (canónico)
- `data/corrections_report.md` — auditoría de las correcciones manuales
- `figures/01_top_species.png` … `06_panel_por_especie.png`
- `informe_anual_2025.docx` (y opcionalmente `.pdf`)

---

## 1. Prerrequisitos

Necesitas tres cosas instaladas en tu computador. Lee la sección que
corresponda a tu sistema operativo.

### 1.1 Python 3.11 (vía Miniforge — recomendado)

Miniforge es una distribución mínima de conda; te servirá tanto para crear el
entorno aislado para los scripts como para instalar pandoc.

- **Windows**: descarga el instalador desde
  <https://github.com/conda-forge/miniforge/releases/latest> (archivo
  `Miniforge3-Windows-x86_64.exe`), ejecútalo, acepta los valores por defecto.
  Luego abre **"Miniforge Prompt"** desde el menú de inicio para los pasos
  siguientes.

- **Linux (Debian/Ubuntu)**:
  ```bash
  wget https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh
  bash Miniforge3-Linux-x86_64.sh
  # responde "yes" cuando pregunte si quieres `conda init`
  # cierra y vuelve a abrir tu terminal
  ```

- **macOS** (Apple Silicon o Intel):
  ```bash
  brew install miniforge       # si tienes Homebrew, lo más simple
  conda init "$(basename "${SHELL}")"
  # cierra y vuelve a abrir tu terminal
  ```

Verifica que quedó instalado:
```bash
conda --version
python --version
```

> **Alternativa sin conda.** Si ya tienes Python 3.11+ instalado por otra vía
> (python.org, pyenv, etc.), puedes saltarte conda y usar `venv` puro — ver la
> sección 3 "Opción B".

### 1.2 Pandoc (para generar el Word / PDF)

- **Windows**:
  ```powershell
  winget install --id JohnMacFarlane.Pandoc -e
  ```
  o bien, usando Chocolatey:
  ```powershell
  choco install pandoc
  ```
- **Linux (Debian/Ubuntu)**:
  ```bash
  sudo apt update && sudo apt install pandoc
  ```
- **macOS**:
  ```bash
  brew install pandoc
  ```

Verifica:
```bash
pandoc --version
```

### 1.3 (Opcional) Motor LaTeX para generar PDF

Sólo necesario si quieres salida PDF además del DOCX.

- **Windows**: `winget install --id MiKTeX.MiKTeX -e`
- **Linux**: `sudo apt install texlive-xetex`
- **macOS**: `brew install --cask mactex-no-gui`

---

## 2. Descomprimir esta carpeta

1. Descarga `source_code_CT_2025.zip` desde el Drive de FMA.
2. Descomprime en cualquier ubicación de tu disco — por ejemplo:
   - Windows: `C:\Users\TU_USUARIO\Documents\source_code_CT_2025\`
   - Linux/macOS: `~/source_code_CT_2025/`

Abre una terminal y entra a la carpeta:

- **Windows (PowerShell o Miniforge Prompt):**
  ```powershell
  cd "C:\Users\TU_USUARIO\Documents\source_code_CT_2025"
  ```
- **Linux / macOS:**
  ```bash
  cd ~/source_code_CT_2025
  ```

Verifica que el contenido se ve así (con `dir` en Windows o `ls` en Linux/macOS):

```
GUIA_REPLICACION.md
README.md
requirements.txt
render.sh
render.ps1
informe_anual_2025.md
inputs/
py/
data/
figures/
```

---

## 3. Crear el entorno Python

Tienes dos opciones equivalentes — elige una.

### Opción A — `venv` con pip (la más simple, no requiere conda)

Si ya tienes Python 3.11+ en tu PATH:

- **Windows (PowerShell):**
  ```powershell
  python -m venv .venv
  .venv\Scripts\Activate.ps1
  pip install -r requirements.txt
  ```
  > Si PowerShell te dice "running scripts is disabled", abre una vez
  > PowerShell como administrador y ejecuta:
  > `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`.

- **Linux / macOS:**
  ```bash
  python -m venv .venv
  source .venv/bin/activate
  pip install -r requirements.txt
  ```

### Opción B — entorno conda dedicado

```bash
conda create -n informe_2025 python=3.11 -y
conda activate informe_2025
pip install -r requirements.txt
```

### Validar que el entorno quedó bien

Con el entorno **activado**, corre:

```bash
python -c "import pandas, numpy, pyarrow, openpyxl, yaml, matplotlib; print('OK')"
```

Debe imprimir `OK`. Si falla, vuelve al paso 3 y revisa que `pip install -r
requirements.txt` haya terminado sin errores.

---

## 4. Verificar que los archivos de entrada están presentes

Antes de correr el pipeline, asegúrate de que estos archivos están en su
lugar. Si falta alguno, la carpeta venía incompleta del Drive.

```
inputs/
├── species.yaml                                      (5 KB)
├── boundary.geojson                                  (16 KB)
├── camera_trap_stations.geojson                      (11 KB)
├── Registro de monitoreo CT.xlsx                     (89 KB)
└── campaigns/
    ├── otono_2025/new_labeled_data_reviewed.csv      (242 KB)
    ├── primavera_2025/new_labeled_data_reviewed.csv  (719 KB)
    └── pv_2025_2026/new_labeled_data_reviewed.csv    (240 KB)
```

```
data/
├── records_clean_pre_correction.parquet              ← snapshot, NO editar
├── events_clean_pre_correction.parquet               ← snapshot, NO editar
├── manual_review_verdicts_2026-06-02.csv             ← veredictos manuales
├── manual_review_ciervo_guina.csv / .md              ← listado de revisión
├── records_clean.parquet                             ← canónico (corregido)
├── events_clean.parquet                              ← canónico (corregido)
├── corrections_report.md                             ← auditoría
├── prep_log.txt                                      ← log histórico
└── correction_log.txt                                ← log histórico
```

Los archivos `*_pre_correction.parquet` y `manual_review_verdicts_2026-06-02.csv`
son **insumos críticos** para `apply_verdicts.py`. No los modifiques ni los borres.

---

## 5. Ejecutar el pipeline

Con el entorno activado, corre los **tres** scripts en este orden exacto.
Cada uno tarda menos de 10 segundos.

### 5.1 Paso 1 — Generar parquets pre-revisión

```bash
python py/01_data_prep.py
```

**Qué hace.** Lee los CSVs de las tres campañas, aplica las correcciones de
fecha y los filtros del informe, y escribe:

- `data/records_clean.parquet` (sobreescribe — se restaura en el paso 5.2)
- `data/events_clean.parquet` (sobreescribe — se restaura en el paso 5.2)

**Qué deberías ver en pantalla.** Un encabezado, auditoría de correcciones de
fecha por CT15/CT16/CT19, conteos por campaña y año, y al final:

```
Wrote → .../data/records_clean.parquet
Wrote → .../data/events_clean.parquet
```

> **Nota importante.** Este paso sobreescribe los parquets canónicos con las
> versiones **sin** las correcciones manuales de la revisión visual
> (sec. 1.6 del informe). Para regresar a los parquets canónicos, **debes
> correr el paso 5.2 a continuación**. Si te saltas el 5.2, las figuras
> resultantes no coincidirán con el informe.

### 5.2 Paso 2 — Aplicar veredictos manuales (Ciervo rojo / Güiña)

```bash
python py/apply_verdicts.py
```

**Qué hace.** Lee los snapshots `*_pre_correction.parquet` (que NO se
sobreescriben en el paso 5.1) y aplica los veredictos del archivo
`data/manual_review_verdicts_2026-06-02.csv`. Sobreescribe `records_clean.parquet`
y `events_clean.parquet` con los valores canónicos del informe.

**Qué deberías ver.** Conteos de veredictos aplicados (rewrites, drops) y
una tabla de deltas por especie. El conteo final de eventos debe ser **419**.

### 5.3 Paso 3 — Regenerar todas las figuras

```bash
python py/02_figures_tables.py
```

**Qué hace.** Lee `data/events_clean.parquet` y los GeoJSONs de `inputs/`, y
escribe seis PNG en `figures/`.

**Qué deberías ver.**

```
Loading data...
  events  : 419
  stations: 26 (camera_num 1..26)
  boundary: 164 vertices
  cameras with zero events: [1, 17, 22, 23]

Generating figures...
  ✓ 01_top_species.png
  ✓ 02_native_introduced.png
  ✓ 03_richness_total.png
  ✓ 04_richness_nativas.png
  ✓ 05_richness_introducidas.png
  ✓ 06_panel_por_especie.png
```

Abre cualquier PNG en `figures/` para verificar visualmente que se generaron
bien.

---

## 6. Renderizar el informe a DOCX / PDF

Con pandoc instalado:

- **Linux / macOS:**
  ```bash
  bash render.sh           # produce informe_anual_2025.docx
  bash render.sh pdf       # también produce informe_anual_2025.pdf
  ```

- **Windows (PowerShell, dentro de la carpeta):**
  ```powershell
  .\render.ps1             # produce informe_anual_2025.docx
  .\render.ps1 pdf         # también produce informe_anual_2025.pdf
  ```

- **Alternativa manual (si los scripts no funcionan).** Ejecuta pandoc
  directamente:
  ```bash
  pandoc informe_anual_2025.md --from markdown --to docx --output informe_anual_2025.docx --resource-path=.
  ```

Las figuras de `figures/` quedan embebidas automáticamente en el DOCX.

---

## 7. Glosario de archivos

### `inputs/`
| Archivo | Para qué sirve |
|---|---|
| `species.yaml` | Catálogo canónico de especies (latín ↔ español, taxonómicos, banderas nativa/invasora). |
| `boundary.geojson` | Polígono del predio Bosque Pehuén (para los mapas). |
| `camera_trap_stations.geojson` | Coordenadas de las 26 cámaras trampa. |
| `Registro de monitoreo CT.xlsx` | Hoja "Registro de instalacion" — fechas de instalación usadas para corregir cámaras con reloj en 2017. |
| `campaigns/*/new_labeled_data_reviewed.csv` | Salida del etapa de revisión humana del pipeline para cada campaña: una fila por imagen con `scientificName` ya verificado. |

### `data/`
| Archivo | Para qué sirve |
|---|---|
| `records_clean.parquet` | **Canónico.** Una fila por imagen, post-revisión manual. |
| `events_clean.parquet` | **Canónico.** Una fila por episodio de 30 min, post-revisión manual. Es el archivo que consume `02_figures_tables.py`. |
| `records_clean_pre_correction.parquet` | Snapshot inmutable previo a la revisión visual. Insumo de `apply_verdicts.py`. **No editar.** |
| `events_clean_pre_correction.parquet` | Snapshot inmutable previo a la revisión visual. Insumo de `apply_verdicts.py`. **No editar.** |
| `manual_review_verdicts_2026-06-02.csv` | Veredictos manuales imagen-por-imagen (Felipe, 2026-06-02). Insumo de `apply_verdicts.py`. |
| `manual_review_ciervo_guina.csv / .md` | Listado original de imágenes etiquetadas como Ciervo rojo o Güiña, generado por `list_ciervo_guina_images.py` antes de la revisión visual. |
| `corrections_report.md` | Reporte auto-generado por `apply_verdicts.py` con los deltas por especie y los rewrites/drops aplicados. |
| `prep_log.txt`, `correction_log.txt` | Bitácoras históricas de corridas previas (referencia). |

### `py/`
| Script | Qué hace | ¿Necesario para reproducir? |
|---|---|---|
| `01_data_prep.py` | Limpia los CSVs crudos → genera parquets pre-revisión. | Sí (paso 5.1) |
| `apply_verdicts.py` | Aplica los veredictos manuales sobre los parquets pre-revisión → genera los canónicos. | Sí (paso 5.2) |
| `02_figures_tables.py` | Genera las 6 figuras del informe. | Sí (paso 5.3) |
| `list_ciervo_guina_images.py` | **Histórico.** Helper usado para preparar la lista de imágenes a revisar visualmente. Su salida ya está en `data/manual_review_ciervo_guina.csv`. **No necesitas correrlo.** |

---

## 8. Solución de problemas

### "ModuleNotFoundError: No module named 'pandas'" (o similar)
El entorno no está activado, o `pip install -r requirements.txt` no terminó
bien. Activa el entorno (`source .venv/bin/activate` o `conda activate
informe_2025`) y reinstala.

### "FileNotFoundError: ... Registro de monitoreo CT.xlsx" (o cualquier input)
Verifica que descomprimiste correctamente la carpeta y que `inputs/` está
poblado (paso 4). Si descargaste de Drive y `inputs/` está vacío, vuelve a
descargar — pudo haberse cortado.

### "UnicodeEncodeError: 'charmap' codec can't encode..."
Los scripts ya forzan UTF-8 en stdout. Si igualmente aparece, exporta antes
de correr:
- Windows PowerShell: `$env:PYTHONIOENCODING = "utf-8"`
- Linux / macOS: `export PYTHONIOENCODING=utf-8`

### "pandoc: command not found"
Pandoc no está instalado o no está en el PATH. Repite el paso 1.2 y abre una
terminal nueva.

### `render.ps1` da error "running scripts is disabled"
Abre PowerShell una vez como administrador y ejecuta:
```powershell
Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
```

### Las figuras se generaron pero se ven distintas a las del Drive
Asegúrate de haber corrido los **tres** pasos del pipeline en orden:
`01_data_prep` → `apply_verdicts` → `02_figures_tables`. Si te saltaste
`apply_verdicts`, las figuras reflejarán los datos sin la revisión manual
visual y diferirán del informe.

### El DOCX se generó pero no se ven las figuras
Ejecuta `render.sh` / `render.ps1` desde **dentro** de la carpeta
`source_code_CT_2025` (no desde su carpeta padre). Las rutas en
`informe_anual_2025.md` son relativas.

### "Permission denied" al ejecutar `render.sh` en Linux/macOS
```bash
chmod +x render.sh
bash render.sh
```

### Quiero correr el script `list_ciervo_guina_images.py`
Es histórico y no es necesario para reproducir el informe. Si igual lo
corres, las columnas `thumbnail_path` aparecerán vacías (no se incluyeron
las miniaturas en este bundle por tamaño). La salida `manual_review_ciervo_guina.csv`
sobreescribirá la versión bundleada — son contenidos equivalentes.

---

## 9. Si quieres modificar el informe

`informe_anual_2025.md` es la fuente única de la narrativa. Edítalo con
cualquier editor de texto (VS Code, Obsidian, vim…) y luego corre `render.sh`
/ `render.ps1` para regenerar el DOCX. Las figuras se embeben
automáticamente desde `figures/` mediante rutas relativas.

Para cambiar las **figuras** (paletas, títulos, tamaños), edita
`py/02_figures_tables.py` y vuelve a correrlo (paso 5.3) — no necesitas
re-correr 5.1 ni 5.2 si los parquets canónicos ya están al día.
