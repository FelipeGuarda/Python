# source_code_CT_2025 — Informe Anual 2025 (Fotomonitoreo Bosque Pehuén)

Bundle autocontenido para reproducir el Informe Anual 2025 del Programa de
Fotomonitoreo de Bosque Pehuén (Fundación Mar Adentro).

**Para replicar el informe paso a paso, abre [`GUIA_REPLICACION.md`](GUIA_REPLICACION.md).**

## Resumen rápido

Si ya tienes Python 3.11+ y pandoc instalados:

```bash
# 1. Crear entorno e instalar dependencias
python -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\Activate.ps1
pip install -r requirements.txt

# 2. Regenerar parquets y figuras
python py/01_data_prep.py            # parquets pre-revisión
python py/apply_verdicts.py          # aplica veredictos manuales → canónico
python py/02_figures_tables.py       # 6 PNGs en figures/

# 3. Renderizar a Word
bash render.sh                       # Windows: .\render.ps1
```

## Estructura

```
source_code_CT_2025/
├── GUIA_REPLICACION.md       ← guía exhaustiva paso a paso
├── README.md                 ← este archivo
├── informe_anual_2025.md     ← narrativa fuente (editable)
├── requirements.txt          ← dependencias pip
├── render.sh / render.ps1    ← wrappers pandoc (Linux+Mac / Windows)
├── inputs/                   ← TODOS los archivos de entrada (CSVs, XLSX, GeoJSON, YAML)
├── py/                       ← 4 scripts del pipeline
├── data/                     ← parquets, veredictos, reportes
└── figures/                  ← 6 figuras canónicas
```

## Contacto

Felipe Guarda (`felipe.guarda@fundacionmaradentro.cl`) — Fundación Mar Adentro.
