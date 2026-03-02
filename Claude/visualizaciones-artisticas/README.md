# Visualizaciones Artísticas — FMA

**Owner:** Felipe Guarda — Fundación Mar Adentro
**Status:** Planned / Partially explored.
**Independence:** Standalone creative project. No dependency on the Plataforma Territorial or the data pipeline. Each visualization can be built and used independently.

---

## What This Project Does

A collection of generative / artistic data visualizations that transform FMA's field data (wildlife detections, acoustic recordings, weather, phenology) into visual forms meant for communication, exhibition, or public engagement — not analysis.

The goal is to make data *felt* rather than just understood. These are not dashboards.

Each visualization is its own self-contained piece. Some are animated, some interactive, some generative. All are meant to be beautiful.

---

## Confirmed Pieces

### 1. Retrato Diario (Daily Portrait)

**Concept:** A generative visual "portrait" of the territory for each day — a single image that encodes the key conditions of that day at Bosque Pehuén.

**Inputs:**
- Fire risk level (0–100) → affects color temperature / intensity
- Temperature and humidity → affect shape density / openness
- Wind speed and direction → affect directionality of elements
- Species detected that day → each species adds a specific visual motif
- Precipitation → affects texture

**Output:** A single abstract image (PNG or animated GIF) that looks different every day but always feels like "that place."

**Visual direction:** Organic, not geometric. Think growth patterns, mycelium networks, forest light. Not charts.

**Tech options:**
- `matplotlib` with custom bezier curves / polar scatter
- `p5py` (Processing in Python)
- `vsketch` (generative vector graphics)
- Pure SVG generation

---

### 2. Constelación de Especies (Species Constellation)

**Concept:** Each species detected at Bosque Pehuén over time is a star. Stars are placed in a circular sky based on:
- **Angular position**: time of day when the species is most active (midnight at top, noon at bottom)
- **Distance from center**: how rare the species is (rare = outer ring, common = inner)
- **Size**: number of detections in the selected period
- **Color**: ecological role (predator, prey, invasive, bird, etc.)
- **Connections**: lines between species that are detected at the same station within the same night (potential predator-prey encounters)

**Output:** An interactive SVG or HTML visualization. Hover a star → species name, total detections, activity pattern appears.

**Tech:** Plotly (for interactivity) or D3.js (if JS is acceptable), or a static `matplotlib` version for print.

---

### 3. (Other pieces — to be defined)

Additional ideas discussed or implied:
- **Río de Sonidos**: a flowing river visualization where bird songs from the acoustic station form the current — amplitude = river width, pitch = color
- **Año Térmico**: a circular calendar (like a clock face) showing temperature and fire risk throughout the year — one ring per year if multi-year data is available
- **Mapa Vivo**: an animated version of the Observatorio map that "breathes" — species appear and disappear at their stations across a day

---

## Design Principles

1. **Data-accurate but visually expressive**: The mapping from data to visual form should be explainable and consistent, but the end result should feel like art, not a report.
2. **Self-contained outputs**: Each piece outputs a file (PNG, SVG, HTML) that can be used independently — for social media, printed posters, exhibition screens, or embedded in reports.
3. **Spanish labels**: All text in visualizations is in Spanish (species names, place names, dates).
4. **No explanation required**: A viewer unfamiliar with fire risk indices should be able to look at a Retrato Diario and understand that "today felt intense" without reading a legend.

---

## Tech Stack

No single stack — each piece uses whatever tools fit best. Common options:

| Tool | Best for |
|---|---|
| `matplotlib` + custom styles | Static prints, poster-quality output |
| Plotly | Interactive HTML pieces (hover, zoom) |
| `vsketch` / SVG generation | Vector output for print/laser cutting |
| `p5py` / Processing | Generative animations |
| `manim` | Mathematical/geometric animations |
| `ffmpeg` | Assembling frame sequences into video |

---

## Data Sources

All data comes from the field, not from generated or synthetic sources:

| Visualization | Data needed |
|---|---|
| Retrato Diario | Daily weather + fire risk + species detections (from DuckDB or flat CSVs) |
| Constelación de Especies | Full camera trap dataset (from `Camaras trampa/` CSVs or DuckDB) |
| Río de Sonidos | Bird song audio files (WAV/MP3 from acoustic monitoring) |
| Año Térmico | Multi-year weather station data |

Note: While the Plataforma Territorial uses DuckDB as its data source, these artistic pieces can read directly from the existing CSV files — they don't strictly require the data pipeline to be running.

---

## File Structure (planned)

```
visualizaciones-artisticas/
├── README.md                          ← this file
│
├── retrato-diario/
│   ├── generate.py                    ← takes a date → outputs portrait PNG
│   ├── style.py                       ← visual style parameters + color palettes
│   └── outputs/                       ← generated images
│
├── constelacion-especies/
│   ├── generate.py                    ← takes date range → outputs HTML or SVG
│   └── outputs/
│
└── [other-piece-name]/
    ├── generate.py
    └── outputs/
```

Each piece is its own subfolder with its own `generate.py` entry point.

---

## Ideas & Future Pieces

- **Fenología visual**: A tree silhouette that changes with season — leaves appear based on precipitation, flowers based on temperature peaks, colors based on detected bird species
- **Mapa de calor nocturno**: An animation of a single night at all camera stations — species appear as glowing dots at their detection time and station location
- **Impresión de campo**: A generative "field notes" style layout — handwriting-like font, sketchy lines, data embedded as margin notes — for printed annual reports

---

## Key Design Decisions

1. **Separate from the platform**: These are *outputs*, not *interfaces*. They are generated files, not live dashboards. They should work without a running server.
2. **One `generate.py` per piece**: Each piece can be run independently with a single command: `python generate.py --date 2026-03-01 --output portrait.png`
3. **Data-first, aesthetics-second — but aesthetics matter**: The visual encoding is grounded in real data relationships, but the visual result is judged on whether it is beautiful and communicative, not just accurate.

---

## Context for AI Sessions

When starting a new Claude session on this project:

1. This is a **creative / generative art** project — the goal is aesthetic and communicative, not analytical
2. Each piece lives in its own subfolder with its own `generate.py` — no shared framework
3. Data inputs are real field data from FMA: camera trap CSVs, weather station data, bird audio recordings
4. The existing **Bird Song Visualizer** project (`Volumetric bird songs/`) is a reference for acoustic visualization approach and tech stack (librosa + Plotly)
5. When designing a new piece, start with: what is the *one thing* this data should make the viewer feel? Then work backwards to the visual encoding.
6. Output formats: PNG for print/social, HTML for web/interactive, SVG for vector/exhibition
