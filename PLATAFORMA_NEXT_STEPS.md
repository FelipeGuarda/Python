# Plataforma Territorial FMA — Estado y Próximos Pasos

Last updated: 2026-03-24

---

## Two-Machine Split (project-level ownership)

| Machine | Projects | Rationale |
|---|---|---|
| **Home (Linux)** | `data-pipeline/` (weather), `plataforma-territorial/` frontend + backend (weather, fire risk, Asistente), `literatura-agent/`, `visualizaciones-artisticas/` | CR800 systemd service runs here; all platform frontend lives on one machine |
| **Office (Windows)** | `camera-traps/`, `species-classifier/`, `plataforma-territorial/` **Phase 3 only** (camera trap ingestion, thumbnail pipeline, camera Dashboard tab) | Images and GPU are here; test camera integration against real data |

**Rule:** Platform repo is shared, but Phase 3 tasks = office, everything else = home. Always commit before switching machines.

**Handoff protocol:**
- Office → Home: commit reviewed CSV → pull at home → run ingestor
- Home → Office: commit platform code → pull before starting Phase 3 work

---

## Decisión pendiente: React vs Streamlit

El README de `plataforma-territorial/` describe la plataforma como una app Streamlit.
Lo que existe y funciona hoy es un prototipo React/Vite (`plataforma-demo/`).

**Antes de construir los módulos reales hay que decidir:** ¿el prototipo React se convierte en la versión definitiva, o se construye la versión Streamlit descrita en el README?

Implicaciones:
- **React:** más control visual, mejor para la experiencia del Observatorio (mapa interactivo, animaciones), requiere mantener frontend + backend separados.
- **Streamlit:** más rápido de conectar a Python/DuckDB, menos fricción para los módulos de datos, pero limitaciones visuales en el Observatorio.

---

## Prioridad 1 — Data Pipeline (`data-pipeline/`)

**Estado: No construido. Bloquea todo lo demás.**

Sin este pipeline, la plataforma siempre mostrará datos mock. Es la primera cosa que hay que construir.

### Versión mínima (desbloquea la plataforma)
- [ ] Crear esquema DuckDB (`fma_data.duckdb`) con tablas: `weather_station`, `weather_forecast`, `camera_trap`, `fire_risk`
- [ ] Ingestor de CSV de Timelapse2 (exportaciones manuales de cámara trampa)
- [ ] Fetch de Open-Meteo API → tabla `weather_forecast`
- [ ] Script de cálculo de índice FRI → tabla `fire_risk` (reutilizar lógica de `Estacion meteorologica/`)

### Versión completa (después de lo mínimo)
- [ ] Fetch remoto del datalogger CR800 vía Tailscale VPN → tabla `weather_station`
- [ ] Deduplicación y validación de esquema en ingestión
- [ ] Scheduling con cron o APScheduler
- [ ] Tabla `literatura` para recibir papers del literature-agent

---

## Prioridad 2 — Portar el Fire Risk Dashboard a la plataforma

**Estado: Dashboard standalone completo en `Estacion meteorologica/Fire risk dashboard/`**

El código ya existe y funciona. Solo hay que integrarlo.

- [ ] Portar `risk_calculator.py` y `fire_model.pkl` como módulo compartido
- [ ] Conectar el tab "Riesgo de Incendio" del Dashboard a datos reales de DuckDB (en lugar de mock)
- [ ] Incluir el índice ML (Random Forest) además del índice de reglas en la vista

---

## Prioridad 3 — Asistente con Claude API real

**Estado: UI existe en el prototipo, respuestas son mock.**

- [ ] Conectar tab Asistente a Claude API (Sonnet + tool use)
- [ ] Implementar herramientas de consulta contra DuckDB: riesgo actual, detecciones recientes, tendencias
- [ ] Cada respuesta con valor calculado debe citar su fórmula y datos de entrada (principio de transparencia metodológica)

Ver skill `claude-api` disponible en Claude Code para scaffolding.

---

## Prioridad 4 — Observatorio: mapa real

**Estado: El mapa es un SVG dibujado a mano con datos ficticios.**

- [ ] Definir coordenadas reales de las estaciones de cámara y estación meteorológica
- [ ] Implementar mapa interactivo real (pydeck o folium si Streamlit, mapbox/leaflet si React)
- [ ] Conectar marcadores a datos reales de DuckDB (última detección, temperatura actual, etc.)
- [ ] Capas opcionales: zonas de riesgo, perímetros de incendios históricos

---

## Prioridad 5 — Literature Agent (`literatura-agent/`)

**Estado: Arquitectura diseñada, ningún código escrito. Independiente de todo lo demás.**

Puede construirse en cualquier momento en paralelo con otras prioridades.

- [ ] Configurar topics y keywords relevantes para FMA
- [ ] Implementar fetchers: OpenAlex (primero, más completo), SciELO (crítico para literatura latinoamericana), arXiv, PubMed
- [ ] Deduplicación por DOI
- [ ] Summarización con Claude Haiku en español
- [ ] Envío de digest HTML semanal por email (Gmail API)
- [ ] Opcional: guardar papers en tabla `literatura` de DuckDB

---

## Proyecto nuevo — Dispositivos Acústicos

**Estado: Datos no recuperados aún. Sin código.**

FMA tiene dispositivos de monitoreo acústico desplegados en campo. Los archivos de audio
aún no han sido descargados de los dispositivos. Cuando estén disponibles, este proyecto
tiene tres fases:

### Fase 1 — Recuperación e ingesta de datos
- [ ] Descargar grabaciones de los dispositivos físicos
- [ ] Definir estructura de carpetas y convención de nombres (dispositivo, fecha, hora)
- [ ] Agregar ingestor al data-pipeline: watcher de carpeta de audio → tabla `acoustic` en DuckDB
      (metadatos: dispositivo, timestamp, duración, ruta de archivo — no el audio en sí)

### Fase 2 — Análisis de audio
- [ ] Identificación de especies por vocalización — opciones:
  - **BirdNET** (Cornell Lab, open source) — identificación de aves por canto, sin entrenamiento
  - **PAMGuide / OpenSoundscape** — análisis bioacústico general
  - Modelo personalizado si BirdNET no cubre las especies prioritarias de Bosque Pehuén
- [ ] Output: detecciones acústicas con especie, confianza, timestamp → tabla `acoustic_detections`
- [ ] Integrar detecciones acústicas al pipeline junto con las de cámaras trampa

### Fase 3 — Integración a la plataforma
- [ ] Nuevo tab "Acústica" en el Dashboard o ampliación del tab de Fauna
- [ ] Marcadores de dispositivos acústicos en el mapa del Observatorio
- [ ] Comparación cámara trampa vs acústica para las mismas especies

### Conexión con visualizaciones artísticas
El proyecto `visualizaciones-artisticas/` ya tiene diseñado el concepto "Río de Sonidos"
(visualización de cantos de aves) y un proyecto de referencia en `Volumetric bird songs/`.
Los archivos de audio de este proyecto alimentan directamente esas visualizaciones.

---

## En espera de más datos de campo

### Camera Traps — Fase 2 (`camera-traps/`)

**Estado: Fase 1 operativa. Fase 2 pausada intencionalmente.**

La Fase 1 (MegaDetector + CLIP + revisión humana) debe correr en cada nueva campaña.
La Fase 2 (clasificador EfficientNetV2 personalizado) solo tiene sentido cuando haya ≥50 imágenes revisadas por especie.

- [ ] Correr pipeline Fase 1 en cada nueva campaña de cámaras trampa
- [ ] Acumular CSVs revisados → dataset de entrenamiento
- [ ] Cuando haya volumen suficiente: implementar `phase2_classifier/` con EfficientNetV2-S vía `timm`

---

## Baja prioridad / Cuando haya datos reales

### Visualizaciones Artísticas (`visualizaciones-artisticas/`)

Outputs de comunicación y exhibición, no dashboards operativos. Requieren DuckDB con datos reales.

- [ ] **Retrato Diario:** portrait generativo diario del territorio (riesgo + clima + especies detectadas)
- [ ] **Constelación de Especies:** mapa estelar circular con posición según hora de actividad, distancia según rareza
- [ ] **Río de Sonidos:** visualización de cantos de aves (requiere archivos de audio)
- [ ] **Año Térmico:** calendario circular de temperatura y riesgo anual

---

## Proyectos completados (no requieren trabajo inmediato)

| Proyecto | Estado |
|---|---|
| `Estacion meteorologica/Fire risk dashboard/` | Completo — pendiente integración a plataforma |
| `schedule-agent/` | Completo y desplegado en Linux vía cron |

---

## Proyectos sin documentación

- `Aves en BP/` — Contiene notebooks de comparación de listados de aves y archivos Excel. Sin README. Parece ser datos de referencia taxonómica para la lista de especies de cámaras trampa. Documentar antes de usar en la plataforma.
