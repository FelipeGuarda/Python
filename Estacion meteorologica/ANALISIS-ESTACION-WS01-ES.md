# WS-01 — Auditoría del registro y de lo que lo documenta

**Estación Meteorológica Bosque Pehuén · Campbell CR800 n.º de serie 42107**

| | |
|---|---|
| Fecha | 2026-09-09 |
| Motivo | Solicitud externa para estandarizar la estación e incorporarla a una base de datos meteorológica internacional |
| Alcance | Auditoría de solo lectura. No se modificó ningún archivo de ningún proyecto. |
| Autor | Claude Opus 5, sesión `2026-09-09-estacion-meteorologica-timeline-y-metadatos-wmo` |
| Versión en inglés | `ANALISIS-ESTACION-WS01.md` — mismo contenido |
| Documento complementario | Ficha de metadatos (58 elementos WIGOS) — artifact publicado, ver §9 |

> **Cómo leer esto.** Toda afirmación que sigue es trazable a un archivo de esta máquina o fue calculada durante la auditoría a partir de los registros crudos. Los cálculos están rotulados como tales. Donde estoy infiriendo, la frase lo dice. Donde no sé, lo dice también. Nada aquí está rellenado por plausibilidad.

---

## 1. Resumen ejecutivo

Los **datos** están en excelente estado. Los **metadatos** prácticamente no existen. Y hay tres hallazgos dentro de los datos que importan más para una postulación internacional que cualquier cosa en las carpetas de metadatos.

- Siete años y medio de registros de 15 minutos, del 2018-09-21 al 2026-04-13, con exactamente **un** hueco real en toda la serie (12 horas).
- El reloj del datalogger corre en **UTC−03:00 fijo, sin horario de verano**. El pipeline de ingesta asume hora civil chilena, que sí aplica DST, de modo que la copia en base de datos arrastra marcas de tiempo con una hora de atraso en UTC durante aproximadamente la mitad de cada año.
- Un **desvío de reloj de ~2 horas durante unos tres meses** en 2023, no documentado en ninguna parte.
- Cinco de dieciséis canales instrumentales están fuera de servicio, dos de ellos **desde el primer día**, y ninguna falla está marcada en los datos: se presentan como ceros y constantes imposibles, nunca como nulos.
- **Nunca se ha realizado una calibración.** Ningún sensor está identificado por marca, modelo ni número de serie. Nunca se registró la altura de ningún sensor. La carpeta creada para alojar el protocolo de la estación está vacía.
- La estación está **fuera de línea desde el 2026-04-13** — 149 días. Hay un plazo de recuperación de datos asociado a esto, alrededor del **2027-08-22**.

---

## 2. La línea de tiempo completa

### 2.1 Fuente autoritativa

No son los archivos `.dat`. La serie autoritativa es:

```
data-pipeline/data/recovery/weather_station/{2018..2026}.parquet
```

| | |
|---|---|
| Registros | 264.943 |
| Cobertura | 2018-09-21 16:45 (−03) → 2026-04-13 13:00 (−04) |
| Intervalo | 15 min, marca de tiempo al **fin del intervalo** |
| `station_id` | `bosque_pehuen` |
| Timestamps duplicados | 0 |

Los siete `.dat` de `Linea de tiempo/` llegan hasta el **2025-07-23 12:45**. Los últimos nueve meses existen únicamente en la copia del pipeline.

### 2.2 Una trampa en el conjunto de `.dat`

**Los siete archivos no son siete períodos.** Cada uno es un volcado completo del buffer circular de 47.600 registros del CR800 al momento de la descarga, por lo que se solapan fuertemente. Todos los archivos tienen exactamente 47.604 líneas (4 filas de cabecera + 47.600 registros).

| Archivo | Primer registro | Último registro | Rango del contador |
|---|---|---|---|
| `CR800Series_Table 21092018_30012020.dat` | 2018-09-21 16:45 | 2020-01-30 12:30 | 11138–58737 |
| `CR800Series_Table1_13092019_21012021.dat` | 2019-09-13 23:30 | 2021-01-21 19:15 | 45437–93036 |
| `CR800Series_Table 01122020_11042022.dat` | 2020-12-01 18:00 | 2022-04-11 13:45 | 88135–135734 |
| `CR800Series_Table 03082021_12122022.dat` | 2021-08-03 21:15 | 2022-12-12 17:00 | 111668–159267 |
| `CR800Series_Table 08112022_18032024.dat` | 2022-11-08 21:00 | 2024-03-18 16:45 | 156019–203618 |
| `CR800Series_Table 14012023_24052024.dat` | 2023-01-14 18:15 | 2024-05-24 14:00 | 162440–210039 |
| `CR800Series_Table 14032024_23072025.dat` | 2024-03-14 17:00 | 2025-07-23 12:45 | 203235–250834 |

333.200 filas crudas se reducen a **239.650 timestamps únicos**. Cualquiera que reciba «siete archivos que cubren 2018–2025» sin esta advertencia va a contar doble.

La copia en Synology `CR800Series BP_Table123-07-2025.dat` es idéntica en contenido a `CR800Series_Table 14032024_23072025.dat` — verificado comparando las filas de datos ordenadas; la única diferencia son los terminadores de línea CRLF versus LF.

### 2.3 Cobertura y huecos

Calculado sobre la serie deduplicada: **99,98 % de completitud** contra los 239.697 espacios de 15 minutos que implica el período.

| Período | Extensión | Qué pasó |
|---|---|---|
| ≈2018-05-28 → 2018-09-21 16:45 | 116 d | Registros 0–11.137 sobrescritos antes de la primera descarga. Fecha de inicio retro-extrapolada del contador, no documentada. |
| **2023-07-12 13:45 → 2023-07-13 01:45** | **12 h · 47 reg** | El único hueco verdadero de la serie. Causa no registrada. |
| 2023-07-13 → ≈2023-10-10 | ≈89 d · ≈8.500 reg | Reloj ~2 h atrasado (§4.2). Valores correctos, timestamps no confiables. |
| 2026-04-13 13:00 → hoy | 149 d, abierto | Falla de telemetría — antena. Posiblemente recuperable (§2.4). |
| Cada año, ≈abr → sep | sistemático | Timestamps 1 h atrasados en UTC, solo en la copia en base de datos (§4.1). |

Registros por año: 2018 — 9.725 · 2019 — 35.040 · 2020 — 35.136 · 2021 — 35.040 · 2022 — 35.040 · 2023 — 34.993 · 2024 — 35.136 · 2025 — 19.540 (hasta el 23 de julio).

Los otros ocho «huecos» de 1:15–2:15 h que aparecen en el parquet caen todos en la fecha de término del horario de verano de abril. Son artefactos de §4.1, no interrupciones.

### 2.4 El plazo de recuperación de datos

Table1 almacena **47.600 registros**. A 15 minutos eso son **714.000 minutos = 495,8 días** de memoria a bordo.

Si el datalogger sigue energizado y registrando, todo lo posterior a la pérdida de telemetría del 2026-04-13 sobrevive en el buffer hasta aproximadamente el **2027-08-22**, momento en que empieza a sobrescribirse. Una visita a terreno antes de esa fecha recupera toda la interrupción; después, los datos se perdieron.

Dos incógnitas, ambas relevantes: si el datalogger sigue funcionando, y si el tamaño de la tabla sigue siendo el mismo. Nadie ha ido a terreno desde la falla.

---

## 3. Estado de los sensores

### 3.1 Método

Fracciones de valores no nulos y de ceros por año, en las 40 columnas de los 239.650 registros crudos deduplicados, más verificaciones dirigidas de plausibilidad física (correlación entre canales, varianza, estructura diurna).

### 3.2 Veredicto por canal

| Canal | Variable | Veredicto |
|---|---|---|
| `AirTC_Max/Avg/Min` | Temperatura del aire | **Operativo, registro completo.** Medias anuales 7,0–8,2 °C; extremos −11,7 a +33,6 °C |
| `RH_Max/Avg/Min` | Humedad relativa | **Operativo.** Medias anuales 70–76 %. Anomalías en jun y ago 2021 según el informe de 2022, se normalizaron solas, causa nunca determinada |
| `WS_ms_Max/Avg/Min` | Velocidad del viento | **Operativo.** Medias anuales 1,58–1,89 m s⁻¹ |
| `WindDir_Max/Avg/Min/Std` | Dirección del viento | **Operativo.** Dominantes W, S, SE, SW; vientos del norte prácticamente ausentes |
| `Rain_mm_Tot` | Precipitación | **Operativo.** Totales anuales en §3.3 |
| `T107_10cm_*` | Temperatura de suelo 10 cm | **Operativo, registro completo** |
| `T107_50cm_*` | Temperatura de suelo 50 cm | **Operativo.** La «anomalía» may–nov 2021 que reportó el informe de 2022 fue un artefacto de configuración regional decimal de Excel en el procesamiento de ese informe; los valores crudos están correctos |
| `incomingSW_Avg` | Onda corta incidente | **Operativo.** Máximos anuales 1.086–1.308 W m⁻². Sin calibrar — tratar como valor relativo |
| `PtoRocio_Avg` | Punto de rocío | **Calculado en el datalogger, no medido.** Fórmula no documentada |
| `PTemp_C_Avg`, `BattV_Min` | Temp. del panel, batería | Housekeeping. El mínimo de batería nunca bajó de 12,5 V en siete años |
| `BP_mbar_Avg` | Presión de estación | **Nunca funcionó** (§3.4) |
| `incomingLW_Avg` | Onda larga incidente | **No es onda larga** (§3.5) |
| `outgoingLW_Avg` | Onda larga emitida | **Idénticamente cero en los 264.943 registros.** Nunca funcionó |
| `outgoingSW_Avg` | Onda corta reflejada | **Signo invertido.** Mediana −48 W m⁻², rango −299 a +53. Posiblemente recuperable |
| `albedo_Avg` | Albedo | Calculado en el datalogger a partir del par de onda corta; hereda sus problemas. 19 % de ceros en 2025, el primero el 2025-01-15 |
| `DT_*`, `Q_*`, `TCDT_*` | Distancia a superficie / altura de nieve | **Falló en 2021** (§3.6) |

### 3.3 Totales de precipitación

2019 — 2.126 · 2020 — 1.717 · 2021 — 1.579 · 2022 — 1.924 · 2023 — 2.861 · 2024 — 2.242 mm. Plausible para la Araucanía andina a esta altitud.

**Advertencia que hay que declarar:** se presume que el pluviómetro no es calefaccionado (inferencia a partir de la convención Campbell, no verificada), lo que implica subcaptura de nieve en invierno en un sitio que claramente recibe nieve. Es mejor declararlo que dejar que un análisis comparativo lo descubra.

### 3.4 El barómetro nunca funcionó

| Estadístico | Valor |
|---|---|
| Media | 636,68 mbar |
| **Desviación estándar** | **0,16 mbar** |
| Rango, siete años | 636,4257 – 638,1898 |

La presión real de una estación varía decenas de hPa en escala semanal. Una desviación estándar de 0,16 mbar a lo largo de siete años es físicamente imposible para una medición atmosférica. Aparte de eso, 636 hPa implicaría una altitud cercana a los 3.800 m — el sitio está cerca de los 1.220 m.

Este canal no es una medición de presión. Es una entrada desconectada, un valor por defecto sin escalar, o un voltaje de sesgo. El informe interno de 2022 nunca lo examinó.

*Consecuencia para §7: por esto no se pudo obtener la altitud simplemente invirtiendo la presión de estación.*

### 3.5 Los canales de onda larga

- `outgoingLW_Avg` es **idénticamente cero** en los 264.943 registros. El canal nunca funcionó.
- `incomingLW_Avg` correlaciona con `incomingSW_Avg` a **r = 0,981**, va de −346 a +1.199 W m⁻², y se vuelve negativo de noche. La onda larga incidente real ronda los 200–400 W m⁻² y no sigue a la onda corta. Este canal está mal conectado, o bien está reportando una salida de termopila sin corregir, sin el término de temperatura del cuerpo del instrumento. Inutilizable tal como está.

### 3.6 El sensor de nieve

El conteo mensual de valores no nulos de `DT_Avg` muestra una firma de falla limpia:

- 2020: ~2.900 lecturas no nulas todos los meses — plenamente operativo
- 2021-01: 2.962 · 2021-02: 2.414 · 2021-03: 2.023 · 2021-04: 598 · 2021-05: 83 · 2021-06: 52 · 2021-07: 349 · 2021-08: 18 · 2021-09: 3
- Desde 2021-10: cero, con solo destellos aislados

Operativo de 2018-09 a ~2021-03, degradándose durante mediados de 2021, efectivamente muerto desde agosto de 2021.

El informe de 2022 llegó a la misma conclusión y recomendó revisar el instrumento. **Eso fue hace cinco años y nunca se revisó.**

---

## 4. El reloj — el hallazgo más consecuente

### 4.1 El datalogger corre en UTC−03:00 fijo; el pipeline asume otra cosa

**Evidencia 1 — sin discontinuidades de DST.** A lo largo de siete años de timestamps crudos del datalogger no hay ni un solo timestamp duplicado en una transición de término del horario de verano, ni una sola hora faltante en una transición de inicio. Un reloj que siguiera la hora civil chilena no podría producir eso.

**Evidencia 2 — mediodía solar.** Tomando el mediodía solar como el punto medio entre el primer y el último cuarto de hora de cada día con `incomingSW_Avg > 20 W m⁻²`, y corrigiendo −0,125 h por el etiquetado al fin del intervalo, la mediana del registro completo es **13,88 h**.

En la longitud −71,733 la predicción es:

| Hipótesis de reloj | Mediodía solar predicho |
|---|---|
| UTC−03:00 | 13,78 h |
| UTC−04:00 | 12,78 h |

Observado: 13,88 h. Los valores mensuales oscilan entre 13,62 h (noviembre) y 14,25 h (febrero), consistente con la ecuación del tiempo. **El reloj está en UTC−03:00, fijo, todo el año, durante todo el registro.**

La corrección de 0,125 h es en sí misma una confirmación de la convención de timestamp: sin ella la estimación queda exactamente medio intervalo tarde, que es justo lo que produce el etiquetado al fin del intervalo.

**La consecuencia.** `data-pipeline/src/tz_utils.py::localize_santiago_to_utc` localiza los timestamps ingenuos a `America/Santiago`, que sí observa DST. Durante la ventana en que Chile está en UTC−04:00 (aproximadamente de abril a septiembre), el pipeline lee un `14:00` local como `18:00 UTC` cuando el instante real es `17:00 UTC`.

**Todos los registros de la copia en base de datos están una hora atrasados en UTC durante la mitad de cada año.** Los archivos `.dat` crudos no están afectados. Esto también fabrica los ocho huecos espurios de abril mencionados en §2.3.

### 4.2 El desvío de 2023

Mediana semanal del mediodía solar en torno al hueco de julio de 2023:

| Semana terminada | Mediodía solar (reloj del datalogger) |
|---|---|
| 2023-07-02 | 13,62 |
| 2023-07-09 | 14,88 |
| **2023-07-16** | **12,62** |
| 2023-07-23 | 11,75 |
| 2023-07-30 → 2023-10-08 | 11,69 – 11,75 |
| **2023-10-15** | **13,62** |
| desde 2023-10-22 | 13,62 – 14,12 |

Comenzando inmediatamente después del hueco de 12 horas del 2023-07-12/13 y persistiendo hasta alrededor del 2023-10-10, el reloj corrió **aproximadamente 2 horas atrasado**. Cerca de 8.500 registros tienen timestamps no confiables.

Esto no aparece en ninguna bitácora, informe ni nota de sesión. El informe de 2022 es anterior; nada posterior lo menciona. Se encontró en esta auditoría.

---

## 5. Metadatos — lo que sí existe

| Fuente | Contenido | Confiabilidad |
|---|---|---|
| **Cabeceras TOA5**, en cada `.dat` | CR800, serie **42107**, OS `CR800.Std.31.03`, programa `estacion_tres_hermanas.CR8`, firma `10101`, tabla `Table1`, más nombres, unidades y método de agregación por columna | **Alta.** Escrito por la máquina. Serie y firma constantes 2018–2025, de modo que el programa nunca se modificó. *Salvedad:* el campo de nombre de estación deriva — ver §6.1 |
| `plataforma-territorial/data/stations.yaml` | WS-01, lat −39,453642, lon −71,733092, modelo, endpoint Tailscale | **Alta.** Verificado en terreno, y el archivo documenta su propia corrección del 2026-04-24 (las coordenadas anteriores −39,4417/−71,7420 eran el centro del mapa, nunca la ubicación del datalogger). Sin altitud |
| `Informe Datos Estación Meteorológica.docx`<br>Synology · creado 2022-09-13 · campo creator «dell», autor no identificado | Revisión de operatividad de instrumentos, sep 2018 – abr 2022 | **Media en fechas de falla, baja en causas.** Encontró de forma independiente las mismas fallas de 2021 del radiómetro, albedómetro y nivómetro que encontró esta auditoría — corroboración real. Pero el autor escribe *«desconozco qué mide esta variable»* para PTemp, punto de rocío, DT y Q; atribuye un artefacto decimal de Excel a una posible falla de instrumento; nunca examina el barómetro; y en ninguna parte nombra un modelo de instrumento |
| `ProformaInvoice.pdf`<br>Campbell Scientific Centro Caribe · 166-2025-PA · 2025-03-31 | Panel solar 20 W + gabinete 16×18″, USD 1.563,50 | **Alta pero casi inútil.** Es una orden de reparación de 2025, no la compra original. No incluye ningún sensor |
| `piso_vegetacional.geojson` | Biotopo **Bosque Semidenso**, distrito **Ondulado**, código de especies `NP-AA`, unidad de 11,6 ha — extraído por punto-en-polígono en WS-01 | **Media.** La capa es confiable; la expansión de `NP-AA` a *Nothofagus pumilio – Araucaria araucana* es **inferencia mía** — confirmar contra la leyenda de la capa. El esquema de clasificación es propio de FMA, no un estándar |
| `boundary.geojson` | WS-01 confirmada dentro de la reserva. 868,87 ha, Área de Protección Privada, Fundación Mar Adentro | **Alta.** Contención calculada. *Notar que el polígono del límite está marcado como «en revisión final» en el vault* |
| `camera-traps/data/campaigns/estaciones.csv` | 27 estaciones con `elevation_m`, usado para el análisis de vecindad y relieve | **Media.** Los valores provienen de la columna `Altitud` de las notas de terreno — lecturas de GPS de mano (§7.3) |

### 5.1 Dos cosas que no hay que enviar

- **`Instrumentos Monitoreo Nasampulli CR2.kmz`** — un fluviómetro en el río Trafampulli y dos nodos (abierto y de dosel) en −39,016/−71,688 y −39,027/−71,674, 1.250–1.450 m. Esto es la **Reserva Nasampulli**, un sitio socio de GEO Mountains. No es nuestro.
- **`Fire risk dashboard/README.md` §2.1** — «Easting 263221, Northing 5630634 (≈ lat −39,61°, lon −71,71°)». El par UTM es correcto y convierte a −39,4413/−71,7514. Los grados decimales impresos al lado están **equivocados en unos 19 km en latitud**. Las coordenadas canónicas están en `stations.yaml`.

---

## 6. Metadatos — lo que no existe

Ausentes en toda la máquina:

- **Marcas, modelos y números de serie de los sensores** — de todos. No existe orden de compra original.
- **Alturas de los sensores sobre el suelo**, y las profundidades reales (no nominales) de las sondas de suelo. Sin la altura del anemómetro, el registro de viento no puede reducirse a la altura de referencia de 10 m que exige una comparación.
- **Cualquier registro de calibración, nunca.** Sin programa, sin resultados, sin certificados, sin trazabilidad a un patrón de referencia.
- **El programa CRBasic `estacion_tres_hermanas.CR8`.** Está nombrado en la cabecera de todos los archivos; el archivo no está en ninguna parte. Contiene la frecuencia de muestreo, los multiplicadores y offsets, el cableado, y las fórmulas de punto de rocío y albedo — es el único documento que explicaría tanto el barómetro muerto como los canales de onda larga mal conectados.
- **Fecha de instalación / puesta en marcha.** Lo mejor disponible: primer registro 2018-09-21, retro-extrapolación a ≈2018-05-28.
- **Bitácora de mantención o de visitas.** Sobrevive un solo fragmento: la proforma de 2025, sin registro de cuándo ni por qué se instaló.
- **Descripción de emplazamiento y exposición** — sin fotografías, sin levantamiento de obstáculos, sin croquis de horizonte. No se puede asignar clase de emplazamiento OMM para ninguna variable.
- **Cualquier identificador de registro.** Sin número DGA, DMC ni OMM en el repositorio ni en el vault.
- **Licencia de datos, DOI o declaración de cita.**
- **El protocolo de la estación.** `SynologyDrive/1. Estacion Meteorológica/Protocolo estación meteorológica/` es una **carpeta vacía**, creada en enero de 2025.

### 6.1 Un conflicto activo

La estación tiene **cuatro nombres** en el registro y ninguno es canónico:

- `CR800Series`, `CR800Series_2`, `CR800Series BP` — el campo de nombre de estación de distintos volcados TOA5
- `estacion_tres_hermanas` — el nombre del programa del datalogger, constante en todos

Esta es la deriva de etiquetas Bosque Pehuén / Tres Hermanas aflorando dentro del propio registro instrumental. Un nombre en un registro internacional es permanente y público; **hay que elegir uno antes de postular**.

### 6.2 Modelos de sensores — explícitamente supuestos

Inferidos de la convención de nombres de canales de Campbell. **Ninguno verificado para esta instalación. No enviar como hecho.**

| Canales | Sensor probable | Confianza |
|---|---|---|
| `T107_10cm`, `T107_50cm` | Termistores Campbell Model 107 | Casi certeza — el nombre del canal lo dice |
| `DT`, `Q`, `TCDT` | Sensor de distancia sónico SR50 / SR50A | Casi certeza — ese trío es la firma de ese sensor |
| `AirTC`, `RH` | Sonda clase HMP60 o HC2S3 | Moderada |
| `WS_ms`, `WindDir` | 03002 Wind Sentry, cazoletas y veleta | Moderada |
| `Rain_mm_Tot` | Pluviómetro de balancín familia TE525, sin calefacción | Moderada |
| `BP_mbar` | Barómetro clase CS106 | Moderada |
| Cuarteto de radiación + albedo | Radiómetro neto de cuatro componentes, probablemente CNR4 | Moderada |

---

## 7. Altitud

### 7.1 Valor actual

**1.223 m — provisional.** API de elevación de Open-Meteo, consultada el 2026-09-09.

Todavía no postulable: el DEM subyacente no está confirmado, y una altitud sin fuente nombrada es justamente la entrada que después queda marcada como observada.

### 7.2 Una coincidencia que no hay que leer como corroboración

1.223 m es exactamente la altitud de terreno registrada para CT02. CT02 está a 230 m de distancia y su cifra es una lectura de GPS de mano. Que dos métodos distintos en dos puntos distintos caigan en el mismo entero no dice nada — incluso podría indicar que la celda del DEM abarca ambos puntos.

### 7.3 Vecindad y relieve

| Estación | Altitud de terreno (m) | Distancia al mástil (m) |
|---|---|---|
| CT02 | 1223 | 229 |
| CT05 | 1270 | 244 |
| CT01 | 1263 | 556 |
| CT17 | 1062 | 618 |
| CT07 | 1232 | 723 |
| CT14 | 1048 | 858 |
| CT27 | 1408 | 985 |

Aproximadamente **346 m de desnivel dentro de 1 km** del mástil. Una celda de DEM de 30 m está promediando topografía real aquí, así que dos métodos cualesquiera van a discrepar más de lo que sugieren sus errores nominales. No enviar una cifra interpolada.

Estas altitudes de terreno provienen de la columna `Altitud` de las notas de terreno de cámaras trampa — GPS de mano. El error vertical de un GPS de mano suele ser 1,5–3× el horizontal: ±10–20 m bajo dosel, peor en pendiente. **No** son verdad de terreno.

### 7.4 Pendiente para la próxima sesión

1. **Nombrar el dataset.** Open-Meteo no reporta qué DEM entregó. Mi impresión es que es Copernicus DEM GLO-90; *no estoy seguro*, y la citabilidad era justamente la razón para preferir esta vía por sobre Google Earth. Confirmar en la documentación de Open-Meteo.
2. **Obtener GLO-30 de forma citable.** La instancia pública de Open-Topo-Data no lo tiene — `copernicus30` devuelve `Dataset not in config`. Tres vías:
   - `curl -s "https://api.opentopodata.org/datasets"` para ver qué sí aloja. Si aparece `srtm30m` sirve como contraste citable — NASA SRTM v3, 1 arcosegundo — aunque SRTM se levantó el año 2000 y se degrada en terreno escarpado, así que es una segunda opinión, no un reemplazo.
   - **API de OpenTopography** — gratuita, sirve GLO-30, requiere clave de cuenta.
   - **Descargar la tesela GLO-30** que cubre S40/W072 desde el Copernicus Data Space y leer el píxel localmente. Es una descarga única y deja un archivo que se puede archivar junto al registro de la estación — que es lo que hace auditable un valor de metadato dentro de cinco años. *Recomendada.*

   *Tengo confianza en que las dos últimas sirven GLO-30; no he verificado los detalles actuales de registro de cuenta ni de nomenclatura de teselas.*
3. **Correr la calibración.** Consultar las siete cámaras trampa de arriba por la misma API y comparar con sus altitudes de terreno. Los residuos entregan una incertidumbre empírica para esta ladera en vez de una barra de error nominal. CT17 y CT27 son las informativas — están en las partes escarpadas. Residuos dentro de ±15 m hacen defendible 1.223 m como valor provisional; un error de 50 m en CT27 significa que solo servirá una medición GNSS.
4. **Registrar el datum.** Un DEM entrega altura ortométrica; un receptor GNSS entrega altura elipsoidal por defecto. En Chile difieren en decenas de metros. El número que quede en el registro tiene que decir cuál es.

### 7.5 Lo que lo cerraría de forma permanente

Un **registro GNSS estático** — un receptor de grado topográfico sobre trípode en el punto, grabando observaciones crudas de fase portadora durante una o dos horas, y luego post-procesado contra una estación de referencia o mediante un servicio gratuito como CSRS-PPP de NRCan. Nivel centimétrico. Esto es distinto de promediar waypoints en un GPS de mano, que llega quizás a ±5 m.

O **RTK** — la misma física con correcciones llegando en vivo desde una base cercana. Requiere una base propia sobre un punto conocido, o una suscripción a red más conectividad; sin internet permanente en Bosque Pehuén, el RTK de red probablemente esté descartado.

FMA casi con certeza no tiene ninguno de los dos. La UACh y los socios del LNAS sí, y el grupo de Antonio Lara ya es la contraparte de GEO Mountains. Un receptor prestado sobre el mástil durante dos horas, en una visita que de todos modos hay que hacer. Bajo dosel, registrar por más tiempo. Convertir con un modelo de geoide (EGM2008) antes de reportar metros sobre el nivel del mar.

---

## 8. Qué quiere decir probablemente el solicitante con «metadatos»

No hay registro en el vault de esta solicitud específica — nada posterior a la reunión de coordinación de **GEO Mountains del 2026-06-16**. Esa reunión es el origen probable, y también explica por qué hay un KMZ de Nasampulli en esta carpeta.

Contexto relevante de `Meetings/2026-06-16-geo-mountains-coordinacion-inicial.md`: financiamiento suizo, 12–14 meses, tres sitios andinos (Bosque Pehuén, PN Villarrica, Reserva Nasampulli), alimentando la red CONDESAN y el Atlas de los Andes del Sur. Participantes: Patricio Contreras y Carla Marchant (LNAS), Antonio Lara (UACh), Felipe Ortega (UACh). Las variables climáticas se identificaron como los únicos datos hoy comunes a los tres sitios. Acuerdo de ~10M CLP, condicionado a un resumen metodológico y una demo de la plataforma.

«Base de datos del observatorio meteorológico mundial» apunta muy probablemente a uno de dos destinos, y **cuál de los dos cambia el esquema requerido**:

1. **OSCAR/Surface de la OMM** — el catálogo de metadatos de estaciones WIGOS. Requiere un WIGOS Station Identifier, emitido a través del Representante Permanente de Chile ante la OMM, es decir la **Dirección Meteorológica de Chile**. FMA no puede auto-registrarse. *Tengo confianza en que OSCAR/Surface es el sistema de metadatos de estaciones de la OMM y en que los WSI se tramitan por el servicio meteorológico nacional; no he verificado el procedimiento actual de la DMC.*
2. **Un inventario de observatorios de montaña** de GEO Mountains o CONDESAN — requisitos más laxos, sin control nacional.

En cualquier caso, «metadatos» significa la **descripción de la estación**, no las lecturas: identidad y operador, posición exacta **incluyendo altitud**, emplazamiento y exposición, marca/modelo/serie del instrumento por variable, altura o profundidad del sensor, intervalo de medición y de reporte, método de agregación, unidades, **la referencia horaria y su offset**, historial de calibración y mantención, interrupciones conocidas y banderas de calidad, y política de datos y licencia con un contacto.

Medido contra la propia especificación §3.5 de FMA en `Resources/estandares-datos-socios-plataforma-territorial.md`, podemos entregar `id`, `name`, `lat`, `lon`, `model` y `time_resolution`. No podemos entregar fecha de instalación, observaciones de calibración, modelos de sensores ni alturas de sensores.

---

## 9. Documento complementario

La lista completa de 58 elementos WIGOS — cada categoría, rellenada donde el registro lo permite, con los vacíos dejados en blanco — está publicada como artifact:

**https://claude.ai/code/artifact/f66a5dd4-5e57-412e-898e-9da6a0572998**

Conteo actual: **21** en el registro · **11** derivados en esta auditoría · **2** supuestos · **23** ausentes · **1** en conflicto.

> La estructura de diez categorías de ese documento es mi reconstrucción del WIGOS Metadata Standard (WMO-No. 1192) de memoria. **No** ha sido contrastada contra la publicación vigente ni contra el formulario en vivo de OSCAR/Surface. Tratar la estructura como una lista a verificar; los contenidos rellenados son la parte confiable.

Para renderizar este análisis y circularlo: `pandoc ANALISIS-ESTACION-WS01-ES.md -o ANALISIS-ESTACION-WS01-ES.docx`

---

## 10. Recomendaciones

### 10.1 Antes de responderle al solicitante

1. **Preguntar a qué registro va.** Si es OSCAR/WIGOS hay que pasar por la DMC; no conviene redactar nada hasta que eso esté zanjado.
2. **Iniciar la solicitud del WSI en paralelo** si el destino es OSCAR/Surface. Es el único ítem de la lista que FMA no controla.
3. **Elegir el nombre de la estación** (§6.1). Permanente y público una vez enviado.
4. **Decidir la política de datos y la licencia** (§6). Esto es una decisión, no una consulta. Notar que el dashboard de riesgo de incendios en esta misma carpeta lleva CC BY-NC 4.0 — eso cubre el dashboard, no los datos de la estación, y una cláusula no comercial puede ser incompatible con lo que espera el registro. *Verificar contra la política declarada del registro y no contra mi recuerdo de ella.*

### 10.2 Qué entregar

Entregar: temperatura del aire, humedad relativa, velocidad y dirección del viento, precipitación, temperatura de suelo a ambas profundidades, onda corta incidente.

Retener, declarando cada uno como fuera de servicio con fechas: presión de estación, ambos canales de onda larga, onda corta reflejada, albedo, altura de nieve.

Calificar: punto de rocío, como calculado y no observado.

**No dejar pasar el barómetro.** Un valor constante de 636 hPa en una base de datos internacional es peor que un campo ausente.

### 10.3 Una visita a terreno cierra casi todo

El plazo del buffer circular (≈2027-08-22) le pone fecha a esto.

- [ ] Descargar Table1 — recupera la interrupción desde el 2026-04-13, y resuelve si el datalogger siguió funcionando
- [ ] Recuperar `estacion_tres_hermanas.CR8` — llena cuatro elementos de metadatos ausentes de una vez
- [ ] Leer el reloj del datalogger contra una hora de referencia correcta; **anotar ambas lecturas crudas, no una conclusión**. Confirma o refuta §4.1, que hoy descansa enteramente en un cálculo. (La misma lección que aprendió el programa de cámaras trampa cuando terreno entregó un veredicto y se perdió la observación que había detrás.)
- [ ] Fotografiar y registrar cada sensor: marca, modelo, número de serie
- [ ] Medir alturas de sensores sobre el suelo y profundidades reales de las sondas de suelo
- [ ] Registro GNSS estático para la altitud (§7.5)
- [ ] Cuatro fotografías desde el mástil, una por punto cardinal, más un croquis de horizonte
- [ ] Anotar si el pluviómetro es calefaccionado — una mirada, define la salvedad de subcaptura de nieve

### 10.4 Correcciones de este lado

- [ ] Corregir la política de zona horaria del pipeline a offset fijo UTC−03:00 (§4.1), y rehacer la copia en base de datos
- [ ] Marcar 2023-07-13 → ≈2023-10-10 como período de reloj no confiable en el warehouse (§4.2)
- [ ] Agregar banderas de calidad — hoy todos los canales son 100 % no nulos, incluidos los muertos, así que un consumidor ingenuo lee las fallas como datos válidos
- [ ] Corregir o eliminar los grados decimales equivocados en `Fire risk dashboard/README.md` §2.1
- [ ] Escribir el protocolo de la estación en la carpeta vacía que se creó para eso

---

## Apéndice · Qué se revisó

**Repositorio:** los 7 volcados `.dat` en `Estacion meteorologica/Linea de tiempo/` · `merged_timeline.csv` y `unified timeline.py` · ambos notebooks · `data-pipeline/data/recovery/weather_station/*.parquet` (264.943 registros) · `data-pipeline/data/cr800_state.json` · `data-pipeline/src/fetchers/cr800.py` · `plataforma-territorial/data/stations.yaml`, `boundary.geojson`, `piso_vegetacional.geojson` · `camera-traps/data/campaigns/estaciones.csv` · `Fire risk dashboard/README.md` · `Instrumentos Monitoreo Nasampulli CR2.kmz`

**Synology:** `SynologyDrive/Datos/1. Estacion Meteorológica/` completo, incluyendo `Informe Datos Estación Meteorológica.docx` y `ProformaInvoice.pdf` · `SynologyDrive/1. Estacion Meteorológica/Protocolo estación meteorológica/` (vacía)

**Vault:** `Topics/` — Estacion-Meteorologica, CR800, Bosque-Pehuen, LNAS, Expediente-Reserva-Natural-Bosque-Pehuen · `Meetings/2026-06-16-geo-mountains-coordinacion-inicial.md` · `Resources/estandares-datos-socios-plataforma-territorial.md` · `Sessions/2026-05-06-data-pipeline-session-a-cr800-dst.md`

**Buscado y no encontrado:** ningún archivo CRBasic `.CR8` o `.dld` · ningún directorio de instalación de LoggerNet o Campbell · ningún DEM ni ráster de elevación · ningún documento que nombre un modelo de sensor · ningún registro de calibración · ningún identificador de estación en un registro nacional o internacional

**Bitácora de sesión:** `SecondBrain/Sessions/2026-09-09-estacion-meteorologica-timeline-y-metadatos-wmo.md`
