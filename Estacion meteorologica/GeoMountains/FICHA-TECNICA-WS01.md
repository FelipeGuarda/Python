---
title: "Estación meteorológica Bosque Pehuén (WS-01) — ficha técnica y registro"
project: GEO Mountains
date: 2026-09-10
version: 1.0
---

# Estación meteorológica Bosque Pehuén — WS-01

**Fundación Mar Adentro** · Reserva Bosque Pehuén, Pucón, Región de La Araucanía, Chile
Ficha técnica y descripción del registro entregado · 2026-09-10

| | |
|---|---|
| Entregable de datos | `data/weather_data_WS-01.csv` — un archivo, el registro completo |
| Registros | 265.038 |
| Cobertura | 2018-09-21 16:45 → 2026-04-13 13:00 (UTC−03:00) |
| Intervalo | 15 minutos, estampa al fin del intervalo |
| Completitud | 99,9985 % de la grilla de 15 minutos |
| Reporte de construcción | `registry_report.json` — todas las cifras de este documento salen de ahí |
| Registro construido | 2026-09-10 |
| Versión Word | `FICHA-TECNICA-WS01.docx` — exportada el **2026-09-10** desde este archivo |

---

## 1. Lo que se entrega

### 1.1 El registro

Un archivo CSV, UTF-8, separador coma, decimal punto, una fila por estampa,
nombres de columna según la especificación de datos de FMA para socios de la
plataforma territorial (§3.6).

| | |
|---|---|
| Archivo | `weather_data_WS-01.csv` (53 MB) |
| Filas | 265.038 |
| Columnas | 37 |
| Primera estampa | `2018-09-21T16:45:00-03:00` |
| Última estampa | `2026-04-13T13:00:00-03:00` |
| Slots de 15 min en el período | 265.042 |
| Faltantes | 4 (`2025-09-07 00:00` a `00:45`) |

**Falta una hora en siete años y siete meses.** No hay otros cortes. El registro
se reconstruyó desde el contador `RECORD` del datalogger, que corre contiguo de
11138 a 250834 sin un solo faltante; la columna `record` va en el CSV para que
esa continuidad sea verificable fila por fila.

Totales anuales, calculados sobre el archivo entregado:

| Año | Registros | Precipitación (mm) | Temperatura media (°C) |
|---|---|---|---|
| 2018 (desde 21-09) | 9.725 | 811,1 | 7,25 |
| 2019 | 35.040 | 2.126,3 | 7,12 |
| 2020 | 35.136 | 1.716,8 | 7,14 |
| 2021 | 35.040 | 1.578,5 | 8,17 |
| 2022 | 35.040 | 1.923,5 | 7,04 |
| 2023 | 35.040 | 2.860,9 | 7,17 |
| 2024 | 35.136 | 2.241,5 | 6,98 |
| 2025 | 35.036 | 2.049,4 | 7,67 |
| 2026 (hasta 13-04) | 9.845 | 497,9 | 11,70 |

**Los conteos anuales son el máximo aritmético, no una aproximación.** A 15 minutos
hay 96 registros por día, de modo que un año normal da 365 × 96 = **35.040** y uno
bisiesto 366 × 96 = **35.136**; los 96 registros de diferencia en 2020 y 2024 son
exactamente el 29 de febrero. Cada año completo da su máximo exacto, sin un solo
registro de menos, salvo 2025, que queda en 35.036 por los cuatro registros del
2025-09-07 descritos en §1.5. Que 2023 dé 35.040 exactos confirma además que el
salto de reloj de ese año no perdió datos: sólo movió estampas.

### 1.2 Referencia temporal

**El reloj del datalogger corre en UTC−03:00 fijo, todo el año, sin horario de
verano.** Todas las estampas del CSV llevan el offset explícito `-03:00`. La
estampa marca el **fin** del intervalo: una fila rotulada `14:00` cubre
13:45–14:00.

El reloj tuvo dos eventos en todo el registro, ambos en 2023 y de signo opuesto:

| Evento | Registro | Estampa original | Salto |
|---|---|---|---|
| Adelanto | `RECORD 179607` | 2023-07-13 01:45 | +11:45:00 |
| Corrección | `RECORD 187673` | 2023-10-04 14:30 | −11:45:00 |

Los 8.066 registros entre ambos eventos llevan sus estampas corregidas en
−11:45:00 en el archivo entregado, y están marcados con `clock_corrected = true`
para que la corrección sea auditable y reversible. Ningún otro registro fue
alterado.

### 1.3 Inventario de canales

<!-- GENERADO:inventario -->
El datalogger emite **38 columnas de datos**. 33 se entregan y 5 no. Por estado: **23** opera · **7** interrumpido · **3** inutilizable · **2** servicio · **2** nunca funcionó · **1** derivado.

**Ningún canal está calibrado.** No se ha realizado ninguna calibración en la vida de la
estación, así que eso no distingue un canal de otro y no aparece por fila.

La *ventana con valor no nulo* es el primer y el último valor no nulo observados, medidos
sobre el registro, no una fecha declarada. **Un canal que nunca funcionó igual muestra la
ventana completa**, y eso es exactamente la trampa que esta tabla existe para desarmar: las
fallas no se presentan como nulos sino como ceros y constantes imposibles, así que la
columna de estado — no la ventana — es la que dice si el canal midió algo.

**†** marca las unidades que el logger deja **en blanco** en la fila 3 de la cabecera TOA5:
están inferidas de la magnitud, no declaradas por el instrumento.

| Columna en el CSV | Canal TOA5 | Variable | Unidad | Agregación | Estado | Ventana con valor no nulo | Nota |
|---|---|---|---|---|---|---|---|
| `temperature_air_c_max` | `AirTC_Max` | Temperatura del aire | °C | Max | **Opera** | 2018-09-21 → 2026-04-13 | — |
| `temperature_air_c` | `AirTC_Avg` | Temperatura del aire | °C | Avg | **Opera** | 2018-09-21 → 2026-04-13 | — |
| `temperature_air_c_min` | `AirTC_Min` | Temperatura del aire | °C | Min | **Opera** | 2018-09-21 → 2026-04-13 | — |
| `relative_humidity_pct_max` | `RH_Max` | Humedad relativa | % | Max | **Opera** | 2018-09-21 → 2026-04-13 | — |
| `relative_humidity_pct` | `RH_Avg` | Humedad relativa | % | Avg | **Opera** | 2018-09-21 → 2026-04-13 | — |
| `relative_humidity_pct_min` | `RH_Min` | Humedad relativa | % | Min | **Opera** | 2018-09-21 → 2026-04-13 | — |
| `wind_speed_ms_max` | `WS_ms_Max` | Velocidad del viento | m s⁻¹ | Max | **Opera** | 2018-09-21 → 2026-04-13 | — |
| `wind_speed_ms` | `WS_ms_Avg` | Velocidad del viento | m s⁻¹ | Avg | **Opera** | 2018-09-21 → 2026-04-13 | — |
| `wind_speed_ms_min` | `WS_ms_Min` | Velocidad del viento | m s⁻¹ | Min | **Opera** | 2018-09-21 → 2026-04-13 | — |
| `wind_direction_deg_max` | `WindDir_Max` | Dirección del viento | grados | Max | **Opera** | 2018-09-21 → 2026-04-13 | El máximo de una variable circular en el intervalo no es interpretable; usar Avg y Std |
| `wind_direction_deg` | `WindDir_Avg` | Dirección del viento | grados | Avg | **Opera** | 2018-09-21 → 2026-04-13 | — |
| `wind_direction_deg_min` | `WindDir_Min` | Dirección del viento | grados | Min | **Opera** | 2018-09-21 → 2026-04-13 | El mínimo de una variable circular en el intervalo no es interpretable; usar Avg y Std |
| `wind_direction_deg_std` | `WindDir_Std` | Dirección del viento | grados | Std | **Opera** | 2018-09-21 → 2026-04-13 | — |
| `precipitation_mm` | `Rain_mm_Tot` | Precipitación del intervalo | mm | Tot | **Opera** | 2018-09-21 → 2026-04-13 | Pluviómetro presumiblemente sin calefacción: esperar subcaptura de nieve en invierno |
| `soil_temperature_10cm_c_max` | `T107_10cm_Max` | Temperatura de suelo, 10 cm | °C | Max | **Opera** | 2018-09-21 → 2026-04-13 | — |
| `soil_temperature_10cm_c` | `T107_10cm_Avg` | Temperatura de suelo, 10 cm | °C | Avg | **Opera** | 2018-09-21 → 2026-04-13 | Profundidad nominal del nombre del canal, nunca verificada contra la instalación |
| `soil_temperature_10cm_c_min` | `T107_10cm_Min` | Temperatura de suelo, 10 cm | °C | Min | **Opera** | 2018-09-21 → 2026-04-13 | — |
| `soil_temperature_10cm_c_std` | `T107_10cm_Std` | Temperatura de suelo, 10 cm | °C | Std | **Opera** | 2018-09-21 → 2026-04-13 | — |
| `soil_temperature_50cm_c_max` | `T107_50cm_Max` | Temperatura de suelo, 50 cm | °C | Max | **Opera** | 2018-09-21 → 2026-04-13 | — |
| `soil_temperature_50cm_c` | `T107_50cm_Avg` | Temperatura de suelo, 50 cm | °C | Avg | **Opera** | 2018-09-21 → 2026-04-13 | Profundidad nominal del nombre del canal, nunca verificada contra la instalación |
| `soil_temperature_50cm_c_min` | `T107_50cm_Min` | Temperatura de suelo, 50 cm | °C | Min | **Opera** | 2018-09-21 → 2026-04-13 | — |
| `soil_temperature_50cm_c_std` | `T107_50cm_Std` | Temperatura de suelo, 50 cm | °C | Std | **Opera** | 2018-09-21 → 2026-04-13 | — |
| `dew_point_c` | `PtoRocio_Avg` | Punto de rocío | °C | Avg | **Derivado** | 2018-09-21 → 2026-04-13 | Calculado a bordo por una fórmula no documentada. No es una observación |
| `logger_panel_temperature_c` | `PTemp_C_Avg` | Temperatura del panel del logger | °C | Avg | **Servicio** | 2018-09-21 → 2026-04-13 | Housekeeping del equipo, no una variable geofísica |
| `battery_voltage_v_min` | `BattV_Min` | Voltaje de batería | V | Min | **Servicio** | 2018-09-21 → 2026-04-13 | Housekeeping del equipo, no una variable geofísica |
| *no se entrega* | `BP_mbar_Avg` | Presión de estación | mbar | Avg | **Nunca funcionó** | 2018-09-21 → 2026-04-13 | No es una medición atmosférica: la varianza de siete años es físicamente imposible |
| `surface_distance_m_max` | `DT_Max` | Distancia sónica a la superficie | m † | Max | **Interrumpido** | 2018-09-21 → 2021-09-05 | — |
| `surface_distance_m` | `DT_Avg` | Distancia sónica a la superficie | m † | Avg | **Interrumpido** | 2018-09-21 → 2021-09-05 | Distancia cruda al suelo, NO altura de nieve: convertirla exige la referencia a suelo desnudo, no documentada |
| `surface_distance_m_min` | `DT_Min` | Distancia sónica a la superficie | m † | Min | **Interrumpido** | 2018-09-21 → 2021-09-05 | — |
| `surface_distance_tc_m_max` | `TCDT_Max` | Distancia con corrección de temperatura | m † | Max | **Interrumpido** | 2018-09-21 → 2021-09-05 | — |
| `surface_distance_tc_m_min` | `TCDT_Min` | Distancia con corrección de temperatura | m † | Min | **Interrumpido** | 2018-09-21 → 2021-09-05 | — |
| `surface_distance_quality_max` | `Q_Max` | Índice de calidad del sensor sónico | índice † | Max | **Interrumpido** | 2018-09-21 → 2021-09-05 | Diagnóstico del propio sensor; sirve para filtrar las lecturas de distancia |
| `surface_distance_quality_min` | `Q_Min` | Índice de calidad del sensor sónico | índice † | Min | **Interrumpido** | 2018-09-21 → 2021-09-05 | — |
| `solar_radiation_wm2` | `incomingSW_Avg` | Onda corta incidente | W m⁻² † | Avg | **Opera** | 2018-09-21 → 2026-04-13 | Desviación nocturna negativa. Usar como magnitud relativa, no absoluta |
| *no se entrega* | `incomingLW_Avg` | Onda larga incidente | W m⁻² † | Avg | **Inutilizable** | 2018-09-21 → 2026-04-13 | Sigue a la onda corta y es negativa de noche. No es onda larga |
| *no se entrega* | `outgoingLW_Avg` | Onda larga emitida | W m⁻² † | Avg | **Nunca funcionó** | 2018-09-21 → 2026-04-13 | Idénticamente cero desde el primer día |
| *no se entrega* | `outgoingSW_Avg` | Onda corta reflejada | W m⁻² † | Avg | **Inutilizable** | 2018-09-21 → 2026-04-13 | Signo invertido. Posiblemente recuperable revisando el cableado |
| *no se entrega* | `albedo_Avg` | Albedo | adimensional † | Avg | **Inutilizable** | 2018-09-21 → 2026-04-13 | Calculado a bordo a partir del par de onda corta; hereda sus fallas |

Estados: **Opera** registro completo y físicamente coherente · **Interrumpido** funcionó y
dejó de funcionar; se entrega su ventana de servicio · **Nunca funcionó** sin medición
válida desde el primer día · **Inutilizable** produce valores, pero no la magnitud que
declara · **Derivado** calculado a bordo, no medido · **Servicio** housekeeping del equipo.
<!-- /GENERADO:inventario -->

Además de los canales, el CSV lleva cuatro columnas estructurales: `datetime`,
`station_id`, `record` (el contador del datalogger) y `clock_corrected`.

**Tres cosas que la tabla no puede decir por fila:**

- **Sin control de calidad y sin relleno.** Los valores son los que agregó el
  datalogger. No se aplicaron chequeos de rango, de salto ni de consistencia
  interna, en ninguna etapa, y ningún hueco se rellenó.
- **La desviación nocturna de `solar_radiation_wm2`, cuantificada.** El 47,5 %
  de las filas es negativa, con mediana −1,9 W m⁻² entre ellas, 2.096 filas bajo
  −10 W m⁻² y un mínimo de −328,5 W m⁻².
- **Cómo se recortó el canal sónico.** Operó plenamente de 2018-09 a 2021-03
  (≈2.900 lecturas válidas por mes) y se degradó entre abril y septiembre de 2021
  — 598 · 83 · 52 · 349 · 18 · 3 lecturas mensuales. El corte del canal está
  **declarado** en `2021-09-30`, no inferido de un umbral: se descartaron las 150
  lecturas aisladas posteriores y, dentro de la ventana, las 17.782 lecturas de
  exactamente 0,000 m, que un sensor sónico no puede producir.

### 1.4 Metadatos declarados

| Elemento | Valor | Fuente |
|---|---|---|
| Nombre de la estación | Bosque Pehuén | declarado por FMA |
| Identificador interno | WS-01 | `stations.yaml` |
| Identificador WIGOS (WSI) | No disponible | — |
| Identificador nacional (DGA / DMC) | No disponible | — |
| Tipo de estación | Terrestre fija, automática, un mástil más gabinete | `stations.yaml` |
| Operador | Fundación Mar Adentro | — |
| Organización supervisora | Fundación Mar Adentro | — |
| Territorio | Chile (CHL) · Región de La Araucanía · comuna de Pucón | — |
| Latitud | −39,453642 (WGS84, EPSG:4326) | `stations.yaml`, verificada en terreno |
| Longitud | −71,733092 (WGS84, EPSG:4326) | `stations.yaml`, verificada en terreno |
| Coordenadas UTM | 264.838 E, 5.629.315 N (zona 19S, EPSG:32719) | conversión de las anteriores |
| Altitud | 1.223 m — provisional | API de elevación de Open-Meteo, 2026-09-09 |
| Datum vertical | No disponible | — |
| Método de geoposicionamiento | No disponible | — |
| Dentro de área protegida | Sí · Bosque Pehuén, 868,87 ha, Área de Protección Privada | `boundary.geojson` |
| Cobertura de superficie | Bosque Semidenso, distrito Ondulado, unidad de 11,6 ha | `piso_vegetacional.geojson` (tipología FMA) |
| Topografía local | ≈346 m de desnivel dentro de 1 km del mástil | red de cámaras trampa de Bosque Pehuén |
| Descripción de emplazamiento y obstáculos | No disponible | — |
| Clase de emplazamiento OMM | No disponible | — |
| Clima observado en el sitio | Media anual 7,0–8,2 °C · precipitación anual 1.579–2.861 mm · extremos −11,7 a +33,6 °C | este registro |
| Datalogger | Campbell Scientific CR800, serie 42107 | cabecera TOA5 |
| Sistema operativo del logger | `CR800.Std.31.03` | cabecera TOA5 |
| Programa del logger | `estacion_tres_hermanas.CR8`, firma 10101, tabla `Table1` | cabecera TOA5 |
| Fuente de la observación | Automática, sin componente manual | — |
| Marca, modelo y serie de los sensores | No disponible | — |
| Altura de los sensores sobre el suelo | No disponible | — |
| Profundidad real de las sondas de suelo | No disponible (10 cm y 50 cm son valores nominales) | — |
| Configuración de la instrumentación | No disponible | — |
| Frecuencia de muestreo (scan) | No disponible | — |
| Período de agregación | 15 minutos | cabecera TOA5 |
| Método de agregación | Declarado por canal: Max, Avg, Min, Std, Tot | cabecera TOA5 |
| Convención de estampa | Fin de intervalo | cabecera TOA5, confirmada por mediodía solar |
| Referencia temporal | UTC−03:00 fijo, sin horario de verano | derivado de este registro |
| Intervalo de reporte | 15 minutos, sin remuestreo | — |
| Resolución numérica | Cuatro cifras significativas, como las emite el logger | los archivos |
| Nivel de dato | Agregado por el logger, sin post-proceso | — |
| Formato de dato | TOA5 nativo; CSV UTF-8 e ISO 8601 en la entrega | — |
| Fecha de inicio de operación | Primer registro 2018-09-21 16:45; inicio de registro ≈2018-05-28 por retro-extrapolación del contador | `RECORD` = 11138 |
| Fecha de cierre | No aplica; la estación no está cerrada | — |
| Estado operativo | Abierta, **sin transmitir desde 2026-04-13 13:00** (falla de antena) | `cr800_state.json` |
| Historial de calibración | **Ninguna calibración se ha realizado** | — |
| Resultados de calibración | No disponible | — |
| Trazabilidad a patrón de referencia | No disponible | — |
| Incertidumbre de medición | No disponible | — |
| Sistema de control de calidad | No disponible | — |
| Esquema de banderas de calidad | No disponible | — |
| Bitácora de mantención y eventos | No disponible | — |
| Propósito de la observación | Monitoreo biofísico de área protegida; índice de riesgo de incendio; cruces fauna × clima | — |
| Afiliación a red o programa | Ninguna formal a la fecha | — |
| Política de datos y licencia | No definida | — |
| DOI o cita | No disponible | — |
| Punto focal | Felipe Guarda, Fundación Mar Adentro | — |
| Casilla institucional de contacto | No disponible | — |

### 1.5 Un defecto que queda dentro del archivo

**`2025-09-07`, entre 00:00 y 01:00.** Faltan los cuatro registros de 00:00 a
00:45, y el registro de 01:00 no corresponde a esa hora.

El tramo posterior al 2025-07-23 existe solo en la copia de base de datos de FMA,
y esa copia no conserva el contador `RECORD` del datalogger. La causa es de
ingesta, no del instrumento: los tres caminos de ingesta descartaban el contador,
corregido el 2026-09-10. El mecanismo de la pérdida está confirmado en el código y
reproducido — la localización horaria usa `nonexistent='shift_forward'`, así que
las cuatro estampas inexistentes de 00:00–00:45 y la real de 01:00 caen sobre un
mismo instante UTC y la clave primaria deja una de las cinco. Medido en las cinco
transiciones de septiembre contrastables contra los volcados (2019, 2020, 2021,
2022 y 2024); en las doce transiciones anteriores el defecto se corrigió con los
volcados, y aquí no alcanzan.

**Es recuperable.** El datalogger conserva el contador y siempre lo conservó, de
modo que descargar `Table1` devuelve los cinco registros con su `RECORD` — y el
contador para todo el tramo posterior al 2025-07-23. El anillo guarda 495,8 días:
si el logger siguió escribiendo, la fila más antigua de ese tramo se sobrescribe
hacia **≈2026-12-01**, antes que el plazo del corte de telemetría.

Como contraste general: en las 231.587 filas donde ambas copias cubren el mismo
instante de forma independiente, coinciden en 231.564. Las 23 discrepancias caen
todas en transiciones de horario de verano y en la ventana de estampas repetidas
del 2023-10-04/05, y en las 23 el archivo entregado lleva el valor del volcado
del datalogger, identificado por `RECORD`.

---

## 2. Lo que no se entrega

<!-- GENERADO:retenidos -->
5 canales que el logger emite y que este archivo omite. Ninguno es una
medición, y ninguno se presenta como nulo en los datos crudos: se presentan como ceros y
constantes, de modo que un consumidor que los reciba sin advertencia los leería como
válidos. La evidencia de abajo está calculada sobre el registro completo, no citada.

| Canal | Variable | Unidad | Estado | Evidencia medida sobre los 265.038 registros |
|---|---|---|---|---|
| `BP_mbar_Avg` | Presión de estación | mbar | **Nunca funcionó** | media 636,67 · σ **0,18** · mín 636,02 · máx 638,19 · mediana 636,59<br>No es una medición atmosférica: la varianza de siete años es físicamente imposible |
| `incomingLW_Avg` | Onda larga incidente | W m⁻² † | **Inutilizable** | media 78,85 · σ **208,27** · mín −346,20 · máx 1.199,00 · mediana −0,07 · 25 ceros<br>Sigue a la onda corta y es negativa de noche. No es onda larga |
| `outgoingLW_Avg` | Onda larga emitida | W m⁻² † | **Nunca funcionó** | media 0,00 · σ **0,00** · mín 0,00 · máx 0,00 · mediana 0,00 · 265.038 ceros<br>Idénticamente cero desde el primer día |
| `outgoingSW_Avg` | Onda corta reflejada | W m⁻² † | **Inutilizable** | media −57,06 · σ **60,46** · mín −299,50 · máx 53,04 · mediana −47,23 · 126 ceros<br>Signo invertido. Posiblemente recuperable revisando el cableado |
| `albedo_Avg` | Albedo | adimensional † | **Inutilizable** | media −0,49 · σ **11,29** · mín −1.346,00 · máx 287,40 · mediana 0,11 · 6.380 ceros<br>Calculado a bordo a partir del par de onda corta; hereda sus fallas |
<!-- /GENERADO:retenidos -->

---

## 3. Lo que falta

### 3.1 Se obtiene en una visita a terreno

Todo lo de esta lista es la misma visita, y hay un plazo. `Table1` almacena
47.600 registros, es decir **495,8 días** de memoria a bordo. Si el datalogger
sigue energizado, todo lo posterior al corte de telemetría del 2026-04-13
sobrevive en el anillo hasta aproximadamente el **2027-08-22**, y después se
sobrescribe.

| Qué se obtiene | Qué elemento cierra |
|---|---|
| Descargar `Table1` completo | Recupera el corte desde 2026-04-13 y los cinco registros del 2025-09-07; confirma si el logger siguió registrando |
| Recuperar el programa `estacion_tres_hermanas.CR8` | Frecuencia de muestreo · multiplicadores y offsets · cableado · fórmulas de punto de rocío y albedo · explica el barómetro y los canales de onda larga |
| Fotografiar y anotar cada sensor | Marca, modelo y número de serie de los once sensores |
| Medir alturas sobre el suelo y profundidades reales de las sondas | Altura de sensores · permite reducir el viento a la altura de referencia de 10 m |
| Cuatro fotografías desde el mástil, una por punto cardinal, más croquis de horizonte | Descripción de emplazamiento y obstáculos · clase de emplazamiento OMM |
| Leer el reloj del logger contra una hora de referencia, anotando ambas lecturas crudas | Confirma la referencia temporal declarada, que hoy descansa en un cálculo |
| Anotar si el pluviómetro es calefaccionado | Define si la precipitación invernal lleva salvedad de subcaptura de nieve |
| Registro GNSS estático sobre el mástil, una o dos horas | Altitud definitiva y su datum. FMA no tiene receptor de grado topográfico; UACh y los socios del LNAS sí |

### 3.2 Requiere una decisión institucional, no un dato

Sin costo y sin terreno; son cuatro decisiones de FMA.

- **Nombre canónico de la estación.** Este documento declara *Bosque Pehuén*. El
  registro instrumental contiene además `CR800Series`, `CR800Series_2`,
  `CR800Series BP` y `estacion_tres_hermanas`. Un nombre en un registro
  internacional es permanente y público.
- **Política de datos y licencia.** No está definida. El tablero de riesgo de
  incendio de FMA lleva CC BY-NC 4.0, que cubre el tablero y no estos datos.
- **Casilla institucional de contacto.** Un contacto de registro sobrevive a la
  persona que ocupa el cargo.
- **Depósito versionado con DOI**, si se quiere que la serie sea citable en el
  Atlas de los Andes del Sur.

### 3.3 Requiere gestión externa

- **Identificador WIGOS (WSI)**, si el destino es OSCAR/Surface de la OMM. Se
  tramita a través del Representante Permanente de Chile ante la OMM, es decir la
  Dirección Meteorológica de Chile; FMA no puede autorregistrarse. Es el único
  ítem de esta ficha que FMA no controla, y conviene iniciarlo en paralelo.
- **Fecha de instalación y compra original.** No existe en los archivos
  digitales revisados. La única vía es el archivo administrativo de FMA: una
  orden de compra de 2018 traería, de una vez, fecha de instalación y la lista de
  modelos y series de los sensores.

### 3.4 No se puede reconstruir

- **Los 116 días previos al 2018-09-21.** Los registros 0 a 11.137 se
  sobrescribieron en el anillo antes de la primera descarga.
- **Historial de calibración.** Nunca se realizó ninguna; no hay un registro
  perdido que buscar.

---

## 4. Contacto

Felipe Guarda — Fundación Mar Adentro · `felipe.guarda@fundacionmaradentro.cl`

---

## Anexo · Reproducibilidad

`build_registry.py` construye `data/weather_data_WS-01.csv` y
`registry_report.json` desde las fuentes primarias: los siete volcados TOA5 del
anillo del datalogger y la copia de base de datos de FMA para el tramo posterior
al 2025-07-23. Toda cifra de este documento está en el JSON. El script deduplica
por `RECORD` y no por estampa, aplica la corrección de reloj como dato declarado
y valida que el resultado cierre la grilla de 15 minutos sin duplicados.

```
python build_registry.py
```
