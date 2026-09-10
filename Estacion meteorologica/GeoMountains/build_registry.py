"""Reconstruye el registro completo de WS-01 en tiempo verdadero, en un solo CSV.

Este modulo es dueno de tres decisiones sobre el registro de la estacion, y las
esconde del resto del mundo:

  1. La historia del reloj del datalogger. Corre en UTC-03:00 fijo (sin horario
     de verano) y tiene exactamente dos eventos en todo el registro, simetricos
     y por lo tanto exactamente reparables -- ver CLOCK_EPOCHS.
  2. Como se deduplican los volcados TOA5. Cada .dat es un volcado completo del
     anillo de 47.600 registros del CR800, por lo que se solapan. La clave de
     deduplicacion es RECORD, no el timestamp: el contador es contiguo y
     monotono, el timestamp no lo es (evento de reloj 2 repite 47 estampas).
  3. Como se lee de vuelta la copia en parquet. El pipeline localiza las
     estampas ingenuas del logger como hora civil chilena, que si aplica DST;
     el logger no. Convertir el UTC de vuelta a America/Santiago recupera la
     lectura original del reloj exactamente. Esa lectura es UTC-03:00, no hora
     civil.

Uso:
    python build_registry.py

Escribe data/weather_data_WS-01.csv y registry_report.json en este directorio.
Los numeros que cita la ficha tecnica salen del JSON, no de prosa tipeada a mano.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
DAT_DIR = REPO / "Linea de tiempo"

# data-pipeline es un repositorio hermano, no parte de este. Los ultimos nueve
# meses del registro existen unicamente ahi.
PARQUET_DIR = Path(
    os.environ.get(
        "WS01_PARQUET_DIR",
        REPO.parent / "data-pipeline" / "data" / "recovery" / "weather_station",
    )
)

STATION_ID = "WS-01"
UTC_OFFSET = "-03:00"  # offset fijo del reloj del logger, todo el ano, todo el registro
INTERVAL = pd.Timedelta("15min")


@dataclass(frozen=True)
class ClockEpoch:
    """Ventana de RECORD en que el reloj del logger estuvo desfasado.

    `offset` es lo que hay que RESTAR a la estampa para obtener tiempo verdadero.
    """

    first_record: int
    last_record: int
    offset: pd.Timedelta
    note: str


# Derivado del contador RECORD, que es contiguo de 11138 a 250834 sin faltantes:
# cualquier paso de estampa distinto de 15 min con RECORD contiguo es un evento
# de reloj, no un hueco de datos. Hay dos en siete anos, y suman cero.
CLOCK_EPOCHS: tuple[ClockEpoch, ...] = (
    ClockEpoch(
        first_record=179607,
        last_record=187672,
        offset=pd.Timedelta("11:45:00"),
        note=(
            "Reloj adelantado 11:45:00. Salto +11:45 en REC 179607 "
            "(estampado 2023-07-13 01:45), corregido -11:45 en REC 187673 "
            "(estampado 2023-10-04 14:30)."
        ),
    ),
)


#: Vocabulario de estado. Un canal tiene exactamente uno.
#:   Opera          - registro completo y fisicamente coherente
#:   Interrumpido   - funciono y dejo de funcionar; se entrega su ventana de servicio
#:   Nunca funciono - sin medicion valida desde el primer dia
#:   Inutilizable   - produce valores, pero no la magnitud que declara
#:   Derivado       - calculado a bordo del datalogger, no medido
#:   Servicio       - housekeeping del equipo, no una variable geofisica
STATUSES = ("Opera", "Interrumpido", "Nunca funcionó", "Inutilizable", "Derivado", "Servicio")


@dataclass(frozen=True)
class Channel:
    """Un canal del datalogger: identidad, unidad, agregacion y estado operativo.

    `delivered_as` es el nombre en el CSV entregado, o None si el canal se mide y
    se declara pero no se entrega. `unit_declared` dice si la unidad la escribio
    el logger en la fila 3 de la cabecera TOA5 o si la inferimos: el logger deja
    la unidad EN BLANCO para los doce canales finales -- todo el bloque de
    radiacion y el SR50 -- y esa distincion pertenece al metadato.
    """

    toa5: str  # nombre nativo en la cabecera TOA5
    parquet: str  # nombre en la copia de base de datos de FMA
    delivered_as: str | None
    variable: str
    unit: str
    unit_declared: bool
    aggregation: str
    group: str
    status: str
    note: str = ""

    @property
    def column(self) -> str:
        """Nombre interno con que viaja el canal mientras se construye el registro."""
        return self.delivered_as or self.toa5

    def __post_init__(self) -> None:
        if self.status not in STATUSES:
            raise ValueError(f"{self.toa5}: estado '{self.status}' fuera del vocabulario")


# INVENTARIO DE CANALES -- la unica lista. Nomenclatura de las columnas
# entregadas tomada de estandares-datos-socios-plataforma-territorial.md §3.6, la
# especificacion que FMA exige a sus socios; los companeros Max/Min/Std extienden
# esos nombres con sufijo en vez de inventar un esquema nuevo.
#
# `unit_declared=False` marca los doce canales que el logger deja SIN unidad en la
# fila 3 de la cabecera TOA5. Su unidad esta inferida de la magnitud, no declarada
# por el instrumento.
#
# NINGUN CANAL ESTA CALIBRADO. No se ha realizado ninguna calibracion en la vida
# de la estacion, de modo que eso no distingue a un canal de otro y no aparece
# como nota por fila.
CHANNELS: tuple[Channel, ...] = (
    # --- aire -------------------------------------------------------------
    Channel("AirTC_Max", "AirTC_Max", "temperature_air_c_max", "Temperatura del aire", "°C", True, "Max", "aire", "Opera"),
    Channel("AirTC_Avg", "temperature_air", "temperature_air_c", "Temperatura del aire", "°C", True, "Avg", "aire", "Opera"),
    Channel("AirTC_Min", "AirTC_Min", "temperature_air_c_min", "Temperatura del aire", "°C", True, "Min", "aire", "Opera"),
    Channel("RH_Max", "RH_Max", "relative_humidity_pct_max", "Humedad relativa", "%", True, "Max", "aire", "Opera"),
    Channel("RH_Avg", "relative_humidity", "relative_humidity_pct", "Humedad relativa", "%", True, "Avg", "aire", "Opera"),
    Channel("RH_Min", "RH_Min", "relative_humidity_pct_min", "Humedad relativa", "%", True, "Min", "aire", "Opera"),
    # --- viento -----------------------------------------------------------
    Channel("WS_ms_Max", "WS_ms_Max", "wind_speed_ms_max", "Velocidad del viento", "m s⁻¹", True, "Max", "viento", "Opera"),
    Channel("WS_ms_Avg", "wind_speed", "wind_speed_ms", "Velocidad del viento", "m s⁻¹", True, "Avg", "viento", "Opera"),
    Channel("WS_ms_Min", "WS_ms_Min", "wind_speed_ms_min", "Velocidad del viento", "m s⁻¹", True, "Min", "viento", "Opera"),
    Channel("WindDir_Max", "WindDir_Max", "wind_direction_deg_max", "Dirección del viento", "grados", True, "Max", "viento", "Opera",
            "El máximo de una variable circular en el intervalo no es interpretable; usar Avg y Std"),
    Channel("WindDir_Avg", "wind_direction", "wind_direction_deg", "Dirección del viento", "grados", True, "Avg", "viento", "Opera"),
    Channel("WindDir_Min", "WindDir_Min", "wind_direction_deg_min", "Dirección del viento", "grados", True, "Min", "viento", "Opera",
            "El mínimo de una variable circular en el intervalo no es interpretable; usar Avg y Std"),
    Channel("WindDir_Std", "WindDir_Std", "wind_direction_deg_std", "Dirección del viento", "grados", True, "Std", "viento", "Opera"),
    # --- precipitacion ----------------------------------------------------
    Channel("Rain_mm_Tot", "precipitation", "precipitation_mm", "Precipitación del intervalo", "mm", True, "Tot", "precipitacion", "Opera",
            "Pluviómetro presumiblemente sin calefacción: esperar subcaptura de nieve en invierno"),
    # --- suelo ------------------------------------------------------------
    Channel("T107_10cm_Max", "T107_10cm_Max", "soil_temperature_10cm_c_max", "Temperatura de suelo, 10 cm", "°C", True, "Max", "suelo", "Opera"),
    Channel("T107_10cm_Avg", "T107_10cm_Avg", "soil_temperature_10cm_c", "Temperatura de suelo, 10 cm", "°C", True, "Avg", "suelo", "Opera",
            "Profundidad nominal del nombre del canal, nunca verificada contra la instalación"),
    Channel("T107_10cm_Min", "T107_10cm_Min", "soil_temperature_10cm_c_min", "Temperatura de suelo, 10 cm", "°C", True, "Min", "suelo", "Opera"),
    Channel("T107_10cm_Std", "T107_10cm_Std", "soil_temperature_10cm_c_std", "Temperatura de suelo, 10 cm", "°C", True, "Std", "suelo", "Opera"),
    Channel("T107_50cm_Max", "T107_50cm_Max", "soil_temperature_50cm_c_max", "Temperatura de suelo, 50 cm", "°C", True, "Max", "suelo", "Opera"),
    Channel("T107_50cm_Avg", "T107_50cm_Avg", "soil_temperature_50cm_c", "Temperatura de suelo, 50 cm", "°C", True, "Avg", "suelo", "Opera",
            "Profundidad nominal del nombre del canal, nunca verificada contra la instalación"),
    Channel("T107_50cm_Min", "T107_50cm_Min", "soil_temperature_50cm_c_min", "Temperatura de suelo, 50 cm", "°C", True, "Min", "suelo", "Opera"),
    Channel("T107_50cm_Std", "T107_50cm_Std", "soil_temperature_50cm_c_std", "Temperatura de suelo, 50 cm", "°C", True, "Std", "suelo", "Opera"),
    # --- derivado y servicio ----------------------------------------------
    Channel("PtoRocio_Avg", "PtoRocio_Avg", "dew_point_c", "Punto de rocío", "°C", True, "Avg", "derivado", "Derivado",
            "Calculado a bordo por una fórmula no documentada. No es una observación"),
    Channel("PTemp_C_Avg", "PTemp_C_Avg", "logger_panel_temperature_c", "Temperatura del panel del logger", "°C", True, "Avg", "servicio", "Servicio",
            "Housekeeping del equipo, no una variable geofísica"),
    Channel("BattV_Min", "battery_voltage", "battery_voltage_v_min", "Voltaje de batería", "V", True, "Min", "servicio", "Servicio",
            "Housekeeping del equipo, no una variable geofísica"),
    # --- presion ----------------------------------------------------------
    Channel("BP_mbar_Avg", "BP_mbar_Avg", None, "Presión de estación", "mbar", True, "Avg", "presion", "Nunca funcionó",
            "No es una medición atmosférica: la varianza de siete años es físicamente imposible"),
    # --- SR50, distancia sonica a la superficie ----------------------------
    Channel("DT_Max", "DT_Max", "surface_distance_m_max", "Distancia sónica a la superficie", "m", False, "Max", "sr50", "Interrumpido"),
    Channel("DT_Avg", "DT_Avg", "surface_distance_m", "Distancia sónica a la superficie", "m", False, "Avg", "sr50", "Interrumpido",
            "Distancia cruda al suelo, NO altura de nieve: convertirla exige la referencia a suelo desnudo, no documentada"),
    Channel("DT_Min", "DT_Min", "surface_distance_m_min", "Distancia sónica a la superficie", "m", False, "Min", "sr50", "Interrumpido"),
    Channel("TCDT_Max", "TCDT_Max", "surface_distance_tc_m_max", "Distancia con corrección de temperatura", "m", False, "Max", "sr50", "Interrumpido"),
    Channel("TCDT_Min", "TCDT_Min", "surface_distance_tc_m_min", "Distancia con corrección de temperatura", "m", False, "Min", "sr50", "Interrumpido"),
    Channel("Q_Max", "Q_Max", "surface_distance_quality_max", "Índice de calidad del sensor sónico", "índice", False, "Max", "sr50", "Interrumpido",
            "Diagnóstico del propio sensor; sirve para filtrar las lecturas de distancia"),
    Channel("Q_Min", "Q_Min", "surface_distance_quality_min", "Índice de calidad del sensor sónico", "índice", False, "Min", "sr50", "Interrumpido"),
    # --- radiacion --------------------------------------------------------
    Channel("incomingSW_Avg", "solar_radiation", "solar_radiation_wm2", "Onda corta incidente", "W m⁻²", False, "Avg", "radiacion", "Opera",
            "Desviación nocturna negativa. Usar como magnitud relativa, no absoluta"),
    Channel("incomingLW_Avg", "incomingLW_Avg", None, "Onda larga incidente", "W m⁻²", False, "Avg", "radiacion", "Inutilizable",
            "Sigue a la onda corta y es negativa de noche. No es onda larga"),
    Channel("outgoingLW_Avg", "outgoingLW_Avg", None, "Onda larga emitida", "W m⁻²", False, "Avg", "radiacion", "Nunca funcionó",
            "Idénticamente cero desde el primer día"),
    Channel("outgoingSW_Avg", "outgoingSW_Avg", None, "Onda corta reflejada", "W m⁻²", False, "Avg", "radiacion", "Inutilizable",
            "Signo invertido. Posiblemente recuperable revisando el cableado"),
    Channel("albedo_Avg", "albedo_Avg", None, "Albedo", "adimensional", False, "Avg", "radiacion", "Inutilizable",
            "Calculado a bordo a partir del par de onda corta; hereda sus fallas"),
)

#: Los canales que llegan al CSV. Derivado, no una segunda lista que mantener.
DELIVERED: tuple[Channel, ...] = tuple(c for c in CHANNELS if c.delivered_as)

# Defectos que quedan dentro del CSV y no se pueden reparar desde aca. El tramo
# posterior al 2025-07-23 existe solo en la copia de base de datos de FMA, y esa
# copia no conserva el contador: los tres caminos de ingesta lo descartaban
# (corregido en data-pipeline el 2026-09-10, ver src/cr800_columns.py). El
# datalogger si lo tiene y siempre lo tuvo, de modo que la proxima descarga de
# Table1 devuelve el contador para todo este tramo -- ver "recovery".
KNOWN_DEFECTS: tuple[dict, ...] = (
    {
        "window": "2025-09-07 00:00 a 01:00",
        "defect": (
            "Faltan los cuatro registros de 00:00 a 00:45, y el registro de 01:00 "
            "no corresponde a esa hora."
        ),
        "evidence": (
            "Mecanismo confirmado en el codigo y reproducido: tz_utils.py localiza con "
            "nonexistent='shift_forward', de modo que las cuatro estampas inexistentes "
            "de 00:00-00:45 y la real de 01:00 caen sobre un mismo instante UTC, y la "
            "clave primaria (station_id, timestamp) deja una de las cinco. Verificado "
            "sobre 2019-09-08 (RECORD 44863-44867 -> 2019-09-08 04:00Z) y medido en las "
            "cinco transiciones de septiembre contrastables contra los volcados TOA5 "
            "(2019, 2020, 2021, 2022, 2024). Los volcados no alcanzan al 2025-09-07, "
            "asi que aqui el patron solo se puede declarar."
        ),
        "recovery": (
            "Descargar Table1 del datalogger recupera los cinco registros con su RECORD. "
            "El anillo guarda 495,8 dias, de modo que si el logger siguio escribiendo la "
            "fila mas antigua de este tramo (2025-07-23 13:00) se sobrescribe hacia "
            "2026-12-01."
        ),
    },
)

# Ventana de servicio declarada del sensor sonico de distancia. Lecturas validas
# por mes: ~2.900 hasta 2021-03, luego 598 (abr), 83 (may), 52 (jun), 349 (jul),
# 18 (ago), 3 (sep), y nada en oct-dic 2021. Despues de esa fecha quedan 150
# lecturas aisladas en cuatro anos -- ruido, no medicion. El corte es una regla
# declarada, no un umbral sobre los valores.
SR50_SERVICE_END = pd.Timestamp("2021-09-30 23:45:00")

# Dentro de la ventana, el SR50 tampoco puede reportar exactamente 0,000 m.
# Donde lo hace no hay medicion, y el bloque completo del sensor se anula.
SR50_SENTINEL_COLUMN = "DT_Avg"


def _read_toa5_dumps() -> pd.DataFrame:
    """Los siete volcados del anillo, deduplicados por RECORD, en tiempo verdadero.

    Valida en el borde: RECORD contiguo, y una sola estampa por RECORD.
    """
    paths = sorted(DAT_DIR.glob("*.dat"))
    if not paths:
        raise FileNotFoundError(f"No hay volcados TOA5 en {DAT_DIR}")

    frames = []
    for path in paths:
        frame = pd.read_csv(
            path,
            skiprows=[0, 2, 3],  # cabecera TOA5: linea 1 estacion, 3 unidades, 4 agregacion
            header=0,
            na_values=["NAN"],
            low_memory=False,
        )
        frame["TIMESTAMP"] = pd.to_datetime(frame["TIMESTAMP"])
        frames.append(frame)

    raw = pd.concat(frames, ignore_index=True)
    per_record = raw.groupby("RECORD")["TIMESTAMP"].nunique()
    if (per_record > 1).any():
        raise ValueError("Un mismo RECORD trae estampas distintas; el anillo no es coherente")

    dat = raw.drop_duplicates(subset=["RECORD"]).sort_values("RECORD").reset_index(drop=True)
    missing = int(dat["RECORD"].diff().fillna(1).ne(1).sum())
    if missing:
        raise ValueError(f"El contador RECORD tiene {missing} discontinuidades; se esperaban 0")

    dat["true_time"] = dat["TIMESTAMP"]
    dat["clock_corrected"] = False
    for epoch in CLOCK_EPOCHS:
        window = dat["RECORD"].between(epoch.first_record, epoch.last_record)
        dat.loc[window, "true_time"] = dat.loc[window, "TIMESTAMP"] - epoch.offset
        dat.loc[window, "clock_corrected"] = True

    out = pd.DataFrame({
        "true_time": dat["true_time"],
        "record": dat["RECORD"].astype("int64"),
        "clock_corrected": dat["clock_corrected"],
    })
    for channel in CHANNELS:
        out[channel.column] = dat[channel.toa5]
    return out.sort_values("true_time").reset_index(drop=True)


def _read_pipeline_copy() -> pd.DataFrame:
    """La copia del pipeline, con la lectura original del reloj recuperada.

    El UTC almacenado se convierte de vuelta a America/Santiago, lo que devuelve
    la estampa ingenua que escribio el logger. Esa estampa es UTC-03:00 fija; la
    copia no tiene RECORD, por lo que no se le puede aplicar correccion de reloj.
    """
    paths = sorted(PARQUET_DIR.glob("*.parquet"))
    if not paths:
        raise FileNotFoundError(f"No hay parquet del pipeline en {PARQUET_DIR}")

    parquet = pd.concat([pd.read_parquet(p) for p in paths], ignore_index=True)
    out = pd.DataFrame({
        "true_time": parquet["timestamp"].dt.tz_convert("America/Santiago").dt.tz_localize(None),
        "record": pd.Series(pd.NA, index=parquet.index, dtype="Int64"),
        "clock_corrected": False,
    })
    for channel in CHANNELS:
        out[channel.column] = parquet[channel.parquet]
    return out.sort_values("true_time").reset_index(drop=True)


def _cross_check(logger: pd.DataFrame, pipeline: pd.DataFrame) -> dict:
    """Compara las dos copias donde ambas cubren el mismo instante.

    Los volcados TOA5 son la fuente autoritativa: traen RECORD, y por eso
    resuelven tanto el solape del anillo como la ventana de reloj desfasado. La
    copia del pipeline deduplica por estampa, que en esas zonas no es unica.
    Toda discrepancia se espera ahi, y esta funcion la cuantifica en vez de
    suponerla.
    """
    probe = "temperature_air_c"
    comparable = logger.loc[~logger["clock_corrected"], ["true_time", "record", probe]]
    paired = comparable.merge(
        pipeline[["true_time", probe]], on="true_time", suffixes=("_toa5", "_pipeline")
    )
    differing = paired.loc[
        (paired[f"{probe}_toa5"] - paired[f"{probe}_pipeline"]).abs() > 1e-9
    ]
    return {
        "probe_channel": probe,
        "rows_compared": int(len(paired)),
        "rows_differing": int(len(differing)),
        "dates_differing": sorted({d.strftime("%Y-%m-%d") for d in differing["true_time"]}),
        "note": (
            "Donde difieren, el CSV lleva el valor del volcado TOA5, identificado "
            "por RECORD. Las discrepancias caen en transiciones DST y en la ventana "
            "de estampas repetidas del 2023-10-04/05."
        ),
    }


def _mask_sr50_out_of_service(registry: pd.DataFrame) -> dict:
    """Deja el bloque SR50 solo donde hay medicion: dentro de la ventana y != 0."""
    sr50 = [c.column for c in CHANNELS if c.group == "sr50"]
    sentinel = next(c.column for c in CHANNELS if c.toa5 == SR50_SENTINEL_COLUMN)

    after_service = registry["true_time"].gt(SR50_SERVICE_END)
    zero_reading = registry[sentinel].eq(0)
    dropped_after_service = int((after_service & ~zero_reading).sum())
    dropped_zero = int((~after_service & zero_reading).sum())

    registry.loc[after_service | zero_reading, sr50] = pd.NA
    return {
        "service_end": SR50_SERVICE_END.isoformat(sep=" "),
        "readings_discarded_after_service_end": dropped_after_service,
        "readings_discarded_zero": dropped_zero,
    }


def _gap_runs(missing: list[pd.Timestamp]) -> list[dict]:
    runs = []
    for slot in missing:
        if runs and slot - runs[-1]["_last"] == INTERVAL:
            runs[-1]["_last"] = slot
            runs[-1]["slots"] += 1
        else:
            runs.append({"_last": slot, "from": slot, "slots": 1})
    return [
        {
            "from": r["from"].isoformat(sep=" "),
            "to": r["_last"].isoformat(sep=" "),
            "slots": r["slots"],
        }
        for r in runs
    ]


def _channel_inventory(registry: pd.DataFrame) -> list[dict]:
    """El inventario de los 38 canales, con lo declarado y lo medido en cada uno.

    Se calcula sobre el marco completo -- incluidos los canales que no se
    entregan -- para que las cifras que cita la ficha sobre un canal retenido
    tambien sean medidas y no prosa tipeada a mano.
    """
    inventory = []
    for channel in CHANNELS:
        series = registry[channel.column].astype("float64")
        valid = series.dropna()
        window = registry.loc[series.notna(), "datetime"]
        stat = (lambda v: round(float(v), 4)) if valid.size else (lambda v: None)
        inventory.append({
            "toa5": channel.toa5,
            "delivered_as": channel.delivered_as,
            "variable": channel.variable,
            "unit": channel.unit,
            "unit_declared_by_logger": channel.unit_declared,
            "aggregation": channel.aggregation,
            "group": channel.group,
            "status": channel.status,
            "note": channel.note,
            "n_valid": int(valid.size),
            "n_zero": int((valid == 0).sum()),
            "first_valid": window.min() if valid.size else None,
            "last_valid": window.max() if valid.size else None,
            "min": stat(valid.min()) if valid.size else None,
            "max": stat(valid.max()) if valid.size else None,
            "mean": stat(valid.mean()) if valid.size else None,
            "median": stat(valid.median()) if valid.size else None,
            "std": stat(valid.std()) if valid.size else None,
        })
    return inventory


def _es(value: float | int | None, decimals: int = 2) -> str:
    """Un numero con coma decimal y separador de miles, como el resto del documento."""
    if value is None:
        return "—"
    text = f"{value:,.{decimals}f}".replace(",", "\x00").replace(".", ",").replace("\x00", ".")
    return text.replace("-", "−")  # signo menos, no guion, como el resto del documento


def _render_inventory(inventory: list[dict]) -> str:
    """La tabla unica: las 38 columnas del logger, su unidad y su estado."""
    head = (
        "| Columna en el CSV | Canal TOA5 | Variable | Unidad | Agregación | Estado "
        "| Ventana con valor no nulo | Nota |\n"
        "|---|---|---|---|---|---|---|---|"
    )
    rows = []
    for item in inventory:
        delivered = f"`{item['delivered_as']}`" if item["delivered_as"] else "*no se entrega*"
        unit = item["unit"] if item["unit_declared_by_logger"] else f"{item['unit']} †"
        if item["n_valid"]:
            window = f"{item['first_valid'][:10]} → {item['last_valid'][:10]}"
        else:
            window = "sin dato válido"
        rows.append(
            f"| {delivered} | `{item['toa5']}` | {item['variable']} | {unit} | "
            f"{item['aggregation']} | **{item['status']}** | {window} | {item['note'] or '—'} |"
        )
    tally = {}
    for item in inventory:
        tally[item["status"]] = tally.get(item["status"], 0) + 1
    counts = " · ".join(f"**{n}** {status.lower()}" for status, n in
                        sorted(tally.items(), key=lambda kv: -kv[1]))

    return "\n".join([
        f"El datalogger emite **{len(inventory)} columnas de datos**. "
        f"{sum(1 for i in inventory if i['delivered_as'])} se entregan y "
        f"{sum(1 for i in inventory if not i['delivered_as'])} no. Por estado: {counts}.",
        "",
        "**Ningún canal está calibrado.** No se ha realizado ninguna calibración en la vida de la",
        "estación, así que eso no distingue un canal de otro y no aparece por fila.",
        "",
        "La *ventana con valor no nulo* es el primer y el último valor no nulo observados, medidos",
        "sobre el registro, no una fecha declarada. **Un canal que nunca funcionó igual muestra la",
        "ventana completa**, y eso es exactamente la trampa que esta tabla existe para desarmar: las",
        "fallas no se presentan como nulos sino como ceros y constantes imposibles, así que la",
        "columna de estado — no la ventana — es la que dice si el canal midió algo.",
        "",
        "**†** marca las unidades que el logger deja **en blanco** en la fila 3 de la cabecera TOA5:",
        "están inferidas de la magnitud, no declaradas por el instrumento.",
        "",
        head,
        *rows,
        "",
        "Estados: **Opera** registro completo y físicamente coherente · **Interrumpido** funcionó y",
        "dejó de funcionar; se entrega su ventana de servicio · **Nunca funcionó** sin medición",
        "válida desde el primer día · **Inutilizable** produce valores, pero no la magnitud que",
        "declara · **Derivado** calculado a bordo, no medido · **Servicio** housekeeping del equipo.",
    ])


def _render_withheld(inventory: list[dict]) -> str:
    """Los canales retenidos, con la estadistica medida que sostiene cada veredicto."""
    withheld = [i for i in inventory if not i["delivered_as"]]
    head = (
        "| Canal | Variable | Unidad | Estado | Evidencia medida sobre los 265.038 registros |\n"
        "|---|---|---|---|---|"
    )
    rows = []
    for item in withheld:
        unit = item["unit"] if item["unit_declared_by_logger"] else f"{item['unit']} †"
        evidence = (
            f"media {_es(item['mean'])} · σ **{_es(item['std'])}** · "
            f"mín {_es(item['min'])} · máx {_es(item['max'])} · "
            f"mediana {_es(item['median'])}"
        )
        if item["n_zero"]:
            evidence += f" · {_es(item['n_zero'], 0)} ceros"
        rows.append(
            f"| `{item['toa5']}` | {item['variable']} | {unit} | **{item['status']}** | "
            f"{evidence}<br>{item['note']} |"
        )
    return "\n".join([
        f"{len(withheld)} canales que el logger emite y que este archivo omite. Ninguno es una",
        "medición, y ninguno se presenta como nulo en los datos crudos: se presentan como ceros y",
        "constantes, de modo que un consumidor que los reciba sin advertencia los leería como",
        "válidos. La evidencia de abajo está calculada sobre el registro completo, no citada.",
        "",
        head,
        *rows,
    ])


def _splice(path: Path, marker: str, body: str) -> None:
    """Reemplaza el bloque generado que lleva ese marcador. Falla si no existe.

    La ficha es un documento para la contraparte, no un anexo, asi que la tabla
    vive dentro de ella; estos marcadores son lo que la mantiene regenerable en
    vez de tipeada a mano.
    """
    begin, end = f"<!-- GENERADO:{marker} -->", f"<!-- /GENERADO:{marker} -->"
    text = path.read_text(encoding="utf-8")
    if begin not in text or end not in text:
        raise ValueError(f"{path.name}: faltan los marcadores {begin} / {end}")
    before = text.split(begin)[0]
    after = text.split(end, 1)[1]
    path.write_text(f"{before}{begin}\n{body}\n{end}{after}", encoding="utf-8")


def build(out_dir: Path = HERE) -> dict:
    """Escribe el CSV consolidado y el reporte, y devuelve el reporte."""
    logger = _read_toa5_dumps()
    pipeline = _read_pipeline_copy()
    cross_check = _cross_check(logger, pipeline)

    tail = pipeline.loc[pipeline["true_time"] > logger["true_time"].max()]
    registry = pd.concat([logger, tail], ignore_index=True).sort_values("true_time")

    if registry["true_time"].duplicated().any():
        raise ValueError("Estampas duplicadas tras la union; la correccion de reloj no cierra")

    sr50 = _mask_sr50_out_of_service(registry)

    registry.insert(0, "station_id", STATION_ID)
    registry.insert(0, "datetime", registry.pop("true_time").dt.strftime(f"%Y-%m-%dT%H:%M:%S{UTC_OFFSET}"))
    registry = registry.reset_index(drop=True)

    grid = pd.date_range(logger["true_time"].min(), tail["true_time"].max(), freq=INTERVAL)
    present = set(pd.to_datetime(registry["datetime"].str.slice(0, 19)))
    missing = sorted(set(grid) - present)

    # Medir antes de recortar: los canales retenidos tambien se declaran, y sus
    # cifras deben venir del registro y no de prosa.
    inventory = _channel_inventory(registry)
    delivered = ["datetime", "station_id", "record", "clock_corrected"]
    delivered += [c.delivered_as for c in DELIVERED]
    registry = registry[delivered]

    csv_path = out_dir / "data" / f"weather_data_{STATION_ID}.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    registry.to_csv(csv_path, index=False, encoding="utf-8", lineterminator="\n")

    years = pd.to_datetime(registry["datetime"].str.slice(0, 19)).dt.year
    report = {
        "generated": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
        "station_id": STATION_ID,
        "time_reference": f"UTC{UTC_OFFSET}",
        "interval": "15min",
        "timestamp_convention": "fin de intervalo",
        "csv": {
            "path": csv_path.relative_to(out_dir).as_posix(),
            "rows": int(len(registry)),
            "columns": int(len(registry.columns)),
            "bytes": int(csv_path.stat().st_size),
        },
        "coverage": {
            "first": registry["datetime"].iloc[0],
            "last": registry["datetime"].iloc[-1],
            "grid_slots": int(len(grid)),
            "records": int(len(registry)),
            "missing_slots": len(missing),
            "completeness_pct": round(100 * len(registry) / len(grid), 4),
            "missing_runs": _gap_runs(missing),
        },
        "clock": {
            "offset": f"UTC{UTC_OFFSET}",
            "daylight_saving": False,
            "events": [
                {
                    "first_record": e.first_record,
                    "last_record": e.last_record,
                    "correction_applied": f"-{e.offset.components.hours:02d}:{e.offset.components.minutes:02d}:00",
                    "note": e.note,
                }
                for e in CLOCK_EPOCHS
            ],
            "records_corrected": int(registry["clock_corrected"].sum()),
        },
        "sources": {
            "toa5_dumps": sorted(p.name for p in DAT_DIR.glob("*.dat")),
            "toa5_records": int(len(logger)),
            "toa5_record_range": [int(logger["record"].min()), int(logger["record"].max())],
            "pipeline_parquet_records": int(len(tail)),
        },
        "cross_check": cross_check,
        "known_defects": list(KNOWN_DEFECTS),
        "sr50": sr50,
        "channel_inventory": inventory,
        "annual": [
            {
                "year": int(year),
                "records": int(group_size),
                "precipitation_mm": round(float(registry.loc[years == year, "precipitation_mm"].sum()), 1),
                "temperature_air_c_mean": round(float(registry.loc[years == year, "temperature_air_c"].mean()), 2),
            }
            for year, group_size in years.value_counts().sort_index().items()
        ],
    }

    report_path = out_dir / "registry_report.json"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    ficha = out_dir / "FICHA-TECNICA-WS01.md"
    _splice(ficha, "inventario", _render_inventory(inventory))
    _splice(ficha, "retenidos", _render_withheld(inventory))

    return report


def main() -> None:
    report = build()
    cov = report["coverage"]
    print(f"CSV      {report['csv']['path']}  {report['csv']['rows']:,} filas x {report['csv']['columns']} columnas")
    print(f"Cobertura {cov['first']} -> {cov['last']}")
    print(f"          {cov['records']:,} de {cov['grid_slots']:,} slots  ({cov['completeness_pct']}%)")
    for run in cov["missing_runs"]:
        print(f"          falta {run['from']} -> {run['to']}  ({run['slots']} slots)")
    print(f"Reloj     {report['clock']['records_corrected']:,} registros corregidos")
    xc = report["cross_check"]
    print(f"Contraste {xc['rows_compared']:,} filas comparadas con la copia del pipeline, "
          f"{xc['rows_differing']} discrepancias en {len(xc['dates_differing'])} fechas")
    sr50 = report["sr50"]
    print(f"SR50      en servicio hasta {sr50['service_end']}; "
          f"descartadas {sr50['readings_discarded_after_service_end']:,} lecturas posteriores "
          f"y {sr50['readings_discarded_zero']:,} lecturas en cero")


if __name__ == "__main__":
    main()
