"""
camtrap/visit_schema.py — what a field visit must record, and in what words.

WHAT THIS OWNS

    The shape of a field visit as the technician writes it down: which facts are
    collected, the Spanish wording they read on the sheet, the answers they are
    allowed to give, when a field becomes obligatory, and which `field_notes.csv`
    column each one lands in.

    One declaration. `setup/build_visit_template.py` renders it to Excel; whatever
    later reads a filled workbook resolves headers through `by_label`. Neither knows
    the field list — they ask this module. Adding a covariate is an edit here and
    nowhere else.

RAW READINGS, NEVER CONCLUSIONS

    There is no `clock_state`, no `clock_action`, no `clock_offset_hours` on this
    form, and their absence is deliberate. Those columns exist in the legacy record
    and they are what emptied `camera_datetime_observed` on all 26 otoño 2026 rows:
    asked for a verdict, the technician supplied one (`shifted, -1.0`) and the
    observation it came from was lost.

    The technician also cannot produce that verdict honestly. They compare the camera
    against a phone that silently adjusts itself, so "camera holds a fixed offset and
    civil time moved" and "camera drifted or reset" look identical at the tree. A
    pair of raw readings separates them; a judgement recorded on the spot cannot.

    So the form asks for two clocks — what the camera's screen said, and the
    reference time at that same moment — and offers no cell in which to write a
    correction. See the 2026-08-14 decision (no DST correction, ever).

THE VISIT TIME IS THE REFERENCE CLOCK

    `visit_date` + `visit_time` are not "roughly when we were there". They are the
    reference reading of the clock pair, which is why `visit_time` is obligatory
    here when 27 of 27 otoño 2026 opening visits recorded no time at all and decayed
    into APPROXIMATE `visit_date_only` anchors.

WHY LAT/LON ARE BOUNDED AND NOT MERELY "DECIMAL"

    `field_notes.csv` stores `lat 39.45183, lon 71.72707` — unsigned, which places
    Bosque Pehuén in China. A range check at data entry is the only place that error
    is cheap to catch; by ingest time the sign is unrecoverable without a map.

    The same bounds do a second job in `read_coordinate`. CT26 sat at 39.25447 /
    71.44562 — 19 km outside the reserve — because a cell holding 39°25'44.7" was
    typed as though it were decimal. That was diagnosed and repaired in the platform
    on 2026-04-15, but `build_field_notes.py` was written in August and re-sourced
    the same bad cell, so the repair never reached a consumer that did not yet exist.

    Hence the rule rather than the point fix: a coordinate outside the plausible
    range that BECOMES plausible when re-read as degrees-minutes-seconds is a
    DMS-encoded cell. It catches the same mistake at any station, and it lives beside
    the bounds it depends on so the two cannot drift apart.

WHY DATE CELLS ARE TEXT

    The legacy workbook held three date conventions at once, and the dangerous ones
    were the cells Excel had already parsed using the machine locale — a wrong
    reading that looks clean. Text-formatted cells cannot be silently reinterpreted,
    so ISO 8601 typed in Chile stays ISO 8601 when opened anywhere else.
"""

from __future__ import annotations

from dataclasses import dataclass

from camtrap import stations

# The workbook is filled by Spanish speakers; the CSV is read by code that already
# speaks `yes`/`no`. This is the only place the two vocabularies meet.
CSV_BOOLEAN = {'si': 'yes', 'no': 'no', 'no se sabe': 'unknown'}

SI_NO = ('si', 'no')
SI_NO_NS = ('si', 'no', 'no se sabe')
VISIT_TYPES = ('instalacion', 'revision', 'retiro', 'mantencion')

# Bumped when a column is added, removed or redefined. Stamped on the glossary so a
# filled workbook can be dated without opening a diff.
SCHEMA_VERSION = '1.0'

# A campaign is named for the season its cards are RETRIEVED in, so the dropdown is
# generated from that rule rather than enumerated — an enumeration would have to be
# edited every year, and the year it is forgotten is the year someone free-types.
CAMPAIGN_SEASONS = ('otono', 'primavera')
CAMPAIGN_YEARS = range(2025, 2036)
CAMPAIGNS = tuple(f'{season}_{year}'
                  for year in CAMPAIGN_YEARS for season in CAMPAIGN_SEASONS)

# The reserve, not the country. Derived from `plataforma-territorial/data/`: the
# boundary spans lat −39.46463..−39.42249 and lon −71.76328..−71.72494, and all 27
# stations sit inside it; MARGIN adds ~5.5 km on every side so a genuine new site or
# a re-survey passes without an edit here.
#
# A country-wide box would be useless to `read_coordinate`: with lat −56..−17 BOTH
# readings of 39.25447 land in range and the DMS test can decide nothing. The rule
# works only at reserve scale, which is why the bound is this tight and why widening
# it is a decision with a consequence rather than a matter of taste.
COORD_MARGIN_DEG = 0.05
LAT_BOUNDS = (-39.51, -39.37)
LON_BOUNDS = (-71.81, -71.67)

# The stations that actually exist, from the registry — so adding a site is an edit
# to `estaciones.csv` and nothing else, and the dropdown can never offer a station
# the reference sheet cannot describe. `canonical_id` owns the CT%02d spelling, so
# the fallback restates nothing.
STATION_IDS = (tuple(stations.registry())
               or tuple(stations.canonical_id(n) for n in range(1, 31)))

REQ_ALWAYS = 'siempre'
REQ_IF_MOVED = 'sólo si "¿Se movió o reinstaló?" = si, o si es instalación'
REQ_IF_ADJUSTED = 'sólo si "¿Se ajustó el reloj?" = si'
REQ_OPTIONAL = 'opcional'

# Formats are stated as the technician must type them, not as a strftime pattern.
FMT_DATE = 'AAAA-MM-DD'
FMT_TIME = 'HH:MM (24 h)'
FMT_DATETIME = 'AAAA-MM-DD HH:MM (24 h)'


@dataclass(frozen=True)
class VisitField:
    """One column of the visit form.

    `column` is the name the value carries once it reaches `field_notes.csv`;
    `label` is what the person filling the sheet actually reads. Everything else
    is what the workbook needs in order to refuse a bad answer at the moment it is
    typed, rather than three months later at ingest.
    """

    column: str
    label: str
    fmt: str
    why: str
    options: tuple[str, ...] = ()
    required: str = REQ_ALWAYS
    example: str = ''
    width: int = 18
    # (min, max) for numeric entry. The bound is the check; None means free.
    bounds: tuple[float, float] | None = None
    is_text: bool = False        # force cell to text so Excel cannot reparse it
    length: int | None = None    # exact character count, for the text date formats
    prefix: str = ''             # required leading string, e.g. CAM-

    @property
    def has_list(self) -> bool:
        return bool(self.options)


VISIT_FIELDS: tuple[VisitField, ...] = (
    VisitField(
        column='station_id',
        label='Estación',
        fmt='CT01 … CT30 (lista)',
        options=STATION_IDS,
        example='CT18',
        width=11,
        why='Identifica el SITIO, no el equipo. Es la unidad espacial de todo '
            'análisis (esfuerzo, ocupancia, tasa de detección) y debe coincidir '
            'con el nombre de la carpeta de imágenes que lee Timelapse2.',
    ),
    VisitField(
        column='visit_date',
        label='Fecha de la visita',
        fmt=FMT_DATE,
        example='2026-05-15',
        width=15,
        is_text=True,
        length=10,
        why='Fecha del reloj de referencia (teléfono/GPS), no la de la cámara. '
            'Junto con la hora define el extremo de la ventana de despliegue: una '
            'foto anterior a la instalación o posterior al retiro es imposible, y '
            'así se detecta un reloj adelantado.',
    ),
    VisitField(
        column='visit_time',
        label='Hora de la visita (reloj de referencia)',
        fmt=FMT_TIME,
        example='12:10',
        width=16,
        is_text=True,
        length=5,
        why='OBLIGATORIA. Es la lectura del reloj de referencia en el mismo '
            'instante en que se mira la pantalla de la cámara. Sin ella el ancla '
            'queda APROXIMADA y la hora del día deja de ser utilizable: las 27 '
            'visitas de apertura de otoño 2026 se perdieron exactamente así.',
    ),
    VisitField(
        column='visit_type',
        label='Tipo de visita',
        fmt='lista',
        options=VISIT_TYPES,
        example='revision',
        width=14,
        why='instalacion = se deja una cámara donde no había. revision = se cambia '
            'la tarjeta (cierra una campaña y abre la siguiente). mantencion = se '
            'visita sin cambiar tarjeta. retiro = se levanta la cámara y el sitio '
            'queda sin equipo.',
    ),
    VisitField(
        column='campaign_opened',
        label='Campaña que se ABRE (tarjetas que quedan puestas)',
        fmt='lista',
        options=CAMPAIGNS,
        example='primavera_2026',
        width=20,
        why='La campaña se nombra por la temporada en que se RETIRARÁN las tarjetas '
            'que se están dejando ahora. NO se anota la campaña que se cierra: es '
            'siempre la que abrió la visita anterior a esa misma estación, así que '
            'se deduce sola y nunca puede contradecir al registro.',
    ),
    VisitField(
        column='observers',
        label='Observadores',
        fmt='iniciales separadas por coma',
        example='TA, SC',
        width=14,
        why='Quién estuvo en terreno. Permite volver a preguntar cuando un dato no '
            'cuadra, que es la única forma de resolver una ambigüedad de terreno.',
    ),
    VisitField(
        column='camera_unit_id',
        label='ID físico de la cámara',
        fmt='CAM-<número grabado en el equipo>',
        example='CAM-28',
        width=16,
        prefix='CAM-',
        why='Identifica el EQUIPO, no el sitio. El reloj y la sensibilidad del PIR '
            'pertenecen al cuerpo de la cámara y viajan con él. Se anota en TODA '
            'visita, no sólo al cambiarla: un equipo cambiado sin registro es '
            'invisible. El prefijo CAM- evita la colisión real de mayo 2026, '
            'cuando la estación CT23 recibió el equipo 18 y la CT18 el equipo 28.',
    ),
    VisitField(
        column='camera_working',
        label='¿Funcionaba al llegar?',
        fmt='lista',
        options=SI_NO_NS,
        example='no',
        width=14,
        why='Una cámara muerta deja de muestrear sin avisar: la CT19 estuvo 91 días '
            'apagada antes del retiro. Es lo que separa "no pasó ningún animal" de '
            '"no había cámara", y por tanto el denominador del esfuerzo.',
    ),
    VisitField(
        column='camera_datetime_observed',
        label='Fecha y hora EN LA PANTALLA de la cámara',
        fmt=FMT_DATETIME,
        example='2026-05-15 13:10',
        width=24,
        is_text=True,
        length=16,
        required=REQ_ALWAYS,
        why='LECTURA CRUDA, copiada tal cual de la pantalla, aunque parezca '
            'absurda (la CT18 marcaba 2017). Junto con la hora de la visita da el '
            'desfase exacto del equipo. Es el único dato que distingue un reloj que '
            'se reseteó de uno que siempre estuvo corrido. Si la cámara no '
            'enciende, dejar en blanco y marcar "¿Funcionaba al llegar?" = no.',
    ),
    VisitField(
        column='clock_adjusted',
        label='¿Se ajustó el reloj de la cámara?',
        fmt='lista',
        options=SI_NO,
        example='no',
        width=16,
        why='La política es NO ajustar: un equipo con desfase fijo y conocido es '
            'reparable, uno reajustado en cada visita no. Si aun así hubo que '
            'ajustarlo, se anota aquí y se registra la pantalla después del ajuste.',
    ),
    VisitField(
        column='camera_datetime_after',
        label='Pantalla DESPUÉS del ajuste',
        fmt=FMT_DATETIME,
        required=REQ_IF_ADJUSTED,
        example='',
        width=22,
        is_text=True,
        length=16,
        why='Sin esta lectura, un ajuste deja el desfase desconocido desde ese '
            'instante y todas las fotos siguientes quedan sin reparación posible.',
    ),
    VisitField(
        column='card_changed',
        label='¿Se cambió la tarjeta?',
        fmt='lista',
        options=SI_NO,
        example='si',
        width=14,
        why='El cambio de tarjeta es lo que cierra una campaña y abre la siguiente. '
            'El nombre de la tarjeta no se pide: no se usa en ningún análisis.',
    ),
    VisitField(
        column='batteries_changed',
        label='¿Se cambiaron las pilas?',
        fmt='lista',
        options=SI_NO,
        example='si',
        width=14,
        why='Las pilas agotadas son la causa habitual de que una cámara deje de '
            'disparar a mitad de despliegue; explica los vacíos de esfuerzo.',
    ),
    VisitField(
        column='moved',
        label='¿Se movió o reinstaló?',
        fmt='lista',
        options=SI_NO,
        example='no',
        width=14,
        why='Marca el único caso en que hay que volver a medir la posición. Con '
            '"no", las cinco columnas siguientes se dejan en blanco y valen las de '
            'la visita anterior.',
    ),
    VisitField(
        column='lat',
        label='Latitud (WGS84)',
        fmt='decimal CON SIGNO, 5 decimales',
        required=REQ_IF_MOVED,
        example='-39.45183',
        width=14,
        bounds=LAT_BOUNDS,
        why='CON SIGNO NEGATIVO: estamos en el hemisferio sur. El registro '
            'histórico guarda 39.45183 sin signo, que cae en China. Sólo se aceptan '
            'coordenadas dentro de Bosque Pehuén y sus alrededores (−39,51 a '
            '−39,37); si la planilla la rechaza, revisar el signo y que el GPS esté '
            'en grados decimales y no en grados-minutos-segundos.',
    ),
    VisitField(
        column='lon',
        label='Longitud (WGS84)',
        fmt='decimal CON SIGNO, 5 decimales',
        required=REQ_IF_MOVED,
        example='-71.72707',
        width=14,
        bounds=LON_BOUNDS,
        why='CON SIGNO NEGATIVO: estamos al oeste de Greenwich. Rango aceptado '
            '−71,81 a −71,67. Grados decimales, NUNCA grados-minutos-segundos: la '
            'CT26 quedó 19 km fuera de la reserva durante un año porque se copió '
            '39°25\'44,7" como si fuera 39,25447.',
    ),
    VisitField(
        column='height_m',
        label='Altura del montaje (m)',
        fmt='decimal, metros desde el suelo',
        required=REQ_IF_MOVED,
        example='1.5',
        width=14,
        bounds=(0.0, 10.0),
        why='La altura cambia qué especies entran en el campo de visión: una cámara '
            'a 1,5 m deja de detectar micromamíferos que una a 0,4 m sí registra. '
            'En mayo 2026 se subieron todas las cámaras a 1,5–2 m y no se anotó la '
            'altura por estación, así que ese cambio no es corregible.',
    ),
    VisitField(
        column='bearing_deg',
        label='Azimut al que apunta (°)',
        fmt='entero 0–359, norte = 0',
        required=REQ_IF_MOVED,
        example='210',
        width=14,
        bounds=(0, 359),
        why='Dirección de la brújula del teléfono. Define hacia dónde mira el área '
            'detectada y explica los falsos disparos por sol directo (este/oeste). '
            'No se puede reconstruir después desde la foto.',
    ),
    VisitField(
        column='detection_distance_m',
        label='Distancia al paso objetivo (m)',
        fmt='decimal, metros',
        required=REQ_IF_MOVED,
        example='4.0',
        width=16,
        bounds=(0.0, 100.0),
        why='Distancia desde la cámara hasta el sendero, huella o paso que se está '
            'vigilando. Es el tamaño efectivo del área muestreada: sin ella las '
            'tasas de detección de dos estaciones no son comparables. Tampoco se '
            'puede reconstruir después.',
    ),
    VisitField(
        column='notes',
        label='Observaciones',
        fmt='texto libre',
        required=REQ_OPTIONAL,
        example='No prendía, se cambió la CT a otra',
        width=46,
        why='Todo lo que no cabe en una columna: humedad, daño, animal a la vista, '
            'vegetación que tapa el lente. Se lee, pero no se analiza — si un dato '
            'importa para el análisis, pedir una columna en vez de escribirlo aquí.',
    ),
)


# The standing facts about each site, shown on the `Estaciones` sheet so nobody ever
# retypes a coordinate into a visit row. `camtrap/stations.py` said this registry
# "belongs in a station registry going forward"; `data/campaigns/estaciones.csv` is it.
STATION_FIELDS: tuple[VisitField, ...] = (
    VisitField(
        column='station_id', label='Estación', fmt='CT01 … CT30',
        example='CT18', width=11,
        why='El sitio. Nunca cambia.',
    ),
    VisitField(
        column='grid_id', label='Grilla de monitoreo', fmt='número', example='33',
        width=13,
        why='La celda de la grilla de monitoreo en que cae la estación. Es un dato '
            'del sitio, NO otro nombre para la estación: se anota aquí una vez y no '
            'se repite en cada visita.',
    ),
    VisitField(
        column='lat', label='Latitud (WGS84)', fmt='decimal con signo',
        example='-39.43320', width=13, bounds=LAT_BOUNDS,
        why='Posición vigente del sitio. Sólo cambia si la cámara se reinstala.',
    ),
    VisitField(
        column='lon', label='Longitud (WGS84)', fmt='decimal con signo',
        example='-71.74338', width=13, bounds=LON_BOUNDS,
        why='Posición vigente del sitio.',
    ),
    VisitField(
        column='elevation_m', label='Altitud (m s.n.m.)', fmt='entero',
        example='978', width=12, required=REQ_OPTIONAL,
        why='Del modelo de elevación, no del GPS de mano.',
    ),
    VisitField(
        column='camera_unit_id', label='Último equipo registrado',
        fmt='CAM-<número>', example='CAM-28', width=15, required=REQ_OPTIONAL,
        why='Qué cámara quedó instalada la última vez. Casi todo en blanco: sólo 2 '
            'de 106 visitas históricas anotaron el equipo. Se completa solo a '
            'medida que se llenen las visitas nuevas.',
    ),
    VisitField(
        column='height_m', label='Última altura (m)', fmt='decimal',
        example='', width=13, required=REQ_OPTIONAL, bounds=(0.0, 10.0),
        why='En blanco: en mayo 2026 se subieron todas las cámaras a 1,5–2 m y no '
            'se anotó la altura por estación. Se establece en la próxima salida.',
    ),
    VisitField(
        column='bearing_deg', label='Último azimut (°)', fmt='entero 0–359',
        example='', width=13, required=REQ_OPTIONAL, bounds=(0, 359),
        why='En blanco: nunca se ha registrado. Se establece en la próxima salida.',
    ),
    VisitField(
        column='detection_distance_m', label='Última distancia al paso (m)',
        fmt='decimal', example='', width=15, required=REQ_OPTIONAL,
        bounds=(0.0, 100.0),
        why='En blanco: nunca se ha registrado. Se establece en la próxima salida.',
    ),
    VisitField(
        column='notes', label='Observaciones del sitio', fmt='texto libre',
        example='', width=52, required=REQ_OPTIONAL,
        why='Acceso, referencias para llegar, advertencias.',
    ),
)


def read_coordinate(raw, kind: str) -> tuple[float | None, str]:
    """Read one coordinate cell as signed decimal degrees. Returns (value, flag).

    Bosque Pehuén is south and west of the origin, so a magnitude is all the sheet
    ever carries usefully and the sign is restored here rather than trusted.

    Three outcomes, and the third is a refusal on purpose:

      in range as decimal      -> that value, no flag
      in range only as D°M'S"  -> converted, flagged `coord_dms_as_decimal`
      in range as neither      -> None, flagged `coord_out_of_range`

    A value plausible BOTH ways would be flagged rather than silently picked, which
    is the same discipline `build_field_notes.parse_visit_date` applies to dates. In
    practice the bounds are narrow enough that it cannot happen — 39.44 read as DMS
    is 39.73, already outside — and the check exists so that widening the bounds
    later cannot quietly introduce an ambiguity.
    """
    lo, hi = LAT_BOUNDS if kind == 'lat' else LON_BOUNDS

    if raw is None or (isinstance(raw, str) and not raw.strip()):
        return None, ''
    try:
        magnitude = abs(float(raw))
    except (TypeError, ValueError):
        return None, f'coord_unparseable: {raw!r}'

    as_decimal = -magnitude
    dms = _dms_to_decimal(magnitude)
    as_dms = None if dms is None else -dms

    decimal_ok = lo <= as_decimal <= hi
    dms_ok = as_dms is not None and lo <= as_dms <= hi

    if decimal_ok and dms_ok:
        return as_decimal, (f'coord_ambiguous: {magnitude} is plausible as decimal '
                            f'({as_decimal:.5f}) and as DMS ({as_dms:.5f}) — decimal '
                            f'assumed, VERIFY')
    if decimal_ok:
        return as_decimal, ''
    if dms_ok:
        return as_dms, (f'coord_dms_as_decimal: {magnitude} is out of range as '
                        f'decimal but is {as_dms:.5f} read as degrees-minutes-'
                        f'seconds; converted')
    return None, (f'coord_out_of_range: {magnitude} is outside {kind} bounds '
                  f'{lo}..{hi} as decimal and as DMS; refused rather than guessed')


def _dms_to_decimal(magnitude: float) -> float | None:
    """Re-read `39.25447` as 39°25'44.7", or None when the digits cannot be minutes
    and seconds at all.

    None, not the magnitude unchanged: 20 of the 52 historical coordinates have a
    seconds field ≥ 60 (39.43796 → 43′ 79.6″), and returning the input for those made
    "there is no DMS reading" indistinguishable from "the DMS reading agrees with the
    decimal one", so every one of them was flagged ambiguous.
    """
    frac = f'{magnitude:.6f}'.split('.')[1]
    minutes, seconds = int(frac[:2]), float(f'{frac[2:4]}.{frac[4:]}')
    if minutes >= 60 or seconds >= 60:
        return None
    return int(magnitude) + minutes / 60 + seconds / 3600


_BY_LABEL = {f.label: f for f in VISIT_FIELDS}


def by_label(label: str) -> VisitField:
    """Resolve a workbook header back to its field.

    Raises `KeyError` naming the offending header. A renamed column must fail
    loudly: silently dropping an unrecognised one is how 252 rows of camera 5 went
    missing from the 2025 annual report for a year.
    """
    try:
        return _BY_LABEL[label.strip()]
    except KeyError:
        raise KeyError(
            f'encabezado desconocido en la planilla: {label.strip()!r}. '
            f'Los encabezados los genera setup/build_visit_template.py y no deben '
            f'editarse a mano.'
        ) from None
