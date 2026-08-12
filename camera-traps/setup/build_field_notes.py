"""Convert the legacy monitoring workbook into `data/campaigns/field_notes.csv`.

WHAT THIS OWNS
    How the four heterogeneous sheets of `Registro de monitoreo CT.xlsx` map onto
    one visit record, and which of their date readings are corrupt. This is a
    ONE-TIME migration, committed for provenance rather than for reuse: once
    field_notes.csv exists it is the canonical record and the workbook is legacy.
    Anchors derive from the CSV, never from the workbook.

WHY A SCRIPT AND NOT A HAND-WRITTEN CSV
    Over 100 rows, and every correction below has to be auditable. Each one is
    recorded in the row's `data_flags` column, so a reader of the CSV can see what
    was inferred without coming back here.

THE VISIT MODEL
    A visit is a physical event, not a property of a campaign. At Bosque Pehuén
    every revision swaps the card, so one visit CLOSES one campaign and OPENS the
    next; `campaign_closed` / `campaign_opened` say which. An install closes
    nothing. This avoids duplicating each visit as two rows.

        installs (Oct 2024 – Feb 2025)  → opens  otono_2025
        Otoño 2025 revision             → closes otono_2025,     opens primavera_2025
        Primavera revision              → closes primavera_2025, opens otono_2026
        Mayo 2026 revision              → closes otono_2026,     opens (next)

    `pv_2025_2026` is NOT a separate deployment — it is a re-review of the same
    cards as primavera_2025 (2026-07-30 session), so it shares those visits and
    gets no rows of its own.

DATES
    Output is ISO 8601 only. The workbook holds three conventions at once:
    Chilean d/m/y typed as text, m/d/y from camera screens, and cells Excel already
    parsed into real datetimes using the machine locale — those last are the
    dangerous ones, because a wrong reading looks clean. They are detected by
    plausibility against the visit window and the day/month swapped back; every
    swap is flagged, and a value plausible BOTH ways is flagged rather than
    silently picked.

CLOCK STATE IS `unknown` UNLESS THE SHEET SAYS OTHERWISE
    A visit with no remark is not evidence the clock was fine — nobody was asked to
    record it. Defaulting to `ok` would manufacture the very assurance the repair
    rule is supposed to test for.
"""

from __future__ import annotations

import argparse
import re
import sys
from datetime import date, datetime, time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from camtrap import stations  # noqa: E402

WORKBOOK = Path('data/campaigns/Registro de monitoreo CT.xlsx')
OUT_CSV  = Path('data/campaigns/field_notes.csv')

COLUMNS = [
    'campaign_closed', 'campaign_opened', 'station_id', 'visit_type',
    'visit_date', 'visit_time',
    'camera_unit_id', 'camera_replaced',
    'clock_state', 'camera_datetime_observed', 'clock_action', 'clock_offset_hours',
    'sd_out', 'sd_in', 'card_changed', 'batteries_changed',
    'moved', 'lat', 'lon', 'elevation_m', 'height_m',
    'grid_id', 'waypoint', 'gps_device', 'observers',
    'source_sheet', 'notes', 'data_flags',
]

# Plausible span for each sheet's visits, used to catch a locale-mangled datetime.
WINDOWS = {
    'install':   (date(2024, 9, 1),  date(2025, 3, 1)),
    'otono2025': (date(2025, 4, 1),  date(2025, 7, 1)),
    'primavera': (date(2025, 11, 1), date(2026, 2, 1)),
    'mayo2026':  (date(2026, 5, 1),  date(2026, 6, 1)),
}

# Corrections Felipe confirmed 2026-08-11. Keyed (sheet, station_id).
CONFIRMED_FIXES = {
    ('install', 'CT08'): (
        date(2024, 10, 9),
        'install date corrected 2024-09-09 -> 2024-10-09 (confirmed by Felipe '
        '2026-08-11): the otono_2025 sheet records "las primeras fotos dicen '
        'septiembre, debe decir octubre", and CT03/CT04/CT06/CT09/CT10 were all '
        'installed 2024-10-09; the sheet had transcribed the camera clock'
    ),
    ('otono2025', 'CT17'): (
        date(2025, 6, 6),
        'revision date corrected 2025-04-06 -> 2025-06-06 (confirmed by Felipe '
        '2026-08-11): every other otono_2025 revision falls 14 May - 11 Jun 2025'
    ),
}

# Chile left summer time on this date; the array was not adjusted until the Mayo
# 2026 visit, so every frame in between reads one hour ahead of local time.
DST_END_2026 = date(2026, 4, 4)


def parse_visit_date(value, window_key: str) -> tuple[date | None, list[str]]:
    """Return (ISO date, flags). Never guesses silently."""
    lo, hi = WINDOWS[window_key]
    flags: list[str] = []
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None, ['date_missing']

    if isinstance(value, (datetime, pd.Timestamp)):
        d = value.date()
        swapped = None
        if d.day <= 12:
            try:
                swapped = date(d.year, d.day, d.month)
            except ValueError:
                swapped = None
        d_ok = lo <= d <= hi
        s_ok = swapped is not None and lo <= swapped <= hi
        if d_ok and s_ok and swapped != d:
            flags.append(
                f'date_ambiguous: Excel stored {d.isoformat()}; day/month swap '
                f'{swapped.isoformat()} is equally plausible — VERIFY'
            )
            return d, flags
        if s_ok and not d_ok:
            flags.append(
                f'date_swapped: Excel stored {d.isoformat()} (locale m/d/y misparse), '
                f'read as {swapped.isoformat()}'
            )
            return swapped, flags
        if not d_ok:
            flags.append(f'date_out_of_window: {d.isoformat()} not within {lo}..{hi}')
        return d, flags

    text = str(value).strip()
    m = re.match(r'^(\d{1,2})[/-](\d{1,2})[/-](\d{4})$', text)
    if not m:
        return None, [f'date_unparsed: {text!r}']
    a, b, year = int(m.group(1)), int(m.group(2)), int(m.group(3))
    # Text entered by the Chilean field team: d/m/y.
    try:
        d = date(year, b, a)
    except ValueError:
        return None, [f'date_invalid: {text!r}']
    if not (lo <= d <= hi):
        flags.append(f'date_out_of_window: {d.isoformat()} not within {lo}..{hi}')
    elif a <= 12 and b <= 12 and a != b:
        flags.append(f'date_ambiguous_source: {text!r} read d/m/y as {d.isoformat()}')
    return d, flags


def parse_time(value) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ''
    if isinstance(value, time):
        return value.strftime('%H:%M:%S')
    if isinstance(value, (datetime, pd.Timestamp)):
        return value.strftime('%H:%M:%S')
    text = str(value).strip()
    return text if re.match(r'^\d{1,2}:\d{2}', text) else ''


def read_notes(value) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ''
    return ' | '.join(s.strip(' -') for s in str(value).split('\n') if s.strip())


def clock_from_notes(notes: str) -> tuple[str, list[str]]:
    """The sheet only ever records a clock as BROKEN. Silence means unknown."""
    low = notes.lower()
    if 'desconfigurada' in low or 'error en la fecha' in low:
        return 'wrong', ['clock_state from field remark']
    return 'unknown', []


def replacement_from_notes(notes: str) -> tuple[str, str, list[str]]:
    """(camera_replaced, camera_unit_id, flags) — a new body is a new clock."""
    low = notes.lower()
    replaced = bool(re.search(r'otra c[áa]mara|otra ct|nueva tc|se cambi[óo] la ct', low))
    if not replaced:
        return '', '', []
    unit = re.search(r'\bid\s*(\d+)', low)
    flags = ['camera_replaced from field remark — clock chronology must break here']
    return 'yes', (unit.group(1) if unit else ''), flags


def height_from_notes(notes: str) -> tuple[str, list[str]]:
    m = re.search(r'(?:se subi[óo]|se instal[óo]).{0,30}?(\d[.,]?\d*)\s*m\b', notes.lower())
    if not m:
        return '', []
    return m.group(1).replace(',', '.'), ['height_m approximate, from field remark']


def yesno(value) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ''
    text = str(value).strip().lower()
    return 'yes' if text in {'x', 'si', 'sí', 'yes'} else ('no' if text == 'no' else text)


def clean(value) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ''
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value).strip()


def station_of(tc) -> str | None:
    """The TC column is int on one sheet, float on another, '22*' on a third."""
    if tc is None or (isinstance(tc, float) and pd.isna(tc)):
        return None
    if isinstance(tc, float):
        return stations.canonical_id(int(tc))
    m = re.match(r'\s*(\d+)', str(tc))
    return stations.canonical_id(int(m.group(1))) if m else None


def row(**kw) -> dict:
    base = {c: '' for c in COLUMNS}
    flags = kw.pop('data_flags', [])
    base.update(kw)
    base['data_flags'] = '; '.join(flags)
    return base


def install_dates_from_revisions(book: pd.ExcelFile) -> dict[str, date]:
    """`Fecha de instalación` as it appears on the revision sheets — TEXT, so it
    parses unambiguously as Chilean d/m/y and can settle the install sheet's own
    date cells, which Excel stored as datetimes and are therefore ambiguous."""
    found: dict[str, date] = {}
    for sheet in ('Registro de revisión_Otoño 2025', 'Registro de revisión Mayo 2026'):
        df = book.parse(sheet, header=2)
        if 'Fecha de instalación' not in df.columns:
            continue
        for _, r in df.iterrows():
            sid = station_of(r.get('TC'))
            value = r.get('Fecha de instalación')
            if not sid or sid in found or not isinstance(value, str):
                continue
            m = re.match(r'^(\d{1,2})[/-](\d{1,2})[/-](\d{4})$', value.strip())
            if m:
                try:
                    found[sid] = date(int(m.group(3)), int(m.group(2)), int(m.group(1)))
                except ValueError:
                    pass
    return found


def build_installs(book: pd.ExcelFile) -> list[dict]:
    df = book.parse('Registro de instalacion', header=0)
    corroborating = install_dates_from_revisions(book)
    out = []
    for _, r in df.iterrows():
        sid = station_of(r['N° de Cámara Trampa'])
        if not sid:
            continue
        d, flags = parse_visit_date(r['Fecha'], 'install')
        # An ambiguous datetime that the revision sheets' text column agrees with is
        # not really ambiguous — two independent records read the same way.
        other = corroborating.get(sid)
        if other and any(f.startswith('date_ambiguous') for f in flags):
            if other == d:
                flags = [f for f in flags if not f.startswith('date_ambiguous')]
            else:
                flags = [f'date_conflict: install sheet {d.isoformat() if d else "?"} '
                         f'vs revision sheets {other.isoformat()} — VERIFY']
        fix = CONFIRMED_FIXES.get(('install', sid))
        if fix:
            d, note = fix
            flags = [note]
        notes = read_notes(r.get('Notas'))
        clock, cf = clock_from_notes(notes)
        out.append(row(
            campaign_opened='otono_2025', station_id=sid, visit_type='install',
            visit_date=d.isoformat() if d else '', visit_time=parse_time(r['Hora']),
            clock_state=clock,
            sd_in=clean(r.get('N° de tarjeta de memoria')),
            grid_id=clean(r.get('N° de grilla de monitoreo')),
            waypoint=clean(r.get('ID del waypoint en el GPS')),
            gps_device=clean(r.get('GPS empleado')),
            observers=clean(r.get('Observadores')),
            source_sheet='Registro de instalacion', notes=notes,
            data_flags=flags + cf,
        ))
    return out


def build_revision(book: pd.ExcelFile, sheet: str, key: str,
                   closed: str, opened: str) -> list[dict]:
    df = book.parse(sheet, header=2)
    out = []
    for _, r in df.iterrows():
        sid = station_of(r.get('TC'))
        if not sid:
            continue
        d, flags = parse_visit_date(r.get('Fecha de revisión'), key)
        fix = CONFIRMED_FIXES.get((key, sid))
        if fix:
            d, note = fix
            flags = [note]
        notes = read_notes(r.get('Observaciones'))
        clock, cf = clock_from_notes(notes)
        replaced, unit, rf = replacement_from_notes(notes)
        height, hf = height_from_notes(notes)
        moved = 'yes' if re.search(r'se movi[óo]|se sac[óo]|reinstal', notes.lower()) else ''

        action, offset, af = '', '', []
        if key == 'mayo2026':
            action, offset = 'shifted', '-1'
            af = [
                'clock set back 1 h at this visit for horario de invierno (sheet note, '
                'applies to every station). Chile left summer time ' +
                DST_END_2026.isoformat() + ', so frames between that date and this '
                'visit read 1 h AHEAD of local time'
            ]
            if not height:
                height, hf = '', hf + [
                    'all cameras raised to 1.5-2 m at this visit; exact height per '
                    'station not recorded'
                ]

        out.append(row(
            campaign_closed=closed, campaign_opened=opened, station_id=sid,
            visit_type='revision',
            visit_date=d.isoformat() if d else '',
            visit_time=parse_time(r.get('Hora de revisión')),
            camera_unit_id=unit, camera_replaced=replaced,
            clock_state=clock, clock_action=action, clock_offset_hours=offset,
            sd_out=clean(r.get('SD') if key == 'mayo2026' else r.get('ID SD anterior')),
            sd_in=clean(r.get('ID SD nueva')),
            card_changed=yesno(r.get('Cambio de tarjeta')),
            batteries_changed=yesno(r.get('Cambio de pilas')),
            moved=moved,
            lat=clean(r.get('S')), lon=clean(r.get('W')),
            elevation_m=clean(r.get('Altitud')), height_m=height,
            grid_id=clean(r.get('Grilla') if 'Grilla' in df.columns else r.get('N Grilla')),
            waypoint=clean(r.get('Waypoint')), gps_device=clean(r.get('GPS Empleado')),
            source_sheet=sheet, notes=notes,
            data_flags=flags + cf + rf + hf + af,
        ))
    return out


def add_missing_station_gaps(rows: list[dict]) -> list[dict]:
    """A station with data but no install row needs a placeholder that says so.

    CT27 is the live case: it appears only on the Primavera sheet, yet it has 21
    frames in otoño 2026. Felipe confirmed 2026-08-11 that the station is real and
    must not be dropped, so the gap is recorded rather than left to be rediscovered.
    """
    installed = {r['station_id'] for r in rows if r['visit_type'] == 'install'}
    for sid in sorted({r['station_id'] for r in rows} - installed):
        seen = sorted({r['source_sheet'] for r in rows if r['station_id'] == sid})
        rows.append(row(
            station_id=sid, visit_type='unrecorded', clock_state='unknown',
            source_sheet='(absent)',
            notes=f'{sid} has no install record; it appears only on: {", ".join(seen)}.',
            data_flags=[
                'STATION EXISTS AND HAS DATA but its field record is incomplete. '
                'Confirmed by Felipe 2026-08-11: do not delete. Install and retrieval '
                'dates must be reconstructed before this station can carry an anchor.'
            ],
        ))
    return rows


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--workbook', type=Path, default=WORKBOOK)
    ap.add_argument('--out', type=Path, default=OUT_CSV)
    args = ap.parse_args(argv)

    book = pd.ExcelFile(args.workbook)
    rows = build_installs(book)
    rows += build_revision(book, 'Registro de revisión_Otoño 2025', 'otono2025',
                           'otono_2025', 'primavera_2025')
    rows += build_revision(book, 'Registro de revisión_Primavera ', 'primavera',
                           'primavera_2025', 'otono_2026')
    rows += build_revision(book, 'Registro de revisión Mayo 2026', 'mayo2026',
                           'otono_2026', '')
    rows = add_missing_station_gaps(rows)

    df = pd.DataFrame(rows, columns=COLUMNS).sort_values(
        ['station_id', 'visit_date'], kind='stable')
    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False, encoding='utf-8')

    flagged = df[df['data_flags'] != '']
    print(f'{args.out}: {len(df)} visit rows, {df.station_id.nunique()} stations')
    print(f'{len(flagged)} row(s) carry a flag:\n')
    for _, r in flagged.iterrows():
        print(f"  {r.station_id} {r.visit_date or '(no date)':10} {r.source_sheet[:34]:34} "
              f"{r.data_flags[:150]}")
    missing = df[df['visit_date'] == '']
    if len(missing):
        print(f'\n{len(missing)} row(s) have no usable date.')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
