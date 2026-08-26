"""What a filled visit workbook is allowed to become.

The fixture that matters most is `test_a_clock_reading_survives_the_round_trip`:
`camera_datetime_observed` is the column the whole form was redesigned around and it
is 0 / 107 in the legacy record, so nothing until now has ever proven that a filled
value reaches `field_notes.csv` at all.
"""

import sys
import tempfile
import unittest
from datetime import date, datetime, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from openpyxl import Workbook

from camtrap import visit_form, visit_schema
from camtrap.anchors import FieldRecord

LIVE_CSV = (Path(__file__).resolve().parents[1] / 'data' / 'campaigns'
            / 'field_notes.csv')

#: A complete, valid revision. Every test below is this row with one thing changed.
GOOD = {
    'station_id': 'CT01', 'visit_date': '2026-11-20', 'visit_time': '09:40',
    'visit_type': 'revision', 'campaign_opened': 'otono_2027', 'observers': 'FG, SC',
    'camera_unit_id': 'CAM-01', 'camera_working': 'si',
    'camera_datetime_observed': '2026-11-20 08:40', 'clock_adjusted': 'no',
    'camera_datetime_after': '', 'card_changed': 'si', 'batteries_changed': 'si',
    'moved': 'no', 'lat': '', 'lon': '', 'height_m': '', 'bearing_deg': '',
    'detection_distance_m': '', 'notes': 'Sin novedad.',
}


def workbook(*rows, sheet=visit_form.SHEET, labels=None) -> Path:
    """A workbook shaped like the rendered template: labels in row 1, data below."""
    wb = Workbook()
    ws = wb.active
    ws.title = sheet
    ws.append(labels or [f.label for f in visit_schema.VISIT_FIELDS])
    for row in rows:
        ws.append([row.get(f.column, '') for f in visit_schema.VISIT_FIELDS])
    path = Path(tempfile.mkdtemp()) / 'Registro de visitas CT.xlsx'
    wb.save(path)
    return path


def changed(**overrides) -> dict:
    return {**GOOD, **overrides}


class TestTheRecordKeepsTheFormsShape(unittest.TestCase):

    def test_the_live_record_is_exactly_the_form_plus_provenance(self):
        """The reshape's guarantee, asserted rather than remembered. A column added to
        the form and not to the record is the disagreement 1.14 existed to remove."""
        header = LIVE_CSV.read_text(encoding='utf-8').splitlines()[0]
        self.assertEqual(tuple(header.split(',')), visit_form.FIELD_NOTES_COLUMNS)

    def test_provenance_columns_are_not_collected_on_the_form(self):
        """`source_sheet` and `data_flags` are written here, never asked of anybody."""
        collected = {f.column for f in visit_schema.VISIT_FIELDS}
        for column in visit_form.PROVENANCE_COLUMNS:
            self.assertNotIn(column, collected)


class TestReadingAWorkbook(unittest.TestCase):

    def test_a_valid_row_reads(self):
        rows = visit_form.read(workbook(GOOD))
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]['station_id'], 'CT01')
        self.assertEqual(rows[0]['visit_time'], '09:40')

    def test_si_no_become_the_csv_vocabulary(self):
        """The sheet is Spanish; the record already speaks yes/no."""
        row = visit_form.read(workbook(GOOD))[0]
        self.assertEqual(row['card_changed'], 'yes')
        self.assertEqual(row['camera_working'], 'yes')
        self.assertEqual(
            visit_form.read(workbook(changed(camera_working='no se sabe')))[0]
            ['camera_working'], 'unknown')

    def test_blank_rows_are_not_visits(self):
        """A template is 60 empty rows. Reading them as visits would be absurd."""
        self.assertEqual(visit_form.read(workbook({})), [])

    def test_excel_retyped_values_are_normalised(self):
        """Excel silently converts what looks like a date. `str(datetime)` would put
        `2026-11-20 00:00:00` in a date column."""
        row = visit_form.read(workbook(changed(
            visit_date=date(2026, 11, 20), visit_time=time(9, 40),
            camera_datetime_observed=datetime(2026, 11, 20, 8, 40))))[0]
        self.assertEqual(row['visit_date'], '2026-11-20')
        self.assertEqual(row['visit_time'], '09:40')
        self.assertEqual(row['camera_datetime_observed'], '2026-11-20 08:40')

    def test_an_absurd_clock_reading_is_kept(self):
        """CT18's screen said 2017. A raw reading is not validated against reality —
        that is the whole point of collecting it."""
        row = visit_form.read(workbook(changed(
            camera_datetime_observed='2017-01-05 03:00')))[0]
        self.assertEqual(row['camera_datetime_observed'], '2017-01-05 03:00')

    def test_a_renamed_header_is_refused(self):
        """Silently dropping an unrecognised column is how 252 rows went missing."""
        labels = [f.label for f in visit_schema.VISIT_FIELDS]
        labels[0] = 'Estacion'
        with self.assertRaises(visit_form.VisitFormError) as cm:
            visit_form.read(workbook(GOOD, labels=labels))
        self.assertIn('Estacion', str(cm.exception))

    def test_a_missing_sheet_is_refused(self):
        with self.assertRaises(visit_form.VisitFormError):
            visit_form.read(workbook(GOOD, sheet='Hoja1'))

    def test_every_problem_is_reported_at_once(self):
        """The person fixing it has the sheet in front of them and wants the list."""
        with self.assertRaises(visit_form.VisitFormError) as cm:
            visit_form.read(workbook(changed(station_id='', observers='',
                                             visit_type='paseo')))
        self.assertGreaterEqual(len(cm.exception.problems), 3)


class TestTheFormsObligations(unittest.TestCase):

    def test_an_adjusted_clock_must_record_the_screen_after(self):
        """Without it the offset is unknown from that instant and every later frame
        is unrepairable."""
        with self.assertRaises(visit_form.VisitFormError) as cm:
            visit_form.read(workbook(changed(clock_adjusted='si')))
        self.assertIn('camera_datetime_after', str(cm.exception))

    def test_a_dead_camera_may_leave_the_screen_blank(self):
        """The form tells the technician to do exactly this, so demanding the reading
        would demand one that does not exist."""
        row = visit_form.read(workbook(changed(
            camera_working='no', camera_datetime_observed='')))[0]
        self.assertEqual(row['camera_datetime_observed'], '')

    def test_a_working_camera_must_record_the_screen(self):
        with self.assertRaises(visit_form.VisitFormError) as cm:
            visit_form.read(workbook(changed(camera_datetime_observed='')))
        self.assertIn('camera_datetime_observed', str(cm.exception))

    def test_an_installation_must_be_placed(self):
        """A new camera with no coordinates is CT27's 315 coordinateless images."""
        with self.assertRaises(visit_form.VisitFormError) as cm:
            visit_form.read(workbook(changed(visit_type='instalacion')))
        self.assertIn('lat', str(cm.exception))

    def test_a_moved_camera_must_be_placed_again(self):
        with self.assertRaises(visit_form.VisitFormError) as cm:
            visit_form.read(workbook(changed(moved='si')))
        self.assertIn('lon', str(cm.exception))

    def test_a_retiro_leaves_no_campaign_open(self):
        """The vocabulary defines retiro as leaving the site without equipment, so a
        campaign named there contradicts the visit type."""
        with self.assertRaises(visit_form.VisitFormError) as cm:
            visit_form.read(workbook(changed(visit_type='retiro',
                                             campaign_opened='otono_2027')))
        self.assertIn('campaign_opened', str(cm.exception))
        row = visit_form.read(workbook(changed(visit_type='retiro',
                                               campaign_opened='')))[0]
        self.assertEqual(row['campaign_opened'], '')

    def test_an_unsigned_coordinate_is_signed(self):
        """Bosque Pehuen is south and west, so the sheet carries a magnitude and the
        sign is restored rather than trusted -- the CT26 defect, fixed at source."""
        row = visit_form.read(workbook(changed(
            visit_type='instalacion', moved='no', lat='39.44170', lon='71.74200',
            height_m='1.8', bearing_deg='120', detection_distance_m='4')))[0]
        self.assertEqual(row['lat'], '-39.44170')
        self.assertEqual(row['lon'], '-71.74200')

    def test_an_unusable_coordinate_on_an_installation_is_refused(self):
        """An installation without a position is CT27's 315 coordinateless images, so
        an out-of-range value must not pass as 'recorded'."""
        with self.assertRaises(visit_form.VisitFormError) as cm:
            visit_form.read(workbook(changed(
                visit_type='instalacion', lat='12.00000', lon='71.74200',
                height_m='1.8', bearing_deg='120', detection_distance_m='4')))
        self.assertIn('lat', str(cm.exception))

    def test_an_unusable_coordinate_on_a_revision_is_flagged_not_stored(self):
        """Where the position is not required the row still lands, carrying the reason
        it has no coordinate instead of a number nobody can place."""
        row = visit_form.read(workbook(changed(lat='12.00000')))[0]
        self.assertEqual(row['lat'], '')
        self.assertIn('coord', row['data_flags'])

    def test_a_height_outside_its_bounds_is_refused(self):
        with self.assertRaises(visit_form.VisitFormError) as cm:
            visit_form.read(workbook(changed(
                visit_type='instalacion', lat='-39.44170', lon='-71.74200',
                height_m='47', bearing_deg='120', detection_distance_m='4')))
        self.assertIn('height_m', str(cm.exception))


class TestAppending(unittest.TestCase):

    def setUp(self):
        self.csv = Path(tempfile.mkdtemp()) / 'field_notes.csv'

    def test_a_clock_reading_survives_the_round_trip(self):
        """The fixture V2-REVIEW 1.14 asked for: a filled `camera_datetime_observed`
        reaches the record, and the record still parses into visits."""
        self.assertEqual(visit_form.ingest(workbook(GOOD), self.csv), 1)
        text = self.csv.read_text(encoding='utf-8')
        self.assertIn('2026-11-20 08:40', text)
        self.assertEqual(tuple(text.splitlines()[0].split(',')),
                         visit_form.FIELD_NOTES_COLUMNS)
        record = FieldRecord.load(self.csv)
        self.assertEqual(len(record), 1)

    def test_re_ingesting_the_same_workbook_is_refused(self):
        """"Did that run land?" is answered by running it again, not by opening the
        file and counting."""
        book = workbook(GOOD)
        visit_form.ingest(book, self.csv)
        with self.assertRaises(visit_form.VisitFormError):
            visit_form.ingest(book, self.csv)
        self.assertEqual(len(FieldRecord.load(self.csv)), 1)

    def test_the_same_visit_twice_in_one_workbook_is_refused(self):
        with self.assertRaises(visit_form.VisitFormError):
            visit_form.read(workbook(GOOD, GOOD))

    def test_nothing_is_appended_when_any_row_is_bad(self):
        """All or nothing: half a salida in the record is worse than none, because the
        missing half is invisible."""
        with self.assertRaises(visit_form.VisitFormError):
            visit_form.ingest(workbook(GOOD, changed(station_id='CT02',
                                                     observers='')), self.csv)
        self.assertFalse(self.csv.exists())

    def test_a_pre_reshape_record_is_refused_as_a_target(self):
        """Appending form-shaped rows to a 28-column file would produce a record in
        neither shape."""
        self.csv.write_text('campaign_closed,campaign_opened,station_id\n,,CT01\n',
                            encoding='utf-8')
        with self.assertRaises(visit_form.VisitFormError) as cm:
            visit_form.ingest(workbook(GOOD), self.csv)
        self.assertIn('campaign_closed', str(cm.exception))

    def test_appending_never_touches_the_rows_already_there(self):
        first = self.csv
        visit_form.ingest(workbook(GOOD), first)
        before = first.read_text(encoding='utf-8')
        visit_form.ingest(workbook(changed(visit_date='2027-05-14',
                                           campaign_opened='primavera_2027')), first)
        self.assertTrue(first.read_text(encoding='utf-8').startswith(before))


if __name__ == '__main__':
    unittest.main()
