"""
Species Review App — Phase 1 Labeling UI
==========================================
Streamlit app that shows CLIP-classified camera trap images grouped by proposed
species, lets the user confirm or correct each one, and exports a revised CSV.

Usage (from the camera-traps/ directory):
    streamlit run phase1_labeling/app.py -- --campaign-dir "D:/Otono_2026/SynologyDrive"

`--campaign-dir` is required — see resolve_campaign_dir(). The `--` is Streamlit's,
not ours: without it Streamlit eats the argument. CAMERA_TRAPS_CAMPAIGN_DIR works too.

Output:
    <campaign_dir>/new_labeled_data_reviewed.csv

Two modes (sidebar):
    Revisión   — confirm or correct species; "Con persona" / "Con vehículo" under each
                 image record co-presence (camtrap/copresence.py) in the same pass.
    Etiquetas  — only for campaigns reviewed before those checkboxes existed: shows the
                 reviewed images by confirmed species and saves ONLY the tags. It can
                 never change a species, an outcome or a review date.
How decisions reach the file is phase1_labeling/review_record.py.
"""

import csv
import io
import json
import os
import sys
import unicodedata
from collections import defaultdict
from pathlib import Path

import streamlit as st
import yaml
from PIL import Image

# Make the project root importable when launched as `streamlit run phase1_labeling/app.py`
# (Streamlit puts the script's directory on sys.path, not the project root).
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from camtrap import copresence, exports
from classify_campaign.species import clip_species
from phase1_labeling import review_record as rr
from phase1_labeling.review_record import Decision

# ── Config ────────────────────────────────────────────────────────────────────
CONFIG_PATH   = Path(__file__).parent.parent / "config.yaml"
CAMPAIGN_DIR_ENV = "CAMERA_TRAPS_CAMPAIGN_DIR"
COLS_PER_ROW    = 3      # triptychs per row (each cell holds prev/current/next)
THUMB_SIZE      = 1280   # max px on long edge — high enough to stay sharp in Streamlit's expand view
JPEG_QUALITY    = 85
SPECIAL_OPTIONS = ["No es un animal", "No reconocible", "Otro (especificar)"]
MODE_REVIEW, MODE_TAGS = "Revisión", "Etiquetas"

# Every reviewed row here is an animal row, so it is offered exactly the tags the
# convention allows on one. Labels are the only thing this file adds to the vocabulary.
TAG_LABELS = {copresence.TAG_HUMAN: "Con persona", copresence.TAG_VEHICLE: "Con vehículo"}
OFFERED_TAGS = [t for t in copresence.TAGS if t in copresence.allowed_tags(exports.TYPE_ANIMAL)]


# ── Cached data loaders ───────────────────────────────────────────────────────
# Each public loader stat()s its source file and passes the mtime as an extra
# cache key, so the cached value invalidates when the underlying file changes.

def load_config() -> dict:
    return _load_config_cached(CONFIG_PATH.stat().st_mtime)


@st.cache_data
def _load_config_cached(_mtime: float) -> dict:
    with open(CONFIG_PATH, encoding="utf-8") as f:
        return yaml.safe_load(f)


def resolve_campaign_dir() -> Path:
    """Where this campaign's images live. Required — no default, ever.

    It used to be `campaign_dir` in config.yaml, which meant a committed file decided
    which campaign got reviewed; it stayed pointed at otoño 2025 for three campaigns and
    the path still existed, so a wrong run would have looked entirely normal.

    Two ways in, because `streamlit run` swallows bare arguments:
        streamlit run phase1_labeling/app.py -- --campaign-dir "D:/Otono_2026/SynologyDrive"
        CAMERA_TRAPS_CAMPAIGN_DIR="D:/..." streamlit run phase1_labeling/app.py
    """
    raw = None
    argv = sys.argv[1:]
    for i, a in enumerate(argv):
        if a == "--campaign-dir" and i + 1 < len(argv):
            raw = argv[i + 1]
            break
        if a.startswith("--campaign-dir="):
            raw = a.split("=", 1)[1]
            break
    raw = raw or os.environ.get(CAMPAIGN_DIR_ENV)

    if not raw:
        st.error(
            "**No campaign directory given.** This app will not guess one.\n\n"
            "Relaunch with the campaign's image folder:\n\n"
            "```\nstreamlit run phase1_labeling/app.py -- "
            '--campaign-dir "D:/Otono_2026/SynologyDrive"\n```\n\n'
            f"or set `{CAMPAIGN_DIR_ENV}` in the environment."
        )
        st.stop()

    path = Path(raw)
    if not path.is_dir():
        st.error(f"**Campaign directory does not exist:** `{path}`")
        st.stop()
    return path


def load_classified_csv(path: str) -> tuple[list[str], list[dict]]:
    return _load_classified_csv_cached(path, Path(path).stat().st_mtime)


@st.cache_data
def _load_classified_csv_cached(path: str, _mtime: float) -> tuple[list[str], list[dict]]:
    with open(path, encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = list(reader.fieldnames)
        rows = list(reader)
    return fieldnames, rows


def load_station_index(json_path: str) -> dict[str, list[str]]:
    """Returns {station_relpath_lower: [files sorted alphabetically]}.

    Built from every image in the MD JSON regardless of detections — so the
    chronology includes empty triggers, giving full burst context in the UI.
    Filenames within a deployment sort chronologically (cameras encode the
    capture sequence into the filename).
    """
    return _load_station_index_cached(json_path, Path(json_path).stat().st_mtime)


@st.cache_data
def _load_station_index_cached(json_path: str, _mtime: float) -> dict[str, list[str]]:
    with open(json_path, encoding="utf-8") as f:
        data = json.load(f)
    index: dict[str, list[str]] = defaultdict(list)
    for img in data["images"]:
        fp = img["file"].replace("\\", "/").lower()
        station = fp.rsplit("/", 1)[0] if "/" in fp else ""
        index[station].append(fp)
    for station in index:
        index[station].sort()
    return dict(index)


def neighbors(fp_norm: str, station_index: dict[str, list[str]]) -> tuple[str | None, str | None]:
    """Return (prev, next) relative paths within the same station; None at boundaries."""
    station = fp_norm.rsplit("/", 1)[0] if "/" in fp_norm else ""
    files = station_index.get(station, [])
    try:
        i = files.index(fp_norm)
    except ValueError:
        return None, None
    prev = files[i - 1] if i > 0 else None
    nxt  = files[i + 1] if i < len(files) - 1 else None
    return prev, nxt


@st.cache_data
def load_thumbnail(image_path: str) -> bytes | None:
    """Load full-frame image, resize to THUMB_SIZE and return JPEG bytes. Cached."""
    try:
        img = Image.open(image_path).convert("RGB")
        img.thumbnail((THUMB_SIZE, THUMB_SIZE))
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=JPEG_QUALITY)
        return buf.getvalue()
    except Exception:
        return None


# ── Helpers ───────────────────────────────────────────────────────────────────

def nfc(s: str) -> str:
    """Normalize Unicode to NFC and strip whitespace."""
    return unicodedata.normalize("NFC", s.strip())


def find_default_idx(species: str, options: list[str]) -> int:
    """Return best matching index in options for a species name."""
    if species in options:
        return options.index(species)
    lower = species.lower()
    for i, opt in enumerate(options):
        if opt.lower().startswith(lower) or lower.startswith(opt.lower()):
            return i
    return 0


# ── Navigation helpers ────────────────────────────────────────────────────────

def go_to(idx: int) -> None:
    st.session_state.current_group = idx


def _advance(n_groups: int) -> None:
    next_idx = st.session_state.current_group + 1
    st.session_state.current_group = min(next_idx, n_groups)


# ── Co-presence checkboxes ────────────────────────────────────────────────────
# One widget per (tag, image). Both modes share the keys, so a tick survives a
# mode switch within the session.

def _tag_widget(tag: str, key: str) -> str:
    return f"tag|{tag}|{key}"


def seed_ticks(key: str, tags: str) -> None:
    """Start an image's checkboxes from its saved decision, unless they already hold a
    value. Called on every render, not once: Streamlit drops the state of widgets
    that are not on screen, and an unseeded box would come back unticked."""
    held = copresence.parse(tags)
    for tag in OFFERED_TAGS:
        st.session_state.setdefault(_tag_widget(tag, key), tag in held)


def ticked(key: str) -> str:
    """The observationTags value for an image's checkboxes as they stand."""
    return copresence.join(t for t in OFFERED_TAGS if st.session_state.get(_tag_widget(t, key)))


def render_ticks(key: str) -> None:
    cols = st.columns(len(OFFERED_TAGS))
    for col, tag in zip(cols, OFFERED_TAGS):
        with col:
            st.checkbox(TAG_LABELS.get(tag, tag), key=_tag_widget(tag, key))


# ── Confirm callbacks ─────────────────────────────────────────────────────────

def confirm_all_proposed(rows: list[dict], proposed: str, n_groups: int) -> None:
    """Every image in this group as the proposed species, ignoring dropdowns. The
    checkboxes are kept: they say who else was there, not what the animal is."""
    when = rr.now()
    for row in rows:
        key = rr.row_key(row)
        st.session_state.decisions[key] = Decision(
            proposed, "confirmed", ticked(key), when, rr.image_path(row))
    _advance(n_groups)


def confirm_with_edits(rows: list[dict], per_image_proposed: dict, n_groups: int) -> None:
    """Each image as its own dropdown says, with its checkboxes."""
    when = rr.now()
    for row in rows:
        key      = rr.row_key(row)
        proposed = per_image_proposed[key]
        chosen   = st.session_state.get(f"sel_{key}", proposed)
        if chosen == "Otro (especificar)":
            chosen = st.session_state.get(f"other_{key}", "").strip() or proposed
        outcome = "corrected" if chosen != proposed else "confirmed"
        st.session_state.decisions[key] = Decision(
            chosen, outcome, ticked(key), when, rr.image_path(row))
    _advance(n_groups)


# ── Image cell (triptych: prev / current / next) ──────────────────────────────

def render_triptych(path: str, campaign_dir: Path, station_index: dict) -> None:
    def thumb(rel_path: str | None, label: str) -> None:
        if rel_path is None:
            st.caption(f"_{label}: —_")
            return
        img = load_thumbnail(str(campaign_dir / rel_path.replace("\\", "/")))
        if img:
            st.image(img, use_container_width=True)
        else:
            st.warning("imagen no encontrada")
        st.caption(label)

    prev_fp, next_fp = neighbors(path.lower(), station_index)
    tri = st.columns(3)
    with tri[0]:
        thumb(prev_fp, "anterior")
    with tri[1]:
        thumb(path, "actual")
    with tri[2]:
        thumb(next_fp, "siguiente")
    st.caption(path)


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    st.set_page_config(
        page_title="Revisión de Especies",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    config = load_config()
    campaign_dir    = resolve_campaign_dir()
    input_csv_path  = str(campaign_dir / config["output_csv"])   # classified output
    json_path       = str(campaign_dir / config["megadetector_json"])
    st.sidebar.caption(f"Campaña: `{campaign_dir}`")
    station_index = load_station_index(json_path)

    has_review = (campaign_dir / rr.REVIEWED_FILENAME).exists()
    mode = st.sidebar.radio(
        "Modo", [MODE_REVIEW, MODE_TAGS], horizontal=True,
        disabled=not has_review,
        help="Etiquetas: solo para campañas revisadas antes de las casillas de "
             "co-presencia. Guarda únicamente las etiquetas.",
    )
    if mode == MODE_TAGS:
        tags_mode(campaign_dir, station_index)
    else:
        review_mode(campaign_dir, input_csv_path, station_index)


def review_mode(campaign_dir: Path, input_csv_path: str, station_index: dict) -> None:
    species_list     = clip_species()
    species_options  = sorted([s["spanish"] for s in species_list]) + SPECIAL_OPTIONS
    spanish_to_latin = {s["spanish"]: s["latin"] for s in species_list}

    # ── Load data ─────────────────────────────────────────────────────────────
    fieldnames, all_rows = load_classified_csv(input_csv_path)

    animal_rows = [r for r in all_rows if r["observationType"] == exports.TYPE_ANIMAL]
    if not animal_rows:
        st.error("No se encontraron filas de animales clasificados en el CSV de entrada. Ejecuta run_classification.py primero.")
        return

    groups: dict[str, list[dict]] = defaultdict(list)
    for row in animal_rows:
        groups[nfc(row["observationComments"])].append(row)

    sorted_species = sorted(groups.keys(), key=lambda s: -len(groups[s]))
    n_groups = len(sorted_species)

    # ── Session state init ────────────────────────────────────────────────────
    if "current_group" not in st.session_state:
        st.session_state.current_group = 0

    # ── Resume from previously-exported reviewed CSV ──────────────────────────
    # Streamlit session state is in-memory only; if the server restarts or the
    # cache is cleared in a way that drops the session, the review progress is
    # lost. The exported CSV is the durable record — rehydrate from it on a
    # fresh session so the reviewer doesn't lose work. Decisions come back with
    # their tags and their original review date.
    if "decisions" not in st.session_state:
        st.session_state.decisions = rr.read_decisions(campaign_dir)
        # Jump to the first species batch that still has unconfirmed images
        for i, sp in enumerate(sorted_species):
            if any(rr.row_key(r) not in st.session_state.decisions for r in groups[sp]):
                st.session_state.current_group = i
                break

    decisions      = st.session_state.decisions
    n_confirmed    = sum(1 for r in animal_rows if rr.row_key(r) in decisions)
    n_total        = len(animal_rows)

    # ── Sidebar ───────────────────────────────────────────────────────────────
    with st.sidebar:
        st.title("Revisión de Especies")
        st.metric("Confirmadas", f"{n_confirmed} / {n_total}")
        st.progress(n_confirmed / n_total if n_total else 0)
        st.divider()

        st.caption("Ir a especie:")
        for i, sp in enumerate(sorted_species):
            grp = groups[sp]
            n_done = sum(1 for r in grp if rr.row_key(r) in decisions)
            n_corr = sum(
                1 for r in grp
                if (d := decisions.get(rr.row_key(r))) and d.outcome == "corrected"
            )
            done_mark = "✓" if n_done == len(grp) else ("◑" if n_done else "○")
            corr_tag = f", {n_corr} corr." if n_corr else ""
            label = f"{done_mark} {sp}  ({n_done}/{len(grp)}{corr_tag})"  # "corr." = corregidas
            if st.button(label, key=f"jump_{i}", use_container_width=True):
                go_to(i)
                st.rerun()

        st.divider()
        export_clicked = st.button(
            "Exportar CSV revisado",
            type="primary",
            use_container_width=True,
            disabled=n_confirmed == 0,
        )
        if export_clicked:
            try:
                out = rr.write_review(all_rows, fieldnames, decisions,
                                      spanish_to_latin, campaign_dir)
                st.success(f"Guardado: {out.name}")
            except rr.ReviewRecordError as exc:
                st.error(str(exc))

    # ── All done ──────────────────────────────────────────────────────────────
    idx = st.session_state.current_group
    if idx >= n_groups:
        st.success("¡Todas las especies revisadas! Exporta el CSV desde la barra lateral.")
        return

    current_species = sorted_species[idx]
    current_rows    = groups[current_species]
    n_done_this     = sum(1 for r in current_rows if rr.row_key(r) in decisions)
    per_image_proposed = {rr.row_key(row): current_species for row in current_rows}

    # ── Header row ────────────────────────────────────────────────────────────
    col_prev, col_title, col_skip = st.columns([1, 6, 1])
    with col_prev:
        st.button(
            "← Anterior",
            disabled=idx == 0,
            on_click=go_to,
            args=(idx - 1,),
        )
    with col_title:
        st.subheader(
            f"Grupo {idx + 1} / {n_groups}: **{current_species}** "
            f"— {len(current_rows)} imágenes  "
            f"({n_done_this} confirmadas)"
        )
    with col_skip:
        st.button(
            "Saltar →",
            disabled=idx >= n_groups - 1,
            on_click=go_to,
            args=(idx + 1,),
        )

    # ── Confirm buttons ───────────────────────────────────────────────────────
    btn_all, btn_edits, _ = st.columns([3, 3, 4])
    with btn_all:
        st.button(
            f"✓ Confirmar todo como '{current_species}'",
            type="primary",
            use_container_width=True,
            on_click=confirm_all_proposed,
            args=(current_rows, current_species, n_groups),
            help="Guarda todas las imágenes de este grupo como la especie sugerida, "
                 "ignorando los menús. Las casillas de co-presencia sí se guardan.",
        )
    with btn_edits:
        st.button(
            "✓ Confirmar con ediciones individuales",
            use_container_width=True,
            on_click=confirm_with_edits,
            args=(current_rows, per_image_proposed, n_groups),
            help="Guarda cada imagen con la especie seleccionada en su menú desplegable "
                 "y sus casillas.",
        )
    st.caption(
        "Si todas las imágenes son correctas usa el primer botón. "
        "Si corregiste alguna con el menú desplegable, usa el segundo botón. "
        "Marca 'Con persona' / 'Con vehículo' si aparecen junto al animal."
    )
    st.divider()

    # ── Image grid ────────────────────────────────────────────────────────────
    for chunk_start in range(0, len(current_rows), COLS_PER_ROW):
        chunk = current_rows[chunk_start : chunk_start + COLS_PER_ROW]
        cols  = st.columns(COLS_PER_ROW)

        for col, row in zip(cols, chunk):
            key           = rr.row_key(row)
            clip_proposed = per_image_proposed[key]

            with col:
                render_triptych(rr.image_path(row), campaign_dir, station_index)

                # Per-image override dropdown (attached to centre image)
                default_idx = find_default_idx(clip_proposed, species_options)
                st.selectbox(
                    "species",
                    species_options,
                    index=default_idx,
                    key=f"sel_{key}",
                    label_visibility="collapsed",
                )
                if st.session_state.get(f"sel_{key}") == "Otro (especificar)":
                    st.text_input(
                        "Especificar",
                        key=f"other_{key}",
                        label_visibility="collapsed",
                        placeholder="Escribir nombre…",
                    )
                    typed = st.session_state.get(f"other_{key}", "").strip()
                    if typed:
                        st.caption(f":green[✓ Se guardará como: '{typed}']")
                    else:
                        st.caption(":orange[Escribe el nombre antes de confirmar]")
                seed_ticks(key, d.tags if (d := decisions.get(key)) else "")
                render_ticks(key)


def tags_mode(campaign_dir: Path, station_index: dict) -> None:
    """Retro pass: tags only, written straight into the existing reviewed CSV."""
    decisions = rr.read_decisions(campaign_dir)
    groups: dict[str, list[tuple[str, Decision]]] = defaultdict(list)
    for key, d in decisions.items():
        groups[nfc(d.species)].append((key, d))
        seed_ticks(key, d.tags)
    species = sorted(groups, key=lambda s: -len(groups[s]))
    if not species:
        st.info("El CSV revisado no tiene imágenes confirmadas.")
        return

    with st.sidebar:
        st.title("Etiquetas de co-presencia")
        st.caption("Solo cambia las etiquetas. La especie, el resultado y la fecha "
                   "de revisión no se pueden modificar aquí.")
        current = st.radio(
            "Especie confirmada",
            species,
            format_func=lambda sp: f"{sp}  ({sum(1 for _, d in groups[sp] if d.tags)}"
                                   f" con etiqueta / {len(groups[sp])})",
            key="tags_species",
        )

    items = groups[current]
    st.subheader(f"**{current}** — {len(items)} imágenes revisadas")
    st.caption("Guarda antes de cambiar de especie: las casillas sin guardar se pierden.")

    if st.button("Guardar etiquetas de este grupo", type="primary"):
        try:
            rr.write_tags(campaign_dir, {key: ticked(key) for key, _ in items})
            # Revisión mode holds decisions in memory; without this its next export
            # would write the pre-save tags back over the file.
            st.session_state.pop("decisions", None)
            st.success(f"Etiquetas guardadas en {rr.REVIEWED_FILENAME}")
        except rr.ReviewRecordError as exc:
            st.error(str(exc))
    st.divider()

    for chunk_start in range(0, len(items), COLS_PER_ROW):
        cols = st.columns(COLS_PER_ROW)
        for col, (key, d) in zip(cols, items[chunk_start : chunk_start + COLS_PER_ROW]):
            with col:
                render_triptych(d.path, campaign_dir, station_index)
                render_ticks(key)


if __name__ == "__main__":
    main()
