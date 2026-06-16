"""
Species Review App — Phase 1 Labeling UI
==========================================
Streamlit app that shows CLIP-classified camera trap images grouped by proposed
species, lets the user confirm or correct each one, and exports a revised CSV.

Usage (from the species-classifier/ directory):
    streamlit run phase1_labeling/app.py

Output:
    <campaign_dir>/new_labeled_data_reviewed.csv
"""

import csv
import io
import json
import sys
import unicodedata
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import streamlit as st
import yaml
from PIL import Image

# Make the project root importable when launched as `streamlit run phase1_labeling/app.py`
# (Streamlit puts the script's directory on sys.path, not the project root).
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from classify_campaign.species import clip_species

# ── Config ────────────────────────────────────────────────────────────────────
CONFIG_PATH   = Path(__file__).parent.parent / "config.yaml"
COLS_PER_ROW    = 3      # triptychs per row (each cell holds prev/current/next)
THUMB_SIZE      = 1280   # max px on long edge — high enough to stay sharp in Streamlit's expand view
JPEG_QUALITY    = 85
SPECIAL_OPTIONS = ["No es un animal", "No reconocible", "Otro (especificar)"]


# ── Cached data loaders ───────────────────────────────────────────────────────
# Each public loader stat()s its source file and passes the mtime as an extra
# cache key, so the cached value invalidates when the underlying file changes.

def load_config() -> dict:
    return _load_config_cached(CONFIG_PATH.stat().st_mtime)


@st.cache_data
def _load_config_cached(_mtime: float) -> dict:
    with open(CONFIG_PATH, encoding="utf-8") as f:
        return yaml.safe_load(f)


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


# ── CSV export ────────────────────────────────────────────────────────────────

def export_reviewed_csv(
    all_rows: list[dict],
    fieldnames: list[str],
    confirmed: dict[str, str],
    outcomes: dict[str, str],
    spanish_to_latin: dict[str, str],
    campaign_dir: Path,
) -> Path:
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S")
    out_path = campaign_dir / "new_labeled_data_reviewed.csv"

    if "reviewOutcome" not in fieldnames:
        fieldnames = fieldnames + ["reviewOutcome"]

    for row in all_rows:
        fp = row_fp(row)
        if fp in confirmed:
            spanish = confirmed[fp]
            row["observationComments"]     = spanish
            row["scientificName"]          = spanish_to_latin.get(spanish, "")
            row["classifiedBy"]            = "CLIP zero-shot + human review"
            row["classificationTimestamp"] = timestamp
            row["reviewOutcome"]           = outcomes.get(fp, "confirmed")

    with open(out_path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_rows)

    return out_path


# ── Helpers ───────────────────────────────────────────────────────────────────

def nfc(s: str) -> str:
    """Normalize Unicode to NFC and strip whitespace."""
    return unicodedata.normalize("NFC", s.strip())


def row_fp(row: dict) -> str:
    """Return the relative file path for a CSV row.

    Prefers the filePath column; falls back to RelativePath + File when
    filePath is empty (as in campaigns where Timelapse2 didn't populate it).
    """
    fp = row.get("filePath", "").strip()
    if not fp:
        rel   = row.get("RelativePath", "").strip()
        fname = row.get("File", "").strip()
        fp = rel + "/" + fname if rel and fname else ""
    return fp


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


def confirm_all_proposed(rows: list[dict], proposed: str, n_groups: int) -> None:
    """Set every image in this group to the proposed species, ignoring dropdowns."""
    for row in rows:
        fp = row_fp(row)
        st.session_state.confirmed[fp] = proposed
        st.session_state.outcomes[fp]  = "confirmed"
    _advance(n_groups)


def confirm_with_edits(rows: list[dict], per_image_proposed: dict, n_groups: int) -> None:
    """Save each image using its current per-image dropdown value."""
    for row in rows:
        fp       = row_fp(row)
        proposed = per_image_proposed[fp]
        chosen   = st.session_state.get(f"sel_{fp}", proposed)
        if chosen == "Otro (especificar)":
            chosen = st.session_state.get(f"other_{fp}", "").strip() or proposed
        st.session_state.confirmed[fp] = chosen
        st.session_state.outcomes[fp]  = "corrected" if chosen != proposed else "confirmed"
    _advance(n_groups)


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    st.set_page_config(
        page_title="Revisión de Especies",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    config = load_config()
    campaign_dir    = Path(config["campaign_dir"])
    input_csv_path  = str(campaign_dir / config["output_csv"])   # classified output
    json_path       = str(campaign_dir / config["megadetector_json"])

    species_list     = clip_species()
    species_options  = sorted([s["spanish"] for s in species_list]) + SPECIAL_OPTIONS
    spanish_to_latin = {s["spanish"]: s["latin"] for s in species_list}

    # ── Load data ─────────────────────────────────────────────────────────────
    fieldnames, all_rows = load_classified_csv(input_csv_path)
    station_index = load_station_index(json_path)

    animal_rows = [r for r in all_rows if r["observationType"] == "animal"]
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
    if "confirmed" not in st.session_state:
        st.session_state.confirmed = {}
    if "outcomes" not in st.session_state:
        st.session_state.outcomes = {}

    # ── Resume from previously-exported reviewed CSV ──────────────────────────
    # Streamlit session state is in-memory only; if the server restarts or the
    # cache is cleared in a way that drops the session, the review progress is
    # lost. The exported CSV is the durable record — rehydrate from it on a
    # fresh session so the reviewer doesn't lose work.
    reviewed_csv = campaign_dir / "new_labeled_data_reviewed.csv"
    if reviewed_csv.exists() and not st.session_state.confirmed:
        with open(reviewed_csv, encoding="utf-8-sig", newline="") as f:
            for row in csv.DictReader(f):
                outcome = (row.get("reviewOutcome") or "").strip()
                if not outcome:
                    continue
                fp = row_fp(row)
                if not fp:
                    continue
                st.session_state.confirmed[fp] = (row.get("observationComments") or "").strip()
                st.session_state.outcomes[fp]  = outcome
        # Jump to the first species batch that still has unconfirmed images
        for i, sp in enumerate(sorted_species):
            if any(row_fp(r) not in st.session_state.confirmed for r in groups[sp]):
                st.session_state.current_group = i
                break

    confirmed      = st.session_state.confirmed
    n_confirmed    = len(confirmed)
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
            n_done = sum(1 for r in grp if row_fp(r) in confirmed)
            n_corr = sum(
                1 for r in grp
                if st.session_state.outcomes.get(row_fp(r)) == "corrected"
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
            out = export_reviewed_csv(
                all_rows, fieldnames, confirmed, st.session_state.outcomes,
                spanish_to_latin, campaign_dir
            )
            st.success(f"Guardado: {out.name}")

    # ── All done ──────────────────────────────────────────────────────────────
    idx = st.session_state.current_group
    if idx >= n_groups:
        st.success("¡Todas las especies revisadas! Exporta el CSV desde la barra lateral.")
        return

    current_species = sorted_species[idx]
    current_rows    = groups[current_species]
    n_done_this     = sum(1 for r in current_rows if row_fp(r) in confirmed)
    per_image_proposed = {row_fp(row): current_species for row in current_rows}

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
            help="Guarda todos las imágenes de este grupo como la especie sugerida, ignorando ediciones.",
        )
    with btn_edits:
        st.button(
            "✓ Confirmar con ediciones individuales",
            use_container_width=True,
            on_click=confirm_with_edits,
            args=(current_rows, per_image_proposed, n_groups),
            help="Guarda cada imagen con la especie seleccionada en su menú desplegable.",
        )
    st.caption(
        "Si todas las imágenes son correctas usa el primer botón. "
        "Si corregiste alguna con el menú desplegable, usa el segundo botón."
    )
    st.divider()

    # ── Image grid (triptych per cell: prev / current / next) ────────────────
    def render_thumb(rel_path: str | None, label: str) -> None:
        """Render one slot of the triptych. rel_path=None means no neighbour."""
        if rel_path is None:
            st.caption(f"_{label}: —_")
            return
        full_path = str(campaign_dir / rel_path.replace("\\", "/"))
        thumb = load_thumbnail(full_path)
        if thumb:
            st.image(thumb, use_container_width=True)
        else:
            st.warning("imagen no encontrada")
        st.caption(label)

    for chunk_start in range(0, len(current_rows), COLS_PER_ROW):
        chunk = current_rows[chunk_start : chunk_start + COLS_PER_ROW]
        cols  = st.columns(COLS_PER_ROW)

        for col, row in zip(cols, chunk):
            fp            = row_fp(row)
            norm_fp       = fp.replace("\\", "/").lower()
            prev_fp, next_fp = neighbors(norm_fp, station_index)
            clip_proposed = per_image_proposed[fp]

            with col:
                tri = st.columns(3)
                with tri[0]:
                    render_thumb(prev_fp, "anterior")
                with tri[1]:
                    render_thumb(fp, "actual")
                with tri[2]:
                    render_thumb(next_fp, "siguiente")

                st.caption(fp.replace("\\", "/"))

                # Per-image override dropdown (attached to centre image)
                default_idx = find_default_idx(clip_proposed, species_options)
                st.selectbox(
                    "species",
                    species_options,
                    index=default_idx,
                    key=f"sel_{fp}",
                    label_visibility="collapsed",
                )
                if st.session_state.get(f"sel_{fp}") == "Otro (especificar)":
                    st.text_input(
                        "Especificar",
                        key=f"other_{fp}",
                        label_visibility="collapsed",
                        placeholder="Escribir nombre…",
                    )
                    typed = st.session_state.get(f"other_{fp}", "").strip()
                    if typed:
                        st.caption(f":green[✓ Se guardará como: '{typed}']")
                    else:
                        st.caption(":orange[Escribe el nombre antes de confirmar]")


if __name__ == "__main__":
    main()
