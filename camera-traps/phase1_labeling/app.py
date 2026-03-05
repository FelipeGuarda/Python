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
import unicodedata
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import streamlit as st
import yaml
from PIL import Image

# ── Config ────────────────────────────────────────────────────────────────────
CONFIG_PATH   = Path(__file__).parent.parent / "config.yaml"
COLS_PER_ROW    = 5      # thumbnails per row
THUMB_SIZE      = 280    # max px for thumbnail (width or height)
JPEG_QUALITY    = 75
SPECIAL_OPTIONS = ["No es un animal", "No reconocible", "Otro (especificar)"]
RARE_GROUP      = "Otras especies"


# ── Cached data loaders ───────────────────────────────────────────────────────

@st.cache_data
def load_config() -> dict:
    with open(CONFIG_PATH, encoding="utf-8") as f:
        return yaml.safe_load(f)


@st.cache_data
def load_classified_csv(path: str) -> tuple[list[str], list[dict]]:
    with open(path, encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = list(reader.fieldnames)
        rows = list(reader)
    return fieldnames, rows


@st.cache_data
def load_common_species(old_csv_path: str, min_count: int) -> set[str]:
    """Return Spanish species names with count >= min_count in the historical CSV."""
    with open(old_csv_path, encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    counts = Counter(
        r["Especie"].strip() for r in rows
        if r.get("Animal", "").strip() == "Si" and r.get("Especie", "").strip()
    )
    return {sp for sp, n in counts.items() if n >= min_count}


@st.cache_data
def load_bboxes(json_path: str, threshold: float) -> dict[str, tuple]:
    """Returns {normalised_file_path: (x, y, w, h) or None}."""
    with open(json_path, encoding="utf-8") as f:
        data = json.load(f)
    result: dict[str, tuple | None] = {}
    for img in data["images"]:
        key = img["file"].replace("\\", "/").lower()
        best = None
        for det in img.get("detections", []):
            if det.get("category") == "1" and det.get("conf", 0) >= threshold:
                if best is None or det["conf"] > best["conf"]:
                    best = det
        result[key] = tuple(best["bbox"]) if best else None
    return result


@st.cache_data
def load_thumbnail(image_path: str, bbox: tuple | None) -> bytes | None:
    """Crop image to bbox and return JPEG bytes. Result is cached."""
    try:
        img = Image.open(image_path).convert("RGB")
        if bbox is not None:
            iw, ih = img.size
            x, y, bw, bh = bbox
            pad = 0.05
            x1 = max(0, int((x - pad) * iw))
            y1 = max(0, int((y - pad) * ih))
            x2 = min(iw, int((x + bw + pad) * iw))
            y2 = min(ih, int((y + bh + pad) * ih))
            img = img.crop((x1, y1, x2, y2))
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
        fp = row.get("filePath", "")
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
        fp = row["filePath"]
        st.session_state.confirmed[fp] = proposed
        st.session_state.outcomes[fp]  = "confirmed"
    _advance(n_groups)


def confirm_with_edits(rows: list[dict], per_image_proposed: dict, n_groups: int) -> None:
    """Save each image using its current per-image dropdown value."""
    for row in rows:
        fp       = row["filePath"]
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
        page_title="Species Review",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    config = load_config()
    campaign_dir    = Path(config["campaign_dir"])
    input_csv_path  = str(campaign_dir / config["output_csv"])   # classified output
    json_path       = str(campaign_dir / config["megadetector_json"])
    threshold       = float(config["animal_confidence_threshold"])

    species_list     = config["species"]
    species_options  = sorted([s["spanish"] for s in species_list]) + SPECIAL_OPTIONS
    spanish_to_latin = {s["spanish"]: s["latin"] for s in species_list}

    old_csv_path = str(CONFIG_PATH.parent / config.get("old_data_csv", "old animal data DB.csv"))
    min_count    = int(config.get("min_historical_count", 31))
    common_species = load_common_species(old_csv_path, min_count)

    # ── Load data ─────────────────────────────────────────────────────────────
    fieldnames, all_rows = load_classified_csv(input_csv_path)
    bboxes = load_bboxes(json_path, threshold)

    animal_rows = [r for r in all_rows if r["observationType"] == "animal"]
    if not animal_rows:
        st.error("No classified animal rows found in the input CSV. Run run_classification.py first.")
        return

    # Group by proposed Spanish species name.
    # Common species (>= min_historical_count in old data) get their own batch.
    # Everything else goes into RARE_GROUP, shown last.
    groups: dict[str, list[dict]] = defaultdict(list)
    for row in animal_rows:
        sp = nfc(row["observationComments"])
        groups[sp if sp in common_species else RARE_GROUP].append(row)

    common_sorted  = sorted([sp for sp in groups if sp != RARE_GROUP], key=lambda s: -len(groups[s]))
    sorted_species = common_sorted + ([RARE_GROUP] if RARE_GROUP in groups else [])
    n_groups = len(sorted_species)

    # ── Session state init ────────────────────────────────────────────────────
    if "current_group" not in st.session_state:
        st.session_state.current_group = 0
    if "confirmed" not in st.session_state:
        st.session_state.confirmed: dict[str, str] = {}
    if "outcomes" not in st.session_state:
        st.session_state.outcomes: dict[str, str] = {}

    confirmed      = st.session_state.confirmed
    n_confirmed    = len(confirmed)
    n_total        = len(animal_rows)

    # ── Sidebar ───────────────────────────────────────────────────────────────
    with st.sidebar:
        st.title("Species Review")
        st.metric("Confirmed", f"{n_confirmed} / {n_total}")
        st.progress(n_confirmed / n_total if n_total else 0)
        st.divider()

        st.caption("Jump to species:")
        for i, sp in enumerate(sorted_species):
            grp = groups[sp]
            n_done = sum(1 for r in grp if r["filePath"] in confirmed)
            n_corr = sum(
                1 for r in grp
                if st.session_state.outcomes.get(r["filePath"]) == "corrected"
            )
            done_mark = "✓" if n_done == len(grp) else ("◑" if n_done else "○")
            corr_tag = f", {n_corr} corr." if n_corr else ""
            label = f"{done_mark} {sp}  ({n_done}/{len(grp)}{corr_tag})"
            if st.button(label, key=f"jump_{i}", use_container_width=True):
                go_to(i)
                st.rerun()

        st.divider()
        export_clicked = st.button(
            "Export reviewed CSV",
            type="primary",
            use_container_width=True,
            disabled=n_confirmed == 0,
        )
        if export_clicked:
            out = export_reviewed_csv(
                all_rows, fieldnames, confirmed, st.session_state.outcomes,
                spanish_to_latin, campaign_dir
            )
            st.success(f"Saved: {out.name}")

    # ── All done ──────────────────────────────────────────────────────────────
    idx = st.session_state.current_group
    if idx >= n_groups:
        st.success("All species reviewed! Export the CSV from the sidebar.")
        return

    current_species = sorted_species[idx]
    current_rows    = groups[current_species]
    n_done_this     = sum(1 for r in current_rows if r["filePath"] in confirmed)
    is_rare_group   = current_species == RARE_GROUP

    # per_image_proposed: the CLIP-assigned species for each image.
    # For common-species batches every image shares the same proposed species.
    # For the rare group each image has its own CLIP classification.
    per_image_proposed = {
        row["filePath"]: (nfc(row["observationComments"]) if is_rare_group else current_species)
        for row in current_rows
    }

    # ── Header row ────────────────────────────────────────────────────────────
    col_prev, col_title, col_skip = st.columns([1, 6, 1])
    with col_prev:
        st.button(
            "← Prev",
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
        if is_rare_group:
            st.caption("Especies poco frecuentes agrupadas — cada imagen muestra su propia clasificación CLIP.")
    with col_skip:
        st.button(
            "Skip →",
            disabled=idx >= n_groups - 1,
            on_click=go_to,
            args=(idx + 1,),
        )

    # ── Confirm buttons ───────────────────────────────────────────────────────
    if is_rare_group:
        st.button(
            "✓ Confirmar con ediciones individuales",
            type="primary",
            on_click=confirm_with_edits,
            args=(current_rows, per_image_proposed, n_groups),
            help="Guarda cada imagen con la especie seleccionada en su menú desplegable.",
        )
        st.caption("Corrige los menús desplegables que no correspondan y luego confirma.")
    else:
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

    # ── Image grid ────────────────────────────────────────────────────────────
    for chunk_start in range(0, len(current_rows), COLS_PER_ROW):
        chunk = current_rows[chunk_start : chunk_start + COLS_PER_ROW]
        cols  = st.columns(COLS_PER_ROW)

        for col, row in zip(cols, chunk):
            fp            = row["filePath"]
            norm_fp       = fp.replace("\\", "/").lower()
            bbox          = bboxes.get(norm_fp)
            full_path     = str(campaign_dir / fp.replace("\\", "/"))
            clip_proposed = per_image_proposed[fp]

            with col:
                thumb = load_thumbnail(full_path, bbox)
                if thumb:
                    st.image(thumb, use_container_width=True)
                else:
                    st.warning("image not found")

                # Caption: path, and CLIP suggestion when in rare group
                if is_rare_group:
                    st.caption(f"{fp.replace(chr(92), '/')}  ·  CLIP: *{clip_proposed}*")
                else:
                    st.caption(fp.replace("\\", "/"))

                # Per-image override dropdown
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
