#!/usr/bin/env python3
"""
Obsidian Vault Rewriter — Step 2
Adds YAML frontmatter, normalizes tags, and converts text mentions to [[wiki-links]].

Modes:
  --dry-run    Show what would change (default)
  --apply      Actually write changes to files
  --report     Write a detailed change report to rewrite_preview.md
"""

import os
import re
import sys
import yaml
import json
from pathlib import Path
from datetime import datetime
from collections import Counter, defaultdict

VAULT_PATH = Path("/home/fguarda/Documents/Obsidian FG")
EXCLUDE_DIRS = {".obsidian", ".smart-env", ".git", ".trash"}
REPORT_PATH = Path(__file__).parent / "rewrite_preview.md"
DATA_PATH = Path(__file__).parent / "vault_diagnostic_data.json"

# ─── Tag Normalization Map ───────────────────────────────────────────────
# Maps messy existing tags → clean taxonomy tags.
# Multiple old tags can map to the same new tag.
TAG_NORMALIZE = {
    # Area tags — collapse case variants
    "Conservación": "area/conservación",
    "Conservacion": "area/conservación",
    "conservacion": "area/conservación",
    "Arte": "area/arte",
    "Evaluación": "area/evaluación",
    "Aprendizajes": "area/aprendizajes",
    "Educacion": "area/aprendizajes",
    "FondoFMA": "area/fondo",
    "Fondo FMA": "area/fondo",
    "global": "area/global",
    "Personales": "area/personal",

    # Sub-area tags — conservación
    "Conservacion/incendios": "area/conservación/incendios",
    "Conservacion/incendios/diseño": "area/conservación/incendios",
    "incendios": "area/conservación/incendios",
    "Conservacion/montaje": "area/conservación/montaje",
    "Conservacion/camarasTrampa": "area/conservación/cámaras-trampa",
    "CamarasTrampa": "area/conservación/cámaras-trampa",
    "Conservacion/servidor": "area/conservación/servidor",
    "Conservacion/logistica": "area/conservación/logística",
    "Conservación/logística": "area/conservación/logística",
    "Conservacion/rendiciones": "area/conservación/rendiciones",
    "Conservacion/alianzas": "area/conservación/alianzas",
    "conservacion/estudios": "area/conservación/estudios",
    "Smartforest": "area/conservación/smart-forest",
    "herbario": "area/conservación/herbario",
    "Dendrocronologia": "area/conservación/dendrocronología",
    "Bosques": "area/conservación",
    "Pepperwood": "area/conservación/pepperwood",
    "Photosintesis": "area/conservación/photosíntesis",

    # Sub-area tags — arte
    "Residencias": "area/arte/residencias",
    "Poligonal": "area/arte/poligonal",
    "Poligonal/panel": "area/arte/poligonal",
    "poligonal/preparacion": "area/arte/poligonal",
    "Publicaciones": "area/arte/publicaciones",
    "ObrasNaturales": "area/arte/obras-naturales",
    "BienalDeArtesMediales": "area/arte/bienal",
    "arte-investigacion": "area/arte/investigación",
    "SalaLibre": "area/arte/sala-libre",
    "Colaboraciones": "area/arte/colaboraciones",

    # Sub-area tags — aprendizajes
    "EXPLORA": "area/aprendizajes/explora",
    "Escuelaendemica": "area/aprendizajes/escuela-endémica",
    "CECREA": "area/aprendizajes/cecrea",
    "capacitacion": "area/aprendizajes/capacitación",
    "Cursos": "area/aprendizajes/cursos",
    "MIRAS": "area/aprendizajes/miras",
    "PAI": "area/aprendizajes/pai",
    "UAH": "area/aprendizajes/uah",
    "UCV": "area/aprendizajes/ucv",
    "interculturalidad": "area/aprendizajes/interculturalidad",
    "transdisciplina": "area/aprendizajes/transdisciplina",

    # Sub-area tags — fondo
    "seleccionados": "area/fondo/seleccionados",
    "selección_2025": "area/fondo/selección-2025",

    # BP logistics
    "BP/logistica": "bp/logística",
    "BP/": "bp",
    "BP": "bp",
    "BP/cronograma": "bp/cronograma",

    # Type tags
    "Poema": "type/poema",
    "Cuento": "type/cuento",
    "quote": "type/quote",
    "Quote": "type/quote",
    "Diario": "type/diario",
    "Movie": "type/movie",
    "Libro": "type/libro",
    "semilla": "type/semilla",
    "Conceptos": "type/concepto",
    "tatuaje": "type/personal",
    "sueño": "type/personal",
    "fanzine": "type/fanzine",
    "Bitácora": "type/bitácora",
    "NotaEvaluación": "type/nota-evaluación",
    "Mensualidad": "type/personal",

    # Tech tags
    "AI": "tech/ai",
    "ai": "tech/ai",
    "chatgpt": "tech/chatgpt",
    "git": "tech/git",
    "github": "tech/github",
    "Data/visualization": "tech/data-viz",
    "Analisis": "tech/análisis",
    "hardware": "tech/hardware",
    "nas": "tech/nas",
    "servidor": "tech/servidor",
    "gcp": "tech/gcp",
    "datamanagement": "tech/data-management",
    "panel": "tech/panel",
    "script": "tech/script",

    # Method tags
    "Metodologias/mesasDialogo": "método/mesas-diálogo",
    "Metodología": "método",
    "Investigacion": "tipo/investigación",

    # Program/event tags
    "SemanaFMA": "evento/semana-fma",
    "Planificación": "tipo/planificación",
    "planificación": "tipo/planificación",
    "Planillas": "tipo/planilla",
    "posiblesproyectos": "tipo/idea",
    "Proyectos": "tipo/proyecto",
    "saberes_y_sabores": "evento/saberes-y-sabores",
    "La_memoria_del_bosque": "evento/memoria-del-bosque",
    "Semana_Creativa_UCT": "evento/semana-creativa-uct",
    "Palguin/comunidad": "lugar/palguín",
    "Toltén": "lugar/toltén",
    "Ecología": "tema/ecología",
    "teoria": "tema/teoría",

    # Organization tags
    "FMA": "org/fma",
    "fma": "org/fma",
    "NUBE": "org/nube",
    "ManzanaVerde": "org/manzana-verde",
    "GWW": "org/gww",
    "CentreforCulturalValue": "org/centre-cultural-value",
    "CEDEL": "org/cedel",
    "EarsToSee": "proyecto/ears-to-see",
    "Ladera_sur": "org/ladera-sur",

    # Person tags → person/ namespace
    "JenniferGabrys": "persona/jennifer-gabrys",
    "LisaMicheli": "persona/lisa-micheli",
    "GonzaloRozas": "persona/gonzalo-rozas",
    "NicolásLagos": "persona/nicolás-lagos",
    "FelipeAraneda": "persona/felipe-araneda",
    "Andres_Keller": "persona/andrés-keller",
    "DanielOpazo": "persona/daniel-opazo",
    "LeaMoro": "persona/lea-moro",
    "LinaGomez": "persona/lina-gómez",
    "RodrigoAstaburuaga": "persona/rodrigo-astaburuaga",
    "JosefinaAstorga": "persona/josefina-astorga",
    "Paula_de_Solminihac": "persona/paula-de-solminihac",
    "José_Antonio_Gutierrez": "persona/josé-antonio-gutiérrez",
    "Felipe_Bravo": "persona/felipe-bravo",
    "Pamela_Chavez": "persona/pamela-chávez",
    "Eduardo_Garzúa": "persona/eduardo-garzúa",
    "Alfredo_Blanco": "persona/alfredo-blanco",
    "OlgaRuminot": "persona/olga-ruminot",
    "BlancaBesa": "persona/blanca-besa",
    "MaríaPazGonzalez": "persona/maría-paz-gonzález",
    "AnaMejias": "persona/ana-mejías",
    "IviMarifil": "persona/ivi-marifil",
    "ClaudiaRios": "persona/claudia-ríos",
    "DanielaGaete": "persona/daniela-gaete",
    "NicolásAracena": "persona/nicolás-aracena",
    "AliciaUgarte": "persona/alicia-ugarte",
    "ConstanzaMonterrubio": "persona/constanza-monterrubio",
    "MaricarmenPino": "persona/maricarmen-pino",
    "LuzUgarte": "persona/luz-ugarte",

    # Initials — keep as-is but lowercase (context-dependent, too ambiguous to expand)
    "MJO": "persona/mjo",
    "MH": "persona/mh",
    "FG": "persona/fg",
    "AJ": "persona/aj",
    "ME": "persona/me",
    "PdM": "area/conservación/pdm",
    "SN": "lugar/sn",
    "SC": "org/sc",
    "OdC": "org/odc",
    "AP": "persona/ap",
    "DA": "persona/da",

    # SecondBrain / session tags — keep as-is
    "topic-note": "type/topic-note",
    "session-log": "type/session-log",
    "project": "type/project",
    "resource": "type/resource",
    "index": "type/index",
    "secondbrain": "meta/secondbrain",
    "obsidian-secondbrain": "meta/secondbrain",

    # Tech project tags — keep clean
    "plataforma-territorial": "project/plataforma-territorial",
    "weather-dashboard": "project/weather-dashboard",
    "data-pipeline": "project/data-pipeline",
    "comparison-mode": "project/comparison-mode",
    "two-machine": "project/two-machine",
    "systemd": "tech/systemd",
    "fastapi": "tech/fastapi",
    "react": "tech/react",
    "workflow": "tipo/workflow",
    "windows": "tech/windows",
    "platform": "tech/platform",
    "setup": "tipo/setup",
    "tool": "type/tool",
    "organization": "type/organization",

    # Hex color codes (false positives from inline tags) — drop
    "c71a15": "_drop",
    "d35907": "_drop",
    "d38107": "_drop",
    "d3aa07": "_drop",
    "d0d307": "_drop",
    "248107": "_drop",
    "07c5d3": "_drop",
    "0795d3": "_drop",
    "0763d3": "_drop",
    "a067bd": "_drop",
}

# ─── Directory → default tags mapping ────────────────────────────────────
DIR_TAGS = {
    "Journal": ["type/journal"],
    "FMA": ["org/fma"],
    "FMA/Conservación": ["org/fma", "area/conservación"],
    "FMA/Arte": ["org/fma", "area/arte"],
    "FMA/Aprendizajes": ["org/fma", "area/aprendizajes"],
    "FMA/Evaluación": ["org/fma", "area/evaluación"],
    "FMA/Fondo FMA": ["org/fma", "area/fondo"],
    "FMA/Global": ["org/fma", "area/global"],
    "Personales": ["area/personal"],
    "Personales/Cosas escritas y quotes": ["area/personal", "type/creative"],
    "Literature Notes": ["type/literature"],
    "Literature Notes/Kindle Book Highlights": ["type/literature", "type/kindle"],
    "Literature Notes/PDFs": ["type/literature", "type/pdf-notes"],
    "Data Visualization": ["tech/data-viz"],
    "SecondBrain/Sessions": ["type/session-log"],
    "SecondBrain/Topics": ["type/topic-note"],
    "SecondBrain/Resources": ["type/resource"],
}

# ─── Auto-link targets ───────────────────────────────────────────────────
# Note stems that should become [[wiki-links]] when found as plain text.
# Only high-value targets (mentioned in 4+ files without links).
# We load these from the diagnostic data.
MIN_MENTION_COUNT = 3  # minimum unlinked mentions to qualify


def iter_md_files(vault: Path):
    for root, dirs, files in os.walk(vault):
        dirs[:] = [d for d in dirs if d not in EXCLUDE_DIRS]
        for f in files:
            if f.endswith(".md"):
                yield Path(root) / f


def parse_frontmatter(content: str) -> tuple[dict | None, str]:
    """Return (frontmatter_dict_or_None, body_after_frontmatter)."""
    if content.startswith("---"):
        end = content.find("---", 3)
        if end != -1:
            fm_str = content[3:end]
            body = content[end + 3:].lstrip("\n")
            try:
                fm = yaml.safe_load(fm_str)
                if isinstance(fm, dict):
                    return fm, body
            except yaml.YAMLError:
                pass
    return None, content


def serialize_frontmatter(fm: dict) -> str:
    """Serialize frontmatter dict to YAML string between --- markers."""
    # Custom serialization to keep it clean
    lines = ["---"]
    for key, value in fm.items():
        if key == "tags" and isinstance(value, list):
            lines.append("tags:")
            for t in sorted(set(value)):
                lines.append(f"  - {t}")
        elif isinstance(value, list):
            lines.append(f"{key}:")
            for item in value:
                lines.append(f"  - {item}")
        elif isinstance(value, dict):
            # Preserve complex objects like kindle-sync as-is
            dumped = yaml.dump({key: value}, default_flow_style=False,
                               allow_unicode=True).rstrip()
            lines.append(dumped)
        elif isinstance(value, str) and ("\n" in value or ":" in value or "#" in value):
            lines.append(f'{key}: "{value}"')
        else:
            lines.append(f"{key}: {value}")
    lines.append("---")
    return "\n".join(lines)


def infer_date_from_filename(filepath: Path) -> str | None:
    """Try to extract a date from the filename."""
    stem = filepath.stem

    # YYYY-MM-DD
    m = re.match(r"^(\d{4}-\d{2}-\d{2})", stem)
    if m:
        return m.group(1)

    # YYYY_MM_DD
    m = re.match(r"^(\d{4})_(\d{2})_(\d{2})", stem)
    if m:
        return f"{m.group(1)}-{m.group(2)}-{m.group(3)}"

    # DD-MM-YYYY
    m = re.match(r"^(\d{2})-(\d{2})-(\d{4})", stem)
    if m:
        return f"{m.group(3)}-{m.group(2)}-{m.group(1)}"

    return None


def infer_date_from_content(content: str) -> str | None:
    """Try to extract a date from inline metadata like 'Created: DD-MM-YYYY'."""
    m = re.search(r"Created:\s*(\d{2})-(\d{2})-(\d{4})", content)
    if m:
        return f"{m.group(3)}-{m.group(2)}-{m.group(1)}"
    m = re.search(r"date:\s*(\d{4}-\d{2}-\d{2})", content)
    if m:
        return m.group(1)
    return None


def extract_inline_metadata(content: str) -> tuple[dict, str]:
    """Extract legacy inline metadata (## Área: #Tag, Created:, tags:) from top of file.
    Returns (metadata_dict, cleaned_body)."""
    meta = {}
    lines = content.split("\n")
    consumed = 0

    for i, line in enumerate(lines[:10]):  # Only check first 10 lines
        stripped = line.strip()

        # ## Área: #Tag #Tag2 or # Tema: #Tag — extract all #tags from the line
        m = re.match(r"^#{1,2}\s*[ÁáAa]rea:\s*(.+)", stripped)
        if m:
            # Extract plain #tags (not #[[wikilinks]])
            area_tags = re.findall(r"#(?!\[\[)([\w/\-áéíóúñüÁÉÍÓÚÑÜ]+)", m.group(1))
            if area_tags:
                meta.setdefault("inline_tags", []).extend(area_tags)
            else:
                meta["area"] = m.group(1).strip().lstrip("#")
            consumed = i + 1
            continue

        m = re.match(r"^#{1,2}\s*[Tt]ema:\s*(.+)", stripped)
        if m:
            tema_tags = re.findall(r"#(?!\[\[)([\w/\-áéíóúñüÁÉÍÓÚÑÜ]+)", m.group(1))
            if tema_tags:
                meta.setdefault("inline_tags", []).extend(tema_tags)
            else:
                meta["tema"] = m.group(1).strip().lstrip("#")
            consumed = i + 1
            continue

        # Created: DD-MM-YYYY HH:MM
        m = re.match(r"^Created:\s*(.+)", stripped)
        if m:
            meta["created"] = m.group(1).strip()
            consumed = i + 1
            continue

        # tags: #Tag1 #Tag2
        m = re.match(r"^tags:\s*(.+)", stripped)
        if m and not stripped.startswith("tags:"):
            pass  # only match if it looks like inline, not YAML
        if m and "---" not in stripped:
            raw_tags = re.findall(r"#(?!\[\[)([\w/\-áéíóúñüÁÉÍÓÚÑÜ]+)", m.group(1))
            if raw_tags:
                meta["inline_tags"] = raw_tags
                consumed = i + 1
                continue

        # # Conceptos: (header with no content, skip)
        m = re.match(r"^#{1,2}\s*Conceptos:\s*$", stripped)
        if m:
            consumed = i + 1
            continue

        # Empty line after metadata block
        if stripped == "" and consumed > 0:
            consumed = i + 1
            continue

        # If we hit real content, stop
        if consumed > 0 and stripped:
            break

    if consumed > 0:
        # Remove consumed lines from body
        cleaned = "\n".join(lines[consumed:]).lstrip("\n")
        return meta, cleaned

    return meta, content


def normalize_tag(tag: str) -> str | None:
    """Normalize a single tag. Returns None if it should be dropped."""
    normalized = TAG_NORMALIZE.get(tag)
    if normalized == "_drop":
        return None
    if normalized:
        return normalized
    # If not in map, keep it lowercase with hyphens
    clean = tag.lower().replace("_", "-").replace(" ", "-")
    return clean


def get_dir_tags(filepath: Path, vault: Path) -> list[str]:
    """Get default tags based on directory."""
    rel = filepath.relative_to(vault)
    rel_dir = str(rel.parent)

    # Try most specific first, then less specific
    for dir_prefix in sorted(DIR_TAGS.keys(), key=len, reverse=True):
        if rel_dir.startswith(dir_prefix):
            return list(DIR_TAGS[dir_prefix])
    return []


def load_link_targets() -> dict[str, int]:
    """Load potential link targets from diagnostic data."""
    if not DATA_PATH.exists():
        return {}
    data = json.loads(DATA_PATH.read_text())
    targets = {}
    for stem, sources in data.get("potential_links", {}).items():
        if len(sources) >= MIN_MENTION_COUNT:
            # Skip date-like stems and very short names
            if re.match(r"^\d{4}[-_]", stem) or len(stem) <= 3:
                continue
            if stem in ("Tasks", "Ideas", "Gone", "Index", "Postulación"):
                continue  # Too generic
            targets[stem] = len(sources)
    return targets


def add_wiki_links(content: str, targets: dict[str, int], existing_links: set[str]) -> tuple[str, list[str]]:
    """Convert plain-text mentions to [[wiki-links]]. Returns (new_content, list_of_conversions)."""
    conversions = []

    for stem in sorted(targets.keys(), key=len, reverse=True):
        if stem in existing_links:
            continue

        # Match whole word, not inside existing [[...]] or code blocks
        # Use a careful regex: word boundary, not preceded by [[ or followed by ]]
        pattern = r'(?<!\[\[)(?<!\w)' + re.escape(stem) + r'(?!\w)(?!\]\])(?!\|)'

        # Only replace first occurrence per file
        match = re.search(pattern, content)
        if match:
            # Check we're not inside a code block or frontmatter
            before = content[:match.start()]
            if before.count("```") % 2 == 1:
                continue  # Inside code block
            if before.count("---") >= 1 and before.count("---") < 2 and content.startswith("---"):
                continue  # Inside frontmatter

            content = content[:match.start()] + f"[[{stem}]]" + content[match.end():]
            conversions.append(stem)
            existing_links.add(stem)

    return content, conversions


def process_file(filepath: Path, link_targets: dict[str, int], apply: bool = False) -> dict | None:
    """Process a single file. Returns change description or None if no changes."""
    rel_path = filepath.relative_to(VAULT_PATH)

    # Skip templates
    if "Templates" in str(rel_path):
        return None

    try:
        content = filepath.read_text(encoding="utf-8", errors="replace")
    except Exception:
        return None

    # Skip empty files (separate cleanup)
    if len(content.strip()) == 0:
        return None

    original_content = content
    changes = []

    # ─── Parse existing state ────────────────────────────────────────
    existing_fm, body = parse_frontmatter(content)
    has_existing_fm = existing_fm is not None

    # Extract legacy inline metadata if no frontmatter
    inline_meta = {}
    if not has_existing_fm:
        inline_meta, body = extract_inline_metadata(body)

    # ─── Build frontmatter ───────────────────────────────────────────
    if has_existing_fm:
        fm = dict(existing_fm)
    else:
        fm = {}

    # Date
    if "date" not in fm:
        date = infer_date_from_filename(filepath)
        if not date and inline_meta.get("created"):
            date = infer_date_from_content(f"Created: {inline_meta['created']}")
        if not date:
            date = infer_date_from_content(original_content)
        if not date:
            # Fall back to file modification time
            mtime = os.path.getmtime(filepath)
            date = datetime.fromtimestamp(mtime).strftime("%Y-%m-%d")
        if date and "kindle-sync" not in fm:  # Don't add date to Kindle files
            fm["date"] = date
            if not has_existing_fm:
                changes.append(f"+ date: {date}")

    # Tags — collect from all sources
    existing_tags = []
    if "tags" in fm:
        raw = fm["tags"]
        if isinstance(raw, list):
            existing_tags = [str(t) for t in raw if t is not None]
        elif isinstance(raw, str):
            existing_tags = [t.strip() for t in raw.replace(",", " ").split() if t.strip()]

    # Add inline metadata tags (skip wiki-link artifacts)
    if inline_meta.get("inline_tags"):
        for t in inline_meta["inline_tags"]:
            # Skip if it looks like a wiki-link fragment
            if t.startswith("[[") or t.startswith("]]") or "[[" in t:
                continue
            if t not in existing_tags:
                existing_tags.append(t)

    # Add area/tema from inline metadata
    if inline_meta.get("area"):
        area_tag = inline_meta["area"].lstrip("#").strip()
        if area_tag not in existing_tags:
            existing_tags.append(area_tag)
    if inline_meta.get("tema"):
        tema_tag = inline_meta["tema"].lstrip("#").strip()
        if tema_tag not in existing_tags:
            existing_tags.append(tema_tag)

    # Normalize all tags — skip wiki-link artifacts and other noise
    normalized_tags = []
    for t in existing_tags:
        # Skip wiki-link fragments, empty tags, templater syntax
        if "[[" in t or "]]" in t or "<%"  in t or len(t.strip()) == 0:
            continue
        nt = normalize_tag(t)
        if nt and nt not in normalized_tags:
            normalized_tags.append(nt)

    # Add directory-based default tags
    dir_tags = get_dir_tags(filepath, VAULT_PATH)
    for dt in dir_tags:
        if dt not in normalized_tags:
            normalized_tags.append(dt)

    # Record tag changes
    old_tag_set = set(existing_tags)
    new_tag_set = set(normalized_tags)
    if old_tag_set != new_tag_set:
        added = new_tag_set - {normalize_tag(t) for t in old_tag_set
                               if normalize_tag(t) and "[[" not in t and "]]" not in t}
        if added:
            changes.append(f"+ tags: {', '.join(sorted(added))}")
        # Only show renames for clean tags (not wiki-link artifacts or dropped tags)
        renamed = [(old, normalize_tag(old)) for old in old_tag_set
                   if normalize_tag(old) and normalize_tag(old) != old
                   and "[[" not in old and "]]" not in old
                   and normalize_tag(old) != "_drop"]
        if renamed:
            changes.append(f"~ tags renamed: {', '.join(f'{o}→{n}' for o, n in renamed[:5])}")
        # Show dropped tags
        dropped = [t for t in old_tag_set
                   if "[[" in t or "]]" in t or normalize_tag(t) == "_drop" or normalize_tag(t) is None]
        if dropped:
            changes.append(f"- tags dropped: {', '.join(sorted(dropped)[:5])}")

    fm["tags"] = sorted(set(normalized_tags))

    # ─── Auto-link ───────────────────────────────────────────────────
    existing_links = set(re.findall(r"\[\[([^\]|]+?)(?:\|[^\]]+?)?\]\]", body))
    new_body, link_conversions = add_wiki_links(body, link_targets, existing_links)
    if link_conversions:
        changes.append(f"+ links: {', '.join(f'[[{l}]]' for l in link_conversions)}")
        body = new_body

    # ─── Determine if anything changed ───────────────────────────────
    if not changes and has_existing_fm:
        return None

    if not has_existing_fm:
        changes.insert(0, "+ frontmatter added")

    # ─── Reconstruct file ────────────────────────────────────────────
    # Preserve special frontmatter fields (kindle-sync, area_tag, etc.)
    new_content = serialize_frontmatter(fm) + "\n\n" + body

    # ─── Write if applying ───────────────────────────────────────────
    if apply and new_content != original_content:
        filepath.write_text(new_content, encoding="utf-8")

    return {
        "path": str(rel_path),
        "changes": changes,
        "had_frontmatter": has_existing_fm,
        "tag_count": len(normalized_tags),
        "link_count": len(link_conversions),
    }


def main():
    mode = "dry-run"
    if "--apply" in sys.argv:
        mode = "apply"
    report_mode = "--report" in sys.argv or mode == "dry-run"

    print(f"Mode: {mode}")
    print("Loading link targets from diagnostic data...")
    link_targets = load_link_targets()
    print(f"  {len(link_targets)} auto-link targets loaded (≥{MIN_MENTION_COUNT} mentions)")

    print("Processing vault...")
    results = []
    skipped = 0
    errors = 0

    for filepath in sorted(iter_md_files(VAULT_PATH)):
        try:
            result = process_file(filepath, link_targets, apply=(mode == "apply"))
            if result:
                results.append(result)
            else:
                skipped += 1
        except Exception as e:
            print(f"  ERROR: {filepath.relative_to(VAULT_PATH)}: {e}")
            errors += 1

    # ─── Summary ─────────────────────────────────────────────────────
    fm_added = sum(1 for r in results if not r["had_frontmatter"])
    fm_updated = sum(1 for r in results if r["had_frontmatter"])
    total_links = sum(r["link_count"] for r in results)
    total_tag_changes = sum(1 for r in results if any("tags" in c for c in r["changes"]))

    print(f"\n{'═' * 60}")
    print(f"  Files that would change:  {len(results)}")
    print(f"  Files skipped (no change): {skipped}")
    print(f"  Errors:                    {errors}")
    print(f"  ──────────────────────────────────")
    print(f"  Frontmatter added:         {fm_added}")
    print(f"  Frontmatter updated:       {fm_updated}")
    print(f"  Tag normalizations:        {total_tag_changes}")
    print(f"  Wiki-links added:          {total_links}")
    print(f"{'═' * 60}")

    if mode == "apply":
        print(f"\n✓ Changes applied to {len(results)} files.")
    else:
        print(f"\nThis was a DRY RUN. No files were modified.")
        print(f"Run with --apply to make changes.")

    # ─── Write report ────────────────────────────────────────────────
    if report_mode:
        lines = []
        lines.append("---")
        lines.append(f"date: {datetime.now().strftime('%Y-%m-%d')}")
        lines.append("type: rewrite-preview")
        lines.append("---")
        lines.append("")
        lines.append("# Vault Rewrite Preview")
        lines.append(f"*Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}*")
        lines.append(f"*Mode: {mode}*")
        lines.append("")
        lines.append("## Summary")
        lines.append("")
        lines.append(f"| Metric | Count |")
        lines.append(f"|---|---|")
        lines.append(f"| Files changed | {len(results)} |")
        lines.append(f"| Frontmatter added | {fm_added} |")
        lines.append(f"| Frontmatter updated | {fm_updated} |")
        lines.append(f"| Tag normalizations | {total_tag_changes} |")
        lines.append(f"| Wiki-links added | {total_links} |")
        lines.append("")
        lines.append("## Changes by Directory")
        lines.append("")

        by_dir = defaultdict(list)
        for r in results:
            top = Path(r["path"]).parts[0] if len(Path(r["path"]).parts) > 1 else "(root)"
            by_dir[top].append(r)

        for d in sorted(by_dir.keys()):
            items = by_dir[d]
            lines.append(f"### {d}/ ({len(items)} files)")
            lines.append("")
            for r in sorted(items, key=lambda x: x["path"]):
                lines.append(f"**{r['path']}**")
                for c in r["changes"]:
                    lines.append(f"  - {c}")
                lines.append("")

        report = "\n".join(lines)
        REPORT_PATH.write_text(report, encoding="utf-8")
        print(f"\nDetailed report: {REPORT_PATH}")


if __name__ == "__main__":
    main()
