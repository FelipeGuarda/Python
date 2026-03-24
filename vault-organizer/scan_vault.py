#!/usr/bin/env python3
"""
Obsidian Vault Scanner — Diagnostic Inventory Generator
Scans all .md files, extracts structure, tags, links, entities,
and produces a comprehensive diagnostic report.
"""

import os
import re
import yaml
import json
from pathlib import Path
from collections import Counter, defaultdict
from datetime import datetime

VAULT_PATH = Path("/home/fguarda/Documents/Obsidian FG")
EXCLUDE_DIRS = {".obsidian", ".smart-env", ".git", ".trash"}
OUTPUT_PATH = Path(__file__).parent / "vault_diagnostic.md"


def iter_md_files(vault: Path):
    """Yield all .md files in the vault, excluding system directories."""
    for root, dirs, files in os.walk(vault):
        dirs[:] = [d for d in dirs if d not in EXCLUDE_DIRS]
        for f in files:
            if f.endswith(".md"):
                yield Path(root) / f


def parse_frontmatter(content: str):
    """Extract YAML frontmatter if present."""
    if content.startswith("---"):
        end = content.find("---", 3)
        if end != -1:
            try:
                return yaml.safe_load(content[3:end])
            except yaml.YAMLError:
                return {}
    return {}


def extract_tags_from_frontmatter(fm: dict) -> list[str]:
    """Extract tags from YAML frontmatter."""
    tags = fm.get("tags", [])
    if isinstance(tags, str):
        tags = [t.strip() for t in tags.replace(",", " ").split()]
    elif isinstance(tags, list):
        flat = []
        for t in tags:
            if isinstance(t, str):
                flat.append(t.strip())
        tags = flat
    else:
        tags = []
    return [t.lstrip("#") for t in tags if t]


def extract_inline_tags(content: str) -> list[str]:
    """Extract inline #tags (not inside code blocks or URLs)."""
    # Remove code blocks first
    cleaned = re.sub(r"```.*?```", "", content, flags=re.DOTALL)
    cleaned = re.sub(r"`[^`]+`", "", cleaned)
    # Match #tag patterns (allow nested like #area/conservación)
    matches = re.findall(r"(?:^|[\s,(])#([\w/\-áéíóúñüÁÉÍÓÚÑÜ]+)", cleaned)
    return [m for m in matches if len(m) > 1]


def extract_wiki_links(content: str) -> list[str]:
    """Extract [[wiki-link]] targets."""
    # Match [[target]] and [[target|display]]
    matches = re.findall(r"\[\[([^\]|]+?)(?:\|[^\]]+?)?\]\]", content)
    return [m.strip() for m in matches]


def extract_external_links(content: str) -> list[tuple[str, str]]:
    """Extract [text](url) markdown links."""
    return re.findall(r"\[([^\]]*)\]\((https?://[^\)]+)\)", content)


def get_relative_dir(filepath: Path, vault: Path) -> str:
    """Get the top-level directory name relative to vault."""
    rel = filepath.relative_to(vault)
    parts = rel.parts
    if len(parts) > 1:
        return parts[0]
    return "(root)"


def get_subdirectory(filepath: Path, vault: Path) -> str:
    """Get full relative directory path."""
    rel = filepath.relative_to(vault)
    if len(rel.parts) > 1:
        return str(rel.parent)
    return "(root)"


def extract_headings(content: str) -> list[tuple[int, str]]:
    """Extract markdown headings with their level."""
    return [(len(m.group(1)), m.group(2).strip())
            for m in re.finditer(r"^(#{1,6})\s+(.+)$", content, re.MULTILINE)]


def detect_entities(content: str) -> dict[str, set]:
    """Detect potential named entities (people, places, organizations) via patterns."""
    entities = defaultdict(set)

    # Known FMA-related entities to look for
    known_people = [
        "Antonio Lara", "Tomás Ibarra", "Jennifer Gabrys", "Pablo Lobos",
        "Madeline Hurtado", "Felipe Guarda", "Alejandra Sierralta",
        "Francisco Saavedra", "Michelle Morata", "Gabriela Tapia",
        "Francisca Gazitúa", "Thomas Kitzberger", "Aníbal Pauchard",
    ]
    known_places = [
        "Bosque Pehuén", "Pepperwood", "La Araucanía", "Lonquimay",
        "Malalcahuello", "Sierra Nevada", "Tres Hermanas",
    ]
    known_orgs = [
        "Fundación Mar Adentro", "FMA", "CONAF", "SAG", "SMART",
        "Johns Hopkins", "GFW", "Global Forest Watch",
        "Open-Meteo", "DuckDB", "Streamlit",
    ]

    for person in known_people:
        if person.lower() in content.lower():
            entities["people"].add(person)

    for place in known_places:
        if place.lower() in content.lower():
            entities["places"].add(place)

    for org in known_orgs:
        if org.lower() in content.lower():
            entities["organizations"].add(org)

    return entities


def scan_vault():
    """Main scanner — processes all files and builds the diagnostic data."""
    files_data = []
    all_tags = Counter()
    all_inline_tags = Counter()
    all_wiki_links = Counter()
    all_external_domains = Counter()
    dir_counts = Counter()
    subdir_counts = Counter()
    empty_files = []
    files_with_frontmatter = 0
    files_with_tags = 0
    files_with_wiki_links = 0
    entity_mentions = defaultdict(lambda: defaultdict(list))
    link_targets = defaultdict(list)  # wiki-link target -> list of source files
    tag_to_files = defaultdict(list)
    word_counts = Counter()
    total_words = 0

    for filepath in iter_md_files(VAULT_PATH):
        rel_path = filepath.relative_to(VAULT_PATH)
        top_dir = get_relative_dir(filepath, VAULT_PATH)
        sub_dir = get_subdirectory(filepath, VAULT_PATH)
        dir_counts[top_dir] += 1
        subdir_counts[sub_dir] += 1

        try:
            content = filepath.read_text(encoding="utf-8", errors="replace")
        except Exception:
            continue

        # Empty file check
        if len(content.strip()) == 0:
            empty_files.append(str(rel_path))
            files_data.append({
                "path": str(rel_path),
                "top_dir": top_dir,
                "empty": True,
                "words": 0,
            })
            continue

        # Frontmatter
        fm = parse_frontmatter(content)
        has_fm = bool(fm)
        if has_fm:
            files_with_frontmatter += 1

        # Tags
        fm_tags = extract_tags_from_frontmatter(fm) if fm else []
        inline_tags = extract_inline_tags(content)
        combined_tags = list(set(fm_tags + inline_tags))

        if combined_tags:
            files_with_tags += 1

        for t in fm_tags:
            all_tags[t] += 1
            tag_to_files[t].append(str(rel_path))
        for t in inline_tags:
            all_inline_tags[t] += 1
            if t not in fm_tags:
                tag_to_files[t].append(str(rel_path))

        # Wiki links
        wiki_links = extract_wiki_links(content)
        if wiki_links:
            files_with_wiki_links += 1
        for link in wiki_links:
            all_wiki_links[link] += 1
            link_targets[link].append(str(rel_path))

        # External links
        ext_links = extract_external_links(content)
        for text, url in ext_links:
            try:
                domain = url.split("/")[2]
                all_external_domains[domain] += 1
            except IndexError:
                pass

        # Entities
        entities = detect_entities(content)
        for etype, eset in entities.items():
            for entity in eset:
                entity_mentions[etype][entity].append(str(rel_path))

        # Word count
        wc = len(content.split())
        total_words += wc
        word_counts[top_dir] += wc

        # Headings
        headings = extract_headings(content)

        files_data.append({
            "path": str(rel_path),
            "top_dir": top_dir,
            "sub_dir": sub_dir,
            "empty": False,
            "words": wc,
            "has_frontmatter": has_fm,
            "fm_tags": fm_tags,
            "inline_tags": inline_tags,
            "wiki_links": wiki_links,
            "ext_link_count": len(ext_links),
            "entity_count": sum(len(v) for v in entities.values()),
            "heading_count": len(headings),
            "title": fm.get("title", filepath.stem),
        })

    # Identify orphans: files never targeted by any wiki-link
    all_stems = {Path(fd["path"]).stem for fd in files_data}
    linked_stems = set(all_wiki_links.keys())
    orphan_stems = all_stems - linked_stems

    # Find potential link candidates: file stems that appear as text in other files
    potential_links = defaultdict(list)
    file_stems_by_name = {}
    for fd in files_data:
        stem = Path(fd["path"]).stem
        if len(stem) > 3 and stem not in ("Untitled",):
            file_stems_by_name[stem] = fd["path"]

    # For efficiency, only check non-trivial stems
    for filepath in iter_md_files(VAULT_PATH):
        rel_path = str(filepath.relative_to(VAULT_PATH))
        try:
            content = filepath.read_text(encoding="utf-8", errors="replace")
        except Exception:
            continue
        current_stem = filepath.stem
        current_links = set(extract_wiki_links(content))
        for stem, spath in file_stems_by_name.items():
            if stem == current_stem:
                continue
            if stem in current_links:
                continue
            # Check if the stem appears as a word in the content
            if re.search(r'\b' + re.escape(stem) + r'\b', content, re.IGNORECASE):
                potential_links[stem].append(rel_path)

    return {
        "files_data": files_data,
        "all_tags": all_tags,
        "all_inline_tags": all_inline_tags,
        "all_wiki_links": all_wiki_links,
        "all_external_domains": all_external_domains,
        "dir_counts": dir_counts,
        "subdir_counts": subdir_counts,
        "empty_files": empty_files,
        "files_with_frontmatter": files_with_frontmatter,
        "files_with_tags": files_with_tags,
        "files_with_wiki_links": files_with_wiki_links,
        "entity_mentions": entity_mentions,
        "link_targets": link_targets,
        "tag_to_files": tag_to_files,
        "word_counts": word_counts,
        "total_words": total_words,
        "orphan_stems": orphan_stems,
        "potential_links": potential_links,
        "total_files": len(files_data),
    }


def generate_report(data: dict) -> str:
    """Generate the diagnostic markdown report."""
    lines = []
    w = lines.append

    w("---")
    w(f"date: {datetime.now().strftime('%Y-%m-%d')}")
    w("type: vault-diagnostic")
    w("tags: [vault-organizer, diagnostic]")
    w("---")
    w("")
    w("# Obsidian Vault Diagnostic Inventory")
    w(f"*Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}*")
    w(f"*Vault: {VAULT_PATH}*")
    w("")

    # ── Overview ──
    w("## 1. Overview")
    w("")
    w(f"| Metric | Value |")
    w(f"|---|---|")
    w(f"| Total .md files | {data['total_files']} |")
    w(f"| Total words | {data['total_words']:,} |")
    w(f"| Files with frontmatter | {data['files_with_frontmatter']} ({data['files_with_frontmatter']*100//data['total_files']}%) |")
    w(f"| Files with any tags | {data['files_with_tags']} ({data['files_with_tags']*100//data['total_files']}%) |")
    w(f"| Files with wiki-links | {data['files_with_wiki_links']} ({data['files_with_wiki_links']*100//data['total_files']}%) |")
    w(f"| Empty files | {len(data['empty_files'])} |")
    w(f"| Orphan files (never linked to) | {len(data['orphan_stems'])} |")
    w("")

    # ── Directory breakdown ──
    w("## 2. Directory Breakdown")
    w("")
    w("| Directory | Files | Words | Avg words/file |")
    w("|---|---|---|---|")
    for d, count in sorted(data["dir_counts"].items(), key=lambda x: -x[1]):
        wc = data["word_counts"].get(d, 0)
        avg = wc // count if count > 0 else 0
        w(f"| {d} | {count} | {wc:,} | {avg} |")
    w("")

    # Subdirectory detail
    w("### Subdirectory Detail")
    w("")
    for d, count in sorted(data["subdir_counts"].items(), key=lambda x: -x[1]):
        if count >= 3:
            w(f"- **{d}/** — {count} files")
    w("")

    # ── Tag Inventory ──
    w("## 3. Tag Inventory")
    w("")
    combined_tags = Counter()
    for t, c in data["all_tags"].items():
        combined_tags[t] += c
    for t, c in data["all_inline_tags"].items():
        combined_tags[t] += c

    if combined_tags:
        w("### All Tags (frontmatter + inline)")
        w("")
        w("| Tag | Occurrences | Source |")
        w("|---|---|---|")
        for tag, count in combined_tags.most_common():
            source = []
            if tag in data["all_tags"]:
                source.append(f"fm×{data['all_tags'][tag]}")
            if tag in data["all_inline_tags"]:
                source.append(f"inline×{data['all_inline_tags'][tag]}")
            w(f"| `{tag}` | {count} | {', '.join(source)} |")
        w("")

        w("### Tag → Files Mapping")
        w("")
        for tag, files in sorted(data["tag_to_files"].items(), key=lambda x: -len(x[1])):
            if len(files) >= 2:
                w(f"**`{tag}`** ({len(files)} files)")
                for f in sorted(files)[:10]:
                    w(f"  - {f}")
                if len(files) > 10:
                    w(f"  - ...and {len(files)-10} more")
                w("")
    else:
        w("*No tags found in the vault.*")
        w("")

    # ── Wiki-Link Inventory ──
    w("## 4. Wiki-Link Inventory")
    w("")
    if data["all_wiki_links"]:
        w("### Most-Linked Targets")
        w("")
        w("| Target | Times Linked | Linked From |")
        w("|---|---|---|")
        for target, count in data["all_wiki_links"].most_common(40):
            sources = data["link_targets"].get(target, [])
            source_dirs = set()
            for s in sources:
                parts = Path(s).parts
                source_dirs.add(parts[0] if len(parts) > 1 else "(root)")
            w(f"| [[{target}]] | {count} | {', '.join(sorted(source_dirs))} |")
        w("")
    else:
        w("*No wiki-links found.*")
        w("")

    # ── External Links ──
    w("## 5. External Links (Top Domains)")
    w("")
    if data["all_external_domains"]:
        w("| Domain | Links |")
        w("|---|---|")
        for domain, count in data["all_external_domains"].most_common(25):
            w(f"| {domain} | {count} |")
        w("")

    # ── Entity Map ──
    w("## 6. Entity Map")
    w("")
    for etype in ["people", "places", "organizations"]:
        entities = data["entity_mentions"].get(etype, {})
        if entities:
            w(f"### {etype.title()}")
            w("")
            w(f"| Entity | Mentioned In (# files) | Directories |")
            w(f"|---|---|---|")
            for entity, files in sorted(entities.items(), key=lambda x: -len(x[1])):
                dirs = set()
                for f in files:
                    parts = Path(f).parts
                    dirs.add(parts[0] if len(parts) > 1 else "(root)")
                w(f"| {entity} | {len(files)} | {', '.join(sorted(dirs))} |")
            w("")

    # ── Empty Files ──
    w("## 7. Empty Files (candidates for cleanup)")
    w("")
    if data["empty_files"]:
        for f in sorted(data["empty_files"]):
            w(f"- `{f}`")
    else:
        w("*None*")
    w("")

    # ── Orphan Files ──
    w("## 8. Orphan Analysis")
    w(f"")
    w(f"**{len(data['orphan_stems'])} files** are never the target of a `[[wiki-link]]` from any other file.")
    w("")
    w("Top orphans by word count (significant content that's isolated):")
    w("")
    orphan_data = [fd for fd in data["files_data"]
                   if Path(fd["path"]).stem in data["orphan_stems"]
                   and not fd.get("empty", False)]
    orphan_data.sort(key=lambda x: -x["words"])
    w("| File | Words | Directory |")
    w("|---|---|---|")
    for fd in orphan_data[:50]:
        w(f"| {Path(fd['path']).stem} | {fd['words']:,} | {fd['top_dir']} |")
    w("")

    # ── Potential Link Opportunities ──
    w("## 9. Connection Opportunities")
    w("")
    w("Files whose **name appears as text** in other files but is NOT linked with `[[...]]`.")
    w("These are the strongest candidates for new wiki-links.")
    w("")

    sorted_potentials = sorted(data["potential_links"].items(),
                                key=lambda x: -len(x[1]))
    if sorted_potentials:
        w("| Note Name | Mentioned In (# files) | Sample Sources |")
        w("|---|---|---|")
        for stem, sources in sorted_potentials[:60]:
            sample = ", ".join(Path(s).stem for s in sources[:3])
            if len(sources) > 3:
                sample += f" +{len(sources)-3} more"
            w(f"| **{stem}** | {len(sources)} | {sample} |")
        w("")
    else:
        w("*No unlinked mentions found.*")
        w("")

    # ── Cross-Directory Connections ──
    w("## 10. Cross-Directory Link Map")
    w("")
    w("Existing wiki-links that cross directory boundaries:")
    w("")
    cross_links = []
    for fd in data["files_data"]:
        if fd.get("wiki_links"):
            src_dir = fd["top_dir"]
            for link in fd["wiki_links"]:
                # Find the target file's directory
                for fd2 in data["files_data"]:
                    if Path(fd2["path"]).stem == link and fd2["top_dir"] != src_dir:
                        cross_links.append((fd["path"], link, src_dir, fd2["top_dir"]))
    if cross_links:
        w("| Source | Target | From → To |")
        w("|---|---|---|")
        for src, tgt, sdir, tdir in cross_links:
            w(f"| {Path(src).stem} | [[{tgt}]] | {sdir} → {tdir} |")
    else:
        w("*No cross-directory links found.* This is a major structural gap.")
    w("")

    # ── Proposed Tag Taxonomy ──
    w("## 11. Proposed Tag Taxonomy")
    w("")
    w("Based on content analysis, here's a suggested standardized tag system:")
    w("")
    w("### By Area")
    w("```")
    w("#area/conservación    — Conservation programs, monitoring, biodiversity")
    w("#area/arte            — Art programs, residencies, exhibitions")
    w("#area/evaluación      — Program evaluation, impact assessment")
    w("#area/educación       — Docente Activo, learning programs")
    w("#area/fondo           — Fundraising, grants, financial")
    w("#area/tech            — Data pipeline, platform, classifiers")
    w("#area/personal        — Personal writing, reflections")
    w("```")
    w("")
    w("### By Type")
    w("```")
    w("#type/meeting         — Meeting notes, conversations")
    w("#type/research        — Research notes, literature review")
    w("#type/project         — Project documentation")
    w("#type/session-log     — Claude Code session logs")
    w("#type/journal         — Daily journal entries")
    w("#type/reference       — Reference material, guides")
    w("#type/creative        — Poetry, creative writing")
    w("#type/contract        — Legal documents, agreements")
    w("```")
    w("")
    w("### By Project")
    w("```")
    w("#project/plataforma-territorial")
    w("#project/camera-traps")
    w("#project/data-pipeline")
    w("#project/species-classifier")
    w("#project/literature-agent")
    w("#project/schedule-agent")
    w("#project/visualizaciones")
    w("#project/smart-forest")
    w("#project/adn-ambiental")
    w("#project/docente-activo")
    w("```")
    w("")
    w("### By Status")
    w("```")
    w("#status/active        — Currently being worked on")
    w("#status/paused        — On hold")
    w("#status/completed     — Done")
    w("#status/idea          — Not started, exploratory")
    w("```")
    w("")

    # ── Summary & Recommendations ──
    w("## 12. Summary & Recommendations")
    w("")
    w("### Health Score")
    w("")
    total = data["total_files"]
    tag_pct = data["files_with_tags"] * 100 // total
    link_pct = data["files_with_wiki_links"] * 100 // total
    fm_pct = data["files_with_frontmatter"] * 100 // total
    orphan_pct = len(data["orphan_stems"]) * 100 // total

    w(f"| Metric | Score | Target |")
    w(f"|---|---|---|")
    w(f"| Frontmatter coverage | {fm_pct}% | >90% |")
    w(f"| Tag coverage | {tag_pct}% | >80% |")
    w(f"| Wiki-link coverage | {link_pct}% | >60% |")
    w(f"| Orphan rate | {orphan_pct}% | <15% |")
    w("")

    w("### Priority Actions")
    w("")
    w("1. **Add frontmatter to all files** — date, tags, type at minimum")
    w("2. **Apply tag taxonomy** — standardize existing tags + tag untagged files")
    w("3. **Convert text mentions to wiki-links** — Section 9 lists the top opportunities")
    w("4. **Create hub notes** — for top entities (people, projects, places) that appear across directories")
    w("5. **Clean up empty files** — delete or populate the " + str(len(data["empty_files"])) + " empty stubs")
    w("6. **Bridge the silos** — Journal ↔ FMA ↔ SecondBrain need cross-links")
    w("")

    return "\n".join(lines)


def main():
    print("Scanning vault...")
    data = scan_vault()
    print(f"Scanned {data['total_files']} files")

    print("Generating report...")
    report = generate_report(data)

    OUTPUT_PATH.write_text(report, encoding="utf-8")
    print(f"Report written to: {OUTPUT_PATH}")

    # Also save raw data as JSON for future scripts
    json_path = Path(__file__).parent / "vault_diagnostic_data.json"
    json_data = {
        "total_files": data["total_files"],
        "total_words": data["total_words"],
        "files_with_frontmatter": data["files_with_frontmatter"],
        "files_with_tags": data["files_with_tags"],
        "files_with_wiki_links": data["files_with_wiki_links"],
        "empty_files": data["empty_files"],
        "orphan_stems": list(data["orphan_stems"]),
        "all_tags": dict(data["all_tags"]),
        "all_inline_tags": dict(data["all_inline_tags"]),
        "all_wiki_links": dict(data["all_wiki_links"]),
        "dir_counts": dict(data["dir_counts"]),
        "potential_links": {k: v for k, v in list(
            sorted(data["potential_links"].items(), key=lambda x: -len(x[1]))
        )[:100]},
        "entity_mentions": {
            etype: {entity: files for entity, files in ents.items()}
            for etype, ents in data["entity_mentions"].items()
        },
        "files_data": [
            {
                "path": fd["path"],
                "top_dir": fd["top_dir"],
                "empty": fd.get("empty", False),
                "words": fd["words"],
                "has_frontmatter": fd.get("has_frontmatter", False),
                "fm_tags": fd.get("fm_tags", []),
                "inline_tags": fd.get("inline_tags", []),
                "wiki_links": fd.get("wiki_links", []),
            }
            for fd in data["files_data"]
        ],
    }
    json_path.write_text(json.dumps(json_data, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Raw data written to: {json_path}")


if __name__ == "__main__":
    main()
