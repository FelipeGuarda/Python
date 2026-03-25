# Claude Code — Dev/Python Repo Instructions

## Branch Check Protocol (MANDATORY)

Before starting ANY development work, check all git branches:

```bash
git branch -a
git log --oneline --graph --all | head -40
```

For any branch related to the current task:
1. Check what it contains: `git log main..<branch> --oneline`
2. Check if it's finished or WIP (read session logs mentioning it)
3. **TELL THE USER** what you found before writing any code
4. If there's unmerged work, discuss merging it first

**Never assume `main` has the latest work.** Feature branches may be ahead.

## Two-Machine Split

- **Home (Linux):** data-pipeline, plataforma-territorial (frontend + backend except Phase 3), literatura-agent, visualizaciones-artisticas
- **Office (Windows):** camera-traps, species-classifier, plataforma-territorial Phase 3 only (camera trap ingestion)

## Language

All user-facing platform text must be in Spanish.
