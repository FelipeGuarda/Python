#!/usr/bin/env python3
"""
verify_order.py — check capture order against the datetimes, where no DCIM manifest exists.

WHY THIS IS A SCRIPT AND NOT PART OF THE DIAGNOSIS
    `camtrap/clocks.py` establishes capture order from the filename counter, and from
    `dcim_manifest.csv` when the counter collides (the counter restarts inside each DCIM
    folder, so flattening pools many `xxxx0001.JPG` into one directory). Three
    station-campaigns have colliding counters and no manifest, because they were
    flattened before the manifest existed or came out of the CT22/CT23 un-nesting:

        primavera_2025 CT23   1,735 frames    802 colliding counters
        otono_2026     CT15     999 frames    166 colliding counters
        otono_2026     CT08     873 frames     88 colliding counters

    For those, `diagnose()` reports "capture order not established, but no clock failure
    is detectable ... this is unverified, not verified clean" — and admits the frames.
    All three are `clock_clean` with 100% valid_date, so nothing is lost; what is missing
    is a CHECK.

    This script supplies that check. It is deliberately NOT wired into `diagnose()`:
    it derives order FROM the datetimes and then judges the datetimes, and making that
    circularity load-bearing in reset detection would be exactly the kind of heuristic
    precondition this project refuses. It can strengthen a verdict in a note; it must
    never admit a station the deterministic rule refused.

THE TEST, AND THE FALSIFIABLE PART
    DCIM folders are created in sequence, so within one folder the counter rises with
    real time. Sort every frame by datetime and cut a new folder each time a counter
    REPEATS; that reconstructs the folder boundaries. Then check the constraint:

        is the counter monotonically increasing inside every reconstructed block?

    A clock whose datetimes disagree with true capture order cannot satisfy that — the
    counters inside the blocks would jump around. So a pass is evidence; a failure is a
    positive finding.

    It cannot prove the absence of a reset. A clock set FORWARD by a constant, with no
    frames spanning the jump, leaves no trace here or anywhere else. What it rules out is
    a backwards reset (breaks monotonicity) and a factory reset (would show 2017 dates).

VALIDATED AGAINST GROUND TRUTH
    Run with --controls to re-check the stations that DO have a manifest. On 2026-08-20:

        primavera_2025 CT14   reconstructed 9 folders, manifest says 9   blocks 999,999,...
        otono_2025     CT20   reconstructed 3 folders, manifest says 3
        otono_2025     CT04   reconstructed 3 folders, manifest says 3, but FALSIFIED

    CT14 recovering nine folders at exactly the DCIM 999 cap, from datetimes alone, is
    the strongest evidence the method works. CT04's single backwards step is explained by
    its manifest: a MIXED structure (`M5`, `M5/100EK113`, `M5/101EK113`), files both loose
    in the card root and in DCIM subfolders, which breaks the one-folder-one-counter-run
    assumption. It is a limit of the method, not a clock fault — which is precisely why
    this stays a diagnostic.

Usage:
    python scripts/verify_order.py                     # the three manifest-less stations
    python scripts/verify_order.py --controls          # also re-check the known-truth ones
    python scripts/verify_order.py --campaign otono_2026 --station CT15
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from camtrap import clocks, exports

# ── LEGACY HISTORICAL LIST — NOT the live state ───────────────────────────────
#
# This is a FROZEN record of the three station-campaigns this script was written for
# in August 2026: colliding filename counters and no DCIM manifest. It is kept
# hand-maintained on purpose -- a station that later acquires a manifest should drop
# off it by a deliberate edit, so the list goes on recording which cases were ever
# affected. That makes it a historical record and NOT an answer to "which stations
# lack order evidence today".
#
# THE LIVE STATE IS `timestamps_audit.log`, section "Capture-order evidence, all N
# station(s)", regenerated on every `python timestamps.py --campaign <name>` run.
# Read that, never this, when you want the current picture.
#
# The two ALREADY DISAGREE, which is why the distinction is labelled rather than
# assumed. Measured 2026-08-25, order not established for:
#     otono_2025      CT04              (has a manifest, partially -- not in this list)
#     primavera_2025  CT22, CT23        (CT22 not in this list)
#     otono_2026      CT08, CT15, CT27  (CT27 not in this list)
# Those extra three are a different question from the one this script asks, so the
# list is not wrong -- but anyone reading it as the live state would be.
UNVERIFIED_LEGACY_2026_08 = [
    ("primavera_2025", "CT23"),
    ("otono_2026", "CT15"),
    ("otono_2026", "CT08"),
]
UNVERIFIED = UNVERIFIED_LEGACY_2026_08   # name kept so the CLI below is unchanged
# Stations WITH a manifest, used to check the method against ground truth.
CONTROLS = [
    ("primavera_2025", "CT14"),
    ("otono_2025", "CT20"),
    ("otono_2025", "CT04"),
]

CAMPAIGNS_ROOT = Path(__file__).resolve().parents[1] / "data" / "campaigns"


def _norm(s) -> str:
    return str(s).upper().replace("_", "")


def load_station(campaign: str, station: str) -> pd.DataFrame:
    total, _ = exports.read_total_export(CAMPAIGNS_ROOT / campaign)
    total["_dt"] = pd.to_datetime(total["DateTime"], errors="coerce")
    parsed = total["File"].map(lambda n: clocks.parse_filename(str(n)))
    total["_counter"] = [p[1] for p in parsed]
    g = total[total["Deployments"].map(_norm) == _norm(station)]
    return g[g["_counter"].notna() & g["_dt"].notna()].copy()


def reconstruct_folders(df: pd.DataFrame) -> pd.DataFrame:
    """Assign a folder index by cutting whenever a counter repeats in datetime order.

    mergesort because it is stable: frames sharing a datetime (a burst writes 2-3 in the
    same second) keep their input order rather than being permuted arbitrarily, which
    would manufacture a monotonicity failure out of a tie.
    """
    d = df.sort_values("_dt", kind="mergesort").copy()
    folder, seen, out = 0, set(), []
    for c in d["_counter"]:
        if c in seen:
            folder += 1
            seen = set()
        seen.add(c)
        out.append(folder)
    d["_folder"] = out
    return d


def check(df: pd.DataFrame) -> dict:
    d = reconstruct_folders(df)
    blocks, failures = [], []
    for f, blk in d.groupby("_folder"):
        cs = blk["_counter"].tolist()
        drops = sum(1 for a, b in zip(cs, cs[1:]) if b <= a)
        blocks.append(len(blk))
        if drops:
            failures.append((int(f), len(blk), drops))
    return {
        "n_frames": len(d),
        "n_colliding": int(d["_counter"].duplicated().sum()),
        "n_folders": int(d["_folder"].nunique()),
        "block_sizes": blocks,
        "failures": failures,
        "consistent": not failures,
        "dt_min": d["_dt"].min(),
        "dt_max": d["_dt"].max(),
    }


def manifest_folders(campaign: str, station: str) -> int | None:
    p = CAMPAIGNS_ROOT / campaign / clocks.DCIM_MANIFEST_FILENAME
    if not p.exists():
        return None
    m = pd.read_csv(p, dtype=str)
    m = m[m["deployment"].map(_norm) == _norm(station)]
    return None if m.empty else int(m["dcim_folder"].nunique())


def report(campaign: str, station: str) -> bool:
    df = load_station(campaign, station)
    if df.empty:
        print(f"{campaign} {station}: no frames with a parseable counter — nothing to check")
        return True
    r = check(df)
    truth = manifest_folders(campaign, station)

    print(f"\n{campaign} {station}")
    print(f"  {r['n_frames']:,} frames, {r['n_colliding']:,} colliding counters, "
          f"{r['dt_min']:%Y-%m-%d} to {r['dt_max']:%Y-%m-%d}")
    print(f"  reconstructed folders : {r['n_folders']}   sizes {r['block_sizes'][:10]}"
          + (" ..." if len(r["block_sizes"]) > 10 else ""))
    if truth is not None:
        agree = "AGREES" if truth == r["n_folders"] else "DISAGREES"
        print(f"  manifest says         : {truth} folder(s)  -> {agree}")
    if r["consistent"]:
        print("  counter monotone in every block: YES — consistent with sequential DCIM "
              "folders and a well-behaved clock")
    else:
        print("  counter monotone in every block: NO — the datetimes disagree with "
              "capture order:")
        for f, n, drops in r["failures"]:
            print(f"      block {f}: {n} frames, {drops} backwards step(s)")
    return r["consistent"]


def main(argv=None) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1].strip())
    ap.add_argument("--controls", action="store_true",
                    help="also re-check stations that HAVE a manifest, against its folder count")
    ap.add_argument("--campaign", help="check one campaign (with --station)")
    ap.add_argument("--station", help="check one station (with --campaign)")
    args = ap.parse_args(argv)

    if args.campaign and args.station:
        targets, controls = [(args.campaign, args.station)], []
    else:
        targets, controls = UNVERIFIED, (CONTROLS if args.controls else [])

    print("=" * 74)
    print("Stations with colliding counters and NO manifest — order checked against datetimes")
    print("=" * 74)
    ok = [report(c, s) for c, s in targets]

    if controls:
        print("\n" + "=" * 74)
        print("CONTROLS — a manifest exists, so the reconstruction can be checked")
        print("=" * 74)
        for c, s in controls:
            report(c, s)

    print()
    if all(ok):
        print(f"All {len(ok)} unverified station(s) CONSISTENT — not falsified.")
        print("This strengthens the existing verdict; it does not replace it. A forward")
        print("clock jump with no frames spanning it would leave no trace here.")
        return 0
    print("At least one station FAILED the check — treat its datetimes as suspect and")
    print("record the finding in deployment_anchors.csv before using it.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
