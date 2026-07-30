"""Boundary layer between external camera-trap file formats and FMA analysis code.

Each module here owns exactly one external format or convention, so that a change
in a vendor export touches one file rather than every consumer:

    stations      the FMA station-naming convention (CT01..CT27) + legacy aliases
    observations  the canonical observation table written once at ingest

See the DESIGN_NOTES section of the project README for the boundary this enforces.
"""
