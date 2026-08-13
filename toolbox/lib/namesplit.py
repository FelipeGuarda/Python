"""How a person and their organisation get packed into one spreadsheet cell.

Event forms ask for "Nombre y apellido - organización" in a single free-text
field, and respondents answer in at least six different shapes. This module
owns the delimiter zoo and — just as importantly — owns saying how much to
trust each split, so a caller can route the doubtful ones to a human instead
of silently guessing.

Nothing here writes anything. `split_contact` is a pure function of the cell.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass

# Words that mark the start of an organisation name. Their value is not just
# classification: in a cell with no delimiter at all ("Sergio Benavides
# Ministerio del Medio Ambiente") the first hint word IS the boundary.
ORG_HINTS = frozenset({
    "fundacion", "fundación", "universidad", "ministerio", "servicio",
    "centro", "museo", "instituto", "corporacion", "corporación", "ong",
    "parque", "red", "observatorio", "consultora", "independiente",
    "asociacion", "asociación", "agrupacion", "agrupación", "comunidad",
    "sociedad", "facultad", "escuela", "colegio", "programa", "proyecto",
    "cooperativa", "comite", "comité", "consejo", "secretaria", "secretaría",
    "direccion", "dirección", "departamento", "subsecretaria", "subsecretaría",
    "laboratorio", "estacion", "estación", "reserva", "jardin", "jardín",
})

# Acronym organisations are common here (ROC, WCS, SBAP, IEB, PNUD, FMA, UC).
_ACRONYM = re.compile(r"^[A-ZÁÉÍÓÚÑ]{2,6}$")

# Ordered by how much confidence the match earns. A delimiter padded with
# spaces was deliberate; a bare hyphen might be a hyphenated surname.
_SPACED = (" - ", " / ", " | ", " — ", " – ")
_RAGGED = ("- ", " -", "/ ", " /", "-", "/", "|")

# Trailing decoration respondents add: ":)", "( )", stray punctuation.
_JUNK = re.compile(r"[\s:;,.\-–—/|]+$|^[\s:;,.\-–—/|]+")
_EMOTICON = re.compile(r"[:;=][-^]?[)(DPp]+|[\U0001F300-\U0001FAFF]")


@dataclass(frozen=True)
class Split:
    """One packed cell, taken apart."""

    nombre: str
    organizacion: str
    confianza: str  # "alta" | "media" | "baja"
    motivo: str     # why it landed at that confidence, for the reviewer
    original: str

    @property
    def needs_review(self) -> bool:
        return self.confianza != "alta"


def _clean(text: str) -> str:
    text = unicodedata.normalize("NFC", str(text or ""))
    text = _EMOTICON.sub("", text)
    text = re.sub(r"\s+", " ", text)
    return _JUNK.sub("", text).strip()


def _fold(token: str) -> str:
    """Token lowercased and accent-stripped, for hint comparison."""
    folded = unicodedata.normalize("NFKD", token.lower())
    return "".join(c for c in folded if not unicodedata.combining(c))


def _looks_like_org(text: str) -> bool:
    if not text:
        return False
    if _ACRONYM.match(text.strip()):
        return True
    return any(_fold(t) in ORG_HINTS for t in text.split())


def _looks_like_person(text: str) -> bool:
    tokens = text.split()
    if not 1 < len(tokens) <= 5:
        return False
    if any(_fold(t) in ORG_HINTS for t in tokens):
        return False
    return not text.isupper()


def _first_hint_index(tokens: list[str]) -> int | None:
    for index, token in enumerate(tokens):
        if _fold(token) in ORG_HINTS:
            return index
    return None


def split_contact(packed: str) -> Split:
    """Take apart a "name - organisation" cell.

    Returns a `Split` whose `confianza` says how much to trust it:

    - **alta** — a spaced delimiter separated two plausible halves, or the
      whole cell is plainly just a person's name.
    - **media** — the boundary was inferred: a bare hyphen with an
      organisation-looking right side, or a cell with no delimiter where an
      organisation keyword marks where the name ends.
    - **baja** — no delimiter and no keyword. The caller must ask a human;
      the whole string is returned as `nombre` so nothing is lost.
    """
    original = str(packed or "")
    text = _clean(original)
    if not text:
        return Split("", "", "baja", "celda vacía", original)

    for delimiter in _SPACED:
        if delimiter in text:
            left, _, right = text.partition(delimiter)
            left, right = _clean(left), _clean(right)
            if left and right:
                if _looks_like_person(left) or _looks_like_org(right):
                    return Split(left, right, "alta", f"separador '{delimiter.strip()}'", original)
                return Split(left, right, "media", "separador claro, mitades dudosas", original)
            # A delimiter with nothing on one side: treat the survivor below.
            text = left or right

    for delimiter in _RAGGED:
        if delimiter in text:
            left, _, right = text.partition(delimiter)
            left, right = _clean(left), _clean(right)
            if not (left and right):
                continue
            if _looks_like_org(right):
                return Split(left, right, "media", f"separador irregular '{delimiter.strip()}'", original)
            return Split(text, "", "baja",
                         f"'{delimiter.strip()}' podría ser apellido compuesto", original)

    tokens = text.split()
    hint = _first_hint_index(tokens)
    if hint == 0:
        return Split("", text, "media", "parece sólo organización", original)
    if hint is not None:
        return Split(" ".join(tokens[:hint]), " ".join(tokens[hint:]), "media",
                     f"sin separador; '{tokens[hint]}' marca el límite", original)

    if len(tokens) <= 3:
        return Split(text, "", "alta" if len(tokens) > 1 else "media",
                     "sólo nombre, sin organización", original)

    return Split(text, "", "baja", "sin separador ni palabra clave: revisar", original)
