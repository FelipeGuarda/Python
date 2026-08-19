"""How a person and their organisation get packed into one spreadsheet cell.

Event forms ask for "Nombre y apellido - organización" in a single free-text
field, and respondents answer in at least six different shapes. This module
owns the delimiter zoo and — just as importantly — owns saying how much to
trust each split, so a caller can route the doubtful ones to a human instead
of silently guessing.

The same conflation happens one column over: asked for an organisation,
people answer with their job title, so `Organización` fills up with
"Directora Educación MIM". `split_cargo` owns pulling those apart, and lives
here rather than in its own module because it is the same knowledge — which
words name an organisation and which name a person's place in one.

Nothing here writes anything. Both entry points are pure functions of a cell.
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

# Words naming a *role*, not an organisation. Contact lists collect these
# because the person filling the form answers "what are you" when asked
# "where are you from" — so a cargo lands in the organisation cell.
#
# Deliberately excluded, though they look similar: "dirección",
# "departamento" and "secretaría" are in ORG_HINTS above, because they name
# a unit rather than the person heading it. Also excluded: "editor" and
# "depto", which in this list appear as parts of a business or unit name
# ("librería naturaleza, editores", "SBAP- Depto Fondo e IECB").
CARGO_HINTS = frozenset({
    "director", "directora", "subdirector", "subdirectora", "jefe", "jefa",
    "encargado", "encargada", "coordinador", "coordinadora", "curador",
    "curadora", "presidente", "presidenta", "gerente", "gerenta",
    "profesor", "profesora", "investigador", "investigadora", "asesor",
    "asesora", "analista", "academico",
    "academica", "antropologo", "antropologa", "biologo", "biologa",
    "artista", "deportista", "escalador", "escaladora",
})

# Words that carry a cargo forward without being one: "Jefa *de Área*
# Educación", "Director *del* Museo". A run of these continues the role.
_CARGO_GLUE = frozenset({
    "de", "del", "la", "las", "los", "el", "y", "e", "en", "area", "areas",
    "division", "divisiones", "unidad", "curatorial", "cientifico",
    "cientifica", "ejecutivo", "ejecutiva", "general", "nacional", "regional",
    "adjunto", "adjunta", "interino", "interina",
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


@dataclass(frozen=True)
class CargoSplit:
    """An `Organización` cell with the job title taken back out."""

    organizacion: str  # what remains once the role is lifted out; "" if nothing does
    cargo: str         # the role text, for the reviewer to park in Notas
    confianza: str     # "alta" | "media" | "baja"
    motivo: str
    original: str

    @property
    def has_cargo(self) -> bool:
        return bool(self.cargo)

    @property
    def needs_review(self) -> bool:
        return self.confianza != "alta"


def _is_cargo_token(token: str) -> bool:
    return _fold(token) in CARGO_HINTS


def _cargo_run_length(tokens: list[str]) -> int:
    """How many leading tokens belong to the role.

    A run starts on a cargo word and continues through glue ("de", "Área",
    "y") and further cargo words. It stops at the first token that names an
    organisation, because that token is where the role ends and the
    organisation begins — "Jefe curatorial y científico | MNHN".

    What it cannot do is tell "Directora Educación | MIM" from "Coordinador |
    Centros UC": both are a role, a common noun, and an acronym, and only a
    human knows which side the middle word belongs to. Such cells come back
    at `media` rather than being guessed at.
    """
    if not tokens or not _is_cargo_token(tokens[0]):
        return 0
    length = 1
    while length < len(tokens):
        token = tokens[length]
        if _fold(token) in ORG_HINTS or _ACRONYM.match(token):
            break
        if _is_cargo_token(token) or _fold(token) in _CARGO_GLUE:
            length += 1
            continue
        break
    return length


def _remainder_is_org(text: str) -> bool:
    """Looser than `_looks_like_org`: an acronym *token* counts.

    `_looks_like_org` only recognises a cell that is nothing but an acronym,
    which is right when classifying a whole half. Here the remainder is known
    to be an organisation's name already — "WWF Chile", "Centros UC" — and the
    question is only how much to trust it.
    """
    return _looks_like_org(text) or any(_ACRONYM.match(t) for t in text.split())


def split_cargo(organizacion: str) -> CargoSplit:
    """Separate a job title from the organisation it was typed into.

    Two shapes occur, and nothing else does in this list:

    - the organisation first, then the role after a delimiter
      ("SBAP- Jefa División Biodiversidad", "Patagonia - deportista")
    - the role first, then the organisation ("Directora Museo Violeta Parra")

    A cell with no role word comes back unchanged with `cargo` empty, so a
    caller can treat "nothing to do" and "nothing recognised" alike:

    >>> split_cargo("Museo Taller").has_cargo
    False

    `confianza` is **alta** when what remains is recognisably an
    organisation, **media** when it merely looks like a proper name, and
    **baja** when the role consumed the whole cell and no organisation is
    left — a case a human must confirm, since the cell may name a freelance
    occupation rather than an employer.
    """
    original = str(organizacion or "")
    text = _clean(original)
    if not text:
        return CargoSplit("", "", "alta", "celda vacía", original)

    for delimiter in _SPACED + _RAGGED:
        if delimiter not in text:
            continue
        left, _, right = text.partition(delimiter)
        left, right = _clean(left), _clean(right)
        if not (left and right):
            continue
        # Only the organisation-then-role shape is handled here; a role on the
        # left is the prefix case and falls through to the token scan.
        if _cargo_run_length(right.split()) and not _is_cargo_token(left.split()[0]):
            confianza = "alta" if _remainder_is_org(left) else "media"
            return CargoSplit(left, right, confianza,
                              f"cargo tras '{delimiter.strip()}'", original)

    tokens = text.split()
    run = _cargo_run_length(tokens)
    if not run:
        return CargoSplit(text, "", "alta", "sin cargo", original)

    cargo = " ".join(tokens[:run])
    remainder = " ".join(tokens[run:])

    # "Curadora de arte" — an occupation, not a place. One leftover lowercase
    # word that names no organisation is not an employer, so the whole cell is
    # handed back as the cargo. Nothing is dropped; a human decides where it
    # goes, because a freelance occupation and a missing employer look alike.
    bare = remainder.split()
    if not remainder or (len(bare) == 1 and bare[0].islower() and not _remainder_is_org(remainder)):
        return CargoSplit("", text, "baja", "sólo cargo, sin organización", original)

    # An acronym *somewhere* in the remainder is not enough to trust the
    # boundary: "Directora | Educación MIM" leaves an acronym behind and is
    # still cut in the wrong place. Only a remainder that *starts* like an
    # organisation earns alta; anything else is the ambiguous middle word.
    head = bare[0]
    starts_like_org = _fold(head) in ORG_HINTS or bool(_ACRONYM.match(head))
    confianza = "alta" if starts_like_org and _remainder_is_org(remainder) else "media"
    return CargoSplit(remainder, cargo, confianza, f"cargo inicial '{cargo}'", original)
