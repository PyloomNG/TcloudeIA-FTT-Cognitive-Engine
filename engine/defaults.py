"""
Conjuntos predeterminados (Pilar 3: conocimiento base del motor).

El motor FTT declara por defecto las bibliotecas de letras y números,
que representan conocimiento "innato" sobre el alfabeto y los dígitos.
El usuario puede:
    - Usarlas directamente (engine.is_in(c, "VOWELS") == True).
    - Extenderlas con declare_library(parents=("LETTERS",)).
    - Eliminarlas o modificarlas si lo desea.

Principio SOLID aplicado:
    - SRP: este módulo SOLO define los átomos base; no entrena.
    - OCP: añadir nuevos defaults (vocablos básicos, símbolos
      matemáticos) es extender este módulo sin tocar el motor.
"""

from __future__ import annotations

from .atom import Atom, AtomDomain
from .library import Library


# Letras mayúsculas: A-Z (26 átomos)
UPPERCASE_LETTERS: tuple[str, ...] = tuple(chr(c) for c in range(ord("A"), ord("Z") + 1))

# Letras minúsculas: a-z (26 átomos)
LOWERCASE_LETTERS: tuple[str, ...] = tuple(chr(c) for c in range(ord("a"), ord("z") + 1))

# Dígitos: 0-9 (10 átomos)
DIGITS: tuple[str, ...] = tuple(str(d) for d in range(10))

# Vocales (mayúsculas y minúsculas)
VOWELS_UPPER: tuple[str, ...] = ("A", "E", "I", "O", "U")
VOWELS_LOWER: tuple[str, ...] = ("a", "e", "i", "o", "u")
VOWELS: tuple[str, ...] = VOWELS_UPPER + VOWELS_LOWER

# Consonantes mayúsculas (todas las letras menos vocales y Ñ)
CONSONANTS_UPPER: tuple[str, ...] = tuple(
    c for c in UPPERCASE_LETTERS if c not in VOWELS_UPPER
)


def default_letter_atoms() -> list[Atom]:
    """Átomos pre-declarados para letras (mayúsculas y minúsculas)."""
    return [Atom(c, AtomDomain.TEXT) for c in (UPPERCASE_LETTERS + LOWERCASE_LETTERS)]


def default_digit_atoms() -> list[Atom]:
    """Átomos pre-declarados para dígitos 0-9."""
    return [Atom(d, AtomDomain.NUMBER) for d in DIGITS]


def default_libraries() -> dict[str, Library]:
    """Devuelve el conjunto de bibliotecas pre-declaradas.

    Jerarquía:
        LETTERS (todas)
          ├── LETTERS_UPPER
          │     └── VOWELS_UPPER
          │     └── CONSONANTS_UPPER
          └── LETTERS_LOWER
                └── VOWELS_LOWER
        DIGITS
    """
    libs: dict[str, Library] = {}
    # LETTERS: union de upper + lower
    libs["LETTERS"] = Library(
        name="LETTERS",
        members=frozenset(),  # se rellenará con las sub-bibliotecas por transitividad
        measure=0,
        parents=(),
        description="Conjunto de todas las letras del alfabeto (A-Z, a-z).",
        context={"domain": "alphabet", "cardinality": 52},
    )
    # LETTERS_UPPER
    upper_atoms = [Atom(c, AtomDomain.TEXT) for c in UPPERCASE_LETTERS]
    libs["LETTERS_UPPER"] = Library(
        name="LETTERS_UPPER",
        members=frozenset(),  # placeholder, será completado por la fachada
        measure=0,
        parents=("LETTERS",),
        description="Letras mayúsculas A-Z (26 símbolos).",
        context={"domain": "alphabet", "cardinality": 26},
    )
    # LETTERS_LOWER
    lower_atoms = [Atom(c, AtomDomain.TEXT) for c in LOWERCASE_LETTERS]
    libs["LETTERS_LOWER"] = Library(
        name="LETTERS_LOWER",
        members=frozenset(),
        measure=0,
        parents=("LETTERS",),
        description="Letras minúsculas a-z (26 símbolos).",
        context={"domain": "alphabet", "cardinality": 26},
    )
    # VOWELS_UPPER ⊂ LETTERS_UPPER
    libs["VOWELS_UPPER"] = Library(
        name="VOWELS_UPPER",
        members=frozenset(),
        measure=0,
        parents=("LETTERS_UPPER",),
        description="Vocales mayúsculas: A, E, I, O, U.",
        context={"domain": "alphabet", "cardinality": 5},
    )
    # VOWELS_LOWER ⊂ LETTERS_LOWER
    libs["VOWELS_LOWER"] = Library(
        name="VOWELS_LOWER",
        members=frozenset(),
        measure=0,
        parents=("LETTERS_LOWER",),
        description="Vocales minúsculas: a, e, i, o, u.",
        context={"domain": "alphabet", "cardinality": 5},
    )
    # CONSONANTS_UPPER
    libs["CONSONANTS_UPPER"] = Library(
        name="CONSONANTS_UPPER",
        members=frozenset(),
        measure=0,
        parents=("LETTERS_UPPER",),
        description="Consonantes mayúsculas: las 21 letras que no son vocales.",
        context={"domain": "alphabet", "cardinality": 21},
    )
    # DIGITS
    libs["DIGITS"] = Library(
        name="DIGITS",
        members=frozenset(),
        measure=0,
        parents=(),
        description="Dígitos decimales 0-9.",
        context={"domain": "numerals", "cardinality": 10},
    )
    return libs


def default_atoms_by_library() -> dict[str, list[Atom]]:
    """Mapea nombre de biblioteca -> lista de átomos que pertenecen a ella
    (sin considerar transitividad)."""
    return {
        "LETTERS_UPPER": [Atom(c, AtomDomain.TEXT) for c in UPPERCASE_LETTERS],
        "LETTERS_LOWER": [Atom(c, AtomDomain.TEXT) for c in LOWERCASE_LETTERS],
        "VOWELS_UPPER": [Atom(c, AtomDomain.TEXT) for c in VOWELS_UPPER],
        "VOWELS_LOWER": [Atom(c, AtomDomain.TEXT) for c in VOWELS_LOWER],
        "CONSONANTS_UPPER": [Atom(c, AtomDomain.TEXT) for c in CONSONANTS_UPPER],
        "DIGITS": [Atom(d, AtomDomain.NUMBER) for d in DIGITS],
    }
