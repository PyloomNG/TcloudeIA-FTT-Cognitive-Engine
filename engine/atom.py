"""
Átomos combinatorios agnósticos al dominio (Pilar 1 de Fodor).

Pilar 1 (Fodor): las representaciones mentales son estructuras combinatorias
portables. Para servir al mismo motor a texto, imágenes y números, la
identidad de un átomo NO puede ser un entero global (como en v1); debe
incluir su dominio. Sin esta precaución, el bigrama "er" en texto podría
colisionar con el código de borde "er" en una imagen.

Un `Atom` es la cara "humana" (forma de superficie + dominio). Un `AtomId`
es la cara "interna" (identidad inmutable usada como clave de diccionario).
El `AtomRegistry` es el puente bijectivo entre ambas caras *dentro* de
cada dominio.

Principio SOLID aplicado:
    - Single Responsibility: este módulo solo modela la identidad de átomos.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class AtomDomain(Enum):
    """Dominio del átomo. Mantiene los espacios de identificación disjuntos."""

    TEXT = "TEXT"
    IMAGE = "IMAGE"
    NUMBER = "NUMBER"
    MULTIMODAL = "MULTIMODAL"


@dataclass(frozen=True)
class AtomId:
    """Identidad inmutable de un átomo: (dominio, id local al dominio).

    La biyección se garantiza DENTRO de cada dominio. Cruzar dominios con
    el mismo `local_id` no representa ninguna equivalencia semántica.
    """

    domain: AtomDomain
    local_id: int

    def __post_init__(self) -> None:
        if not isinstance(self.domain, AtomDomain):
            raise TypeError(f"domain debe ser AtomDomain, recibí {type(self.domain).__name__}")
        if not isinstance(self.local_id, int) or self.local_id < 0:
            raise ValueError(f"local_id debe ser entero >= 0, recibí {self.local_id!r}")

    def short(self) -> str:
        """Representación corta estable (útil para logs)."""
        return f"{self.domain.value}:{self.local_id}"


@dataclass(frozen=True)
class Atom:
    """Átomo en su forma de superficie: texto legible + dominio.

    Un `Atom` se registra en el `AtomRegistry` para obtener un `AtomId`.
    Dos `Atom` con la misma `surface_form` y el mismo `domain` producen
    el mismo `AtomId` (idempotencia del registro).
    """

    surface_form: str
    domain: AtomDomain

    def __post_init__(self) -> None:
        if not isinstance(self.surface_form, str) or not self.surface_form:
            raise ValueError("surface_form debe ser string no vacío.")
        if not isinstance(self.domain, AtomDomain):
            raise TypeError(f"domain debe ser AtomDomain, recibí {type(self.domain).__name__}")

    def __str__(self) -> str:
        return f"{self.domain.value}::{self.surface_form}"
