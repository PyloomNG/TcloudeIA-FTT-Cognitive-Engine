"""
Asignación de roles agnóstica al dominio (generalización de Sujeto/Predicado).

En v1, la descomposición estructural asumía dos roles fijos (sujeto,
predicado). Eso era lingüístico, no cognitivo. v2 generaliza: cualquier
dominio puede declarar qué átomos cumplen qué rol funcional. Por
defecto hay tres roles: HEAD (el "núcleo" del input), MODIFIER (los
"modificadores" que lo acompañan) y CONTEXT (información periférica).

La consecuencia clave: la memoria ya no codifica un eje (s, p) preferido.
Las asociaciones se forman sobre `frozenset(HEAD ∪ MODIFIER)`, que es
conmutativo.

Principio SOLID aplicado:
    - Open/Closed: añadir un nuevo dominio es una nueva subclase de
      RoleAssignment. La memoria y el trainer no se modifican.
    - Single Responsibility: cada subclase decide roles para UN dominio.
    - Liskov: las tres implementaciones concretas satisfacen el mismo
      contrato.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from enum import Enum
from typing import Any, Callable, Iterable

from .atom import AtomId


class Role(Enum):
    """Roles funcionales asignables a átomos de un input."""

    HEAD = "HEAD"
    MODIFIER = "MODIFIER"
    CONTEXT = "CONTEXT"


# Tipo de función: dado un AtomId, devuelve su forma de superficie.
SurfaceLookup = Callable[[AtomId], str]


class RoleAssignment(ABC):
    """Estrategia de asignación de roles para los átomos de un payload."""

    @abstractmethod
    def assign(
        self,
        payload: Any,
        atoms: Iterable[AtomId],
        surface_of: SurfaceLookup | None = None,
    ) -> dict[Role, list[AtomId]]:
        """Devuelve un mapping rol -> lista de átomos.

        `surface_of` es un callable opcional que el RoleAssignment puede
        usar para clasificar átomos por su forma de superficie (p.ej. el
        NumberRoleAssignment distingue MAG:* de PRIME:* de DIG:*). Si
        no se proporciona, los átomos caen en CONTEXT como fallback.
        """
        raise NotImplementedError


def _by_surface_prefix(
    atoms: Iterable[AtomId],
    surface_of: SurfaceLookup,
    *,
    head_prefix: str | None = None,
    modifier_prefix: str | None = None,
    context_prefix: str | None = None,
) -> dict[Role, list[AtomId]]:
    """Clasificador genérico por prefijo de superficie.

    Cualquier átomo cuya superficie empieza por `head_prefix` va a HEAD;
    análogamente para MODIFIER y CONTEXT. Si no encaja en ninguno, va a
    CONTEXT como fallback.
    """
    head: list[AtomId] = []
    modifier: list[AtomId] = []
    context: list[AtomId] = []
    for atom in atoms:
        surface = surface_of(atom)
        if head_prefix and surface.startswith(head_prefix):
            head.append(atom)
        elif modifier_prefix and surface.startswith(modifier_prefix):
            modifier.append(atom)
        else:
            context.append(atom)
    # Si context_prefix se especificó, filtrar context por él.
    if context_prefix:
        context = [a for a in context if surface_of(a).startswith(context_prefix)]
    return {Role.HEAD: head, Role.MODIFIER: modifier, Role.CONTEXT: context}


# ---------------------------------------------------------------------------
# Texto
# ---------------------------------------------------------------------------


class PositionalRoleAssignment(RoleAssignment):
    """Para texto: primeros X% de átomos = HEAD, resto = MODIFIER.

    No se usa la superficie: la asignación es puramente posicional sobre
    el orden de producción del extractor (que para bigramas corresponde
    al orden de aparición en la palabra).
    """

    def __init__(self, head_ratio: float = 0.5) -> None:
        if not 0.0 < head_ratio < 1.0:
            raise ValueError("head_ratio debe estar en (0, 1).")
        self._ratio = head_ratio

    def assign(
        self,
        payload: Any,
        atoms: Iterable[AtomId],
        surface_of: SurfaceLookup | None = None,
    ) -> dict[Role, list[AtomId]]:
        atoms_list = list(atoms)
        n = len(atoms_list)
        if n == 0:
            return {Role.HEAD: [], Role.MODIFIER: [], Role.CONTEXT: []}
        split_at = max(1, int(n * self._ratio))
        return {
            Role.HEAD: atoms_list[:split_at],
            Role.MODIFIER: atoms_list[split_at:],
            Role.CONTEXT: [],
        }


# ---------------------------------------------------------------------------
# Imagen
# ---------------------------------------------------------------------------


class ImageSpatialRoleAssignment(RoleAssignment):
    """Para imagen: primera mitad de átomos = HEAD, segunda = MODIFIER.

    Los átomos del extractor de imagen (parches 2x2 + histogramas
    agregados) no llevan coordenadas explícitas. Se particionan por
    orden de producción: la primera mitad representa la mitad superior
    de la rejilla, el resto la mitad inferior. Esta es la discretización
    espacial coherente con el resto del motor: "posición" = orden
    relativo en la producción.
    """

    def assign(
        self,
        payload: Any,
        atoms: Iterable[AtomId],
        surface_of: SurfaceLookup | None = None,
    ) -> dict[Role, list[AtomId]]:
        atoms_list = list(atoms)
        n = len(atoms_list)
        if n == 0:
            return {Role.HEAD: [], Role.MODIFIER: [], Role.CONTEXT: []}
        split_at = max(1, n // 2)
        return {
            Role.HEAD: atoms_list[:split_at],
            Role.MODIFIER: atoms_list[split_at:],
            Role.CONTEXT: [],
        }


# ---------------------------------------------------------------------------
# Número
# ---------------------------------------------------------------------------


class NumberRoleAssignment(RoleAssignment):
    """Para número: MAG:<bucket> = HEAD, PRIME:* = MODIFIER, DIG:* = CONTEXT.

    Esta clasificación usa la superficie de los átomos. Requiere que el
    `surface_of` sea proporcionado por el trainer (que sí tiene el
    registry). Si no se proporciona, todos los átomos caen a CONTEXT.
    """

    def assign(
        self,
        payload: Any,
        atoms: Iterable[AtomId],
        surface_of: SurfaceLookup | None = None,
    ) -> dict[Role, list[AtomId]]:
        if surface_of is None or not isinstance(payload, int):
            return {
                Role.HEAD: [],
                Role.MODIFIER: [],
                Role.CONTEXT: list(atoms),
            }
        return _by_surface_prefix(
            atoms,
            surface_of,
            head_prefix="MAG:",
            modifier_prefix="PRIME:",
            context_prefix="DIG:",
        )
