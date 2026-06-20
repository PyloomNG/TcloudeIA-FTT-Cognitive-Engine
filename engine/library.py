"""
Bibliotecas Medibles (Pilar 3: teoría de conjuntos).

Pilar 3 (Motor Matemático): el conocimiento se estructura como "Bibliotecas
Medibles" (Conjuntos Medibles). Las operaciones de inferencia son operaciones
de conjuntos (∩, ∪, ¬, ∈), NO multiplicaciones de matrices densas.

A diferencia de v1, una `Library` puede declarar `parents`: nombres de
otras bibliotecas que la contienen transitivamente. Esto habilita
multi-membresía explícita (p.ej. PERRO ⊂ MAMIFERO ⊂ ANIMAL) sin necesidad
de reglas inferidas: la jerarquía es DECLARATIVA, se consulta, no se
infiere como side-effect.

Una Library es INMUTABLE. Toda operación que parezca "modificar" devuelve
una nueva Library. Esto preserva el "congelado" y permite razonar sobre
estados anteriores con seguridad.

Principio SOLID aplicado:
    - Single Responsibility: modela UN conjunto medible y sus operaciones.
    - Liskov Substitution: cualquier subclase (futura) se comporta igual.
    - Inmutabilidad: thread-safe por construcción.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Iterable, Iterator

from .atom import AtomId

if TYPE_CHECKING:  # solo para type hints, evita import circular
    from .registry import AtomRegistry


@dataclass(frozen=True)
class Library:
    """Conjunto medible inmutable + jerarquía declarativa de padres.

    Una Library puede llevar además metadatos semánticos:
        - `description`: definición en lenguaje natural (p.ej. "cara: parte
          frontal de la cabeza humana con ojos, nariz y boca").
        - `context`:    tupla inmutable de (clave, valor_repr) para reglas,
          etiquetas, sinónimos, o cualquier información adicional. Los
          valores se almacenan como `repr()` para garantizar hashabilidad
          y permitir listas/dicts arbitrarios.
    Estos metadatos son INMUTABLES: `with_description` y `with_context`
    devuelven una nueva Library con los metadatos actualizados.
    """

    name: str
    members: frozenset[AtomId] = field(default_factory=frozenset)
    measure: int = 0
    parents: tuple[str, ...] = ()
    description: str | None = None
    context: tuple[tuple[str, str], ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name:
            raise ValueError("Library.name debe ser string no vacío.")
        if not isinstance(self.measure, int):
            raise TypeError(f"measure debe ser int, recibí {type(self.measure).__name__}")
        if not (-8 <= self.measure <= 7):
            # No cuantizamos automáticamente: el motor se encarga en reinforce().
            # Aquí solo advertimos del rango permitido por el cuantizador.
            raise ValueError(
                f"measure fuera de rango 4 bits [-8, 7]: {self.measure}"
            )
        if not isinstance(self.parents, tuple):
            raise TypeError("parents debe ser tuple[str, ...].")
        if self.description is not None and not isinstance(self.description, str):
            raise TypeError("description debe ser str o None.")

    # --- Fábricas convenientes ---

    @staticmethod
    def empty(name: str) -> "Library":
        return Library(name=name, members=frozenset(), measure=0, parents=())

    @staticmethod
    def of(name: str, atoms: Iterable[AtomId], *, parents: Iterable[str] = ()) -> "Library":
        return Library(
            name=name,
            members=frozenset(atoms),
            measure=0,
            parents=tuple(parents),
        )

    # --- Modificadores de metadatos (devuelven NUEVA Library) ---

    def with_description(self, description: str | None) -> "Library":
        """Devuelve una copia con la descripción (o None) actualizada."""
        return Library(
            name=self.name,
            members=self.members,
            measure=self.measure,
            parents=self.parents,
            description=description,
            context=self.context,
        )

    def with_context(self, context: dict | None) -> "Library":
        """Devuelve una copia con el dict de contexto (o tupla vacía) actualizado.

        Los valores se almacenan como `repr()` para garantizar que la
        Library sigue siendo hashable (los frozensets/tuplas la necesitan).
        """
        ctx_tuple = (
            tuple(sorted((str(k), repr(v)) for k, v in context.items()))
            if context
            else ()
        )
        return Library(
            name=self.name,
            members=self.members,
            measure=self.measure,
            parents=self.parents,
            description=self.description,
            context=ctx_tuple,
        )

    def context_dict(self) -> dict[str, Any]:
        """Devuelve el contexto como dict estándar (evalúa reprs simples).

        Para valores que se almacenaron como `repr(v)` (strings, números,
        tuplas, etc.), intenta reconstruir el valor original con `ast.literal_eval`.
        Si no es posible, devuelve el `repr` como string.
        """
        import ast
        result: dict[str, Any] = {}
        for k, v_repr in self.context:
            try:
                result[k] = ast.literal_eval(v_repr)
            except (ValueError, SyntaxError):
                result[k] = v_repr
        return result

    def with_parents(self, parents: Iterable[str]) -> "Library":
        """Devuelve una copia con el conjunto de padres reemplazado."""
        return Library(
            name=self.name,
            members=self.members,
            measure=self.measure,
            parents=tuple(parents),
            description=self.description,
            context=self.context,
        )

    # --- Pertenencia directa ---

    def contains(self, atom: AtomId) -> bool:
        """∈ directo. NO considera transitividad de padres."""
        return atom in self.members

    def is_empty(self) -> bool:
        return len(self.members) == 0

    def size(self) -> int:
        return len(self.members)

    # --- Operaciones de teoría de conjuntos (devuelven NUEVA Library) ---

    def intersects(self, other: "Library") -> "Library":
        """Intersección (AND)."""
        return Library(
            name=f"({self.name} AND {other.name})",
            members=self.members & other.members,
            measure=min(self.measure, other.measure),
            parents=(),
        )

    def union(self, other: "Library") -> "Library":
        """Unión (OR)."""
        return Library(
            name=f"({self.name} OR {other.name})",
            members=self.members | other.members,
            measure=max(self.measure, other.measure),
            parents=(),
        )

    def complement(self, universe: Iterable[AtomId]) -> "Library":
        """Complemento respecto a un universo explícito."""
        return Library(
            name=f"NOT {self.name}",
            members=frozenset(set(universe) - set(self.members)),
            measure=-self.measure,
            parents=(),
        )

    def difference(self, other: "Library") -> "Library":
        """Diferencia (\\)."""
        return Library(
            name=f"({self.name} \\ {other.name})",
            members=self.members - other.members,
            measure=self.measure,
            parents=(),
        )

    # --- Modificadores (devuelven NUEVA Library) ---

    def add(self, atom: AtomId) -> "Library":
        """Añade un átomo al conjunto. Devuelve una NUEVA Library."""
        if atom in self.members:
            return self
        return Library(
            name=self.name,
            members=self.members | {atom},
            measure=self.measure,
            parents=self.parents,
            description=self.description,
            context=self.context,
        )

    def remove(self, atom: AtomId) -> "Library":
        """Quita un átomo del conjunto. Devuelve una NUEVA Library."""
        if atom not in self.members:
            return self
        return Library(
            name=self.name,
            members=self.members - {atom},
            measure=self.measure,
            parents=self.parents,
            description=self.description,
            context=self.context,
        )

    def merge(self, other: "Library") -> "Library":
        """Une los miembros de otra Library en esta. Devuelve una NUEVA Library."""
        return Library(
            name=self.name,
            members=self.members | other.members,
            measure=max(self.measure, other.measure),
            parents=self.parents,
            description=self.description,
            context=self.context,
        )

    # --- Comparación de conjuntos ---

    def is_subset_of(self, other: "Library") -> bool:
        return self.members <= other.members

    # --- Jerarquía declarativa ---

    def transitive_parents(
        self, library_index: "dict[str, Library] | None" = None
    ) -> tuple[str, ...]:
        """Devuelve la lista de nombres de padres transitivos (DFS sin ciclos).

        Si se pasa `library_index` (dict nombre -> Library), la transitividad
        se expande recursivamente. Si no, solo se devuelven los padres
        directos. La detección de ciclos se hace por conjunto de visitados.
        """
        if library_index is None:
            return tuple(self.parents)
        visited: set[str] = set()
        order: list[str] = []
        stack: list[str] = list(self.parents)
        while stack:
            current = stack.pop()
            if current in visited:
                continue
            visited.add(current)
            order.append(current)
            parent_lib = library_index.get(current)
            if parent_lib is not None:
                for p in parent_lib.parents:
                    if p not in visited:
                        stack.append(p)
        return tuple(order)

    def __iter__(self) -> Iterator[AtomId]:
        return iter(self.members)

    def __str__(self) -> str:
        parents = f" superset_of={{{', '.join(self.parents)}}}" if self.parents else ""
        return (
            f"{self.name}{parents} |mu|={abs(self.measure)} |C|={len(self.members)}"
        )
