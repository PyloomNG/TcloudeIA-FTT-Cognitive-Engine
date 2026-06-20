"""
Inferencia Autónoma de Reglas (metacognición, Pilar 5).

Pilar 5: el sistema debe ser capaz de metacognición: analizar sus propias
medidas almacenadas. Si el sistema detecta que componentes portables
distintos activan las mismas estructuras, debe inferir y crear
automáticamente una nueva regla abstracta sin intervención humana.

Cambios respecto a v1:
    - `InferredRule` ahora es BINARIO pero con `AtomId` (no `int`).
    - Aparece `NaryInferredRule` (v2.1) para reglas N-arias; ambos
      subtipos satisfacen la misma `Rule` base, así que entran al mismo
      `RuleStore` (Liskov).
    - `CoActivationRuleInference` trabaja con la nueva clave
      `frozenset[AtomId]` y produce reglas de un solo dominio (no
      cruza dominios, por construcción).

Principio SOLID aplicado:
    - Open/Closed: la estrategia de inferencia es intercambiable.
    - Single Responsibility: cada clase hace UNA cosa (Rule, Store,
      Strategy, inferencia concreta).
    - Liskov: InferredRule y NaryInferredRule son intercambiables en
      cualquier consumidor de Rule.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from .atom import AtomDomain, AtomId

if TYPE_CHECKING:
    from .association import AssociationMemory
    from .registry import AtomRegistry


# ---------------------------------------------------------------------------
# Modelo de reglas
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Rule:
    """Clase base. Una regla conecta un conjunto de átomos con un contexto
    compartido (el "contexto" es un frozenset de átomos que co-activan)."""

    name: str
    atoms: tuple[AtomId, ...]
    shared_context: frozenset[AtomId]

    def involves(self, atom: AtomId) -> bool:
        return atom in self.atoms


@dataclass(frozen=True)
class InferredRule(Rule):
    """Regla BINARIA: dos átomos son intercambiables en un contexto.

    'Intercambiables' significa que co-activan un conjunto compartido de
    átomos. Esto es la base de la abstracción Fodoriana: si dos átomos
    producen la misma estructura, el sistema puede tratarlos como
    equivalentes en ese contexto.
    """

    component_a: AtomId = field(init=False)
    component_b: AtomId = field(init=False)

    def __init__(self, name: str, component_a: AtomId, component_b: AtomId,
                 shared_context: frozenset[AtomId]) -> None:
        # El orden canónico de `atoms` es: (min, max) por (domain, local_id)
        # para garantizar que (a, b) y (b, a) produzcan la misma regla.
        a, b = sorted([component_a, component_b], key=lambda x: (x.domain.value, x.local_id))
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "atoms", (a, b))
        object.__setattr__(self, "shared_context", shared_context)
        object.__setattr__(self, "component_a", a)
        object.__setattr__(self, "component_b", b)

    def partner(self, atom: AtomId) -> AtomId | None:
        """Dado uno de los dos componentes, devuelve el otro (o None)."""
        if atom == self.component_a:
            return self.component_b
        if atom == self.component_b:
            return self.component_a
        return None


@dataclass(frozen=True)
class NaryInferredRule(Rule):
    """Regla N-ARIA (v2.1): N átomos son intercambiables en un contexto.

    Caso de uso futuro: un mismo contexto activable por 3+ átomos a la
    vez (p.ej. una Library que emerge de tres observaciones distintas).
    """

    def __init__(self, name: str, atoms: tuple[AtomId, ...],
                 shared_context: frozenset[AtomId]) -> None:
        if len(atoms) < 2:
            raise ValueError("NaryInferredRule requiere al menos 2 átomos.")
        canonical = tuple(sorted(set(atoms), key=lambda x: (x.domain.value, x.local_id)))
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "atoms", canonical)
        object.__setattr__(self, "shared_context", shared_context)


# ---------------------------------------------------------------------------
# Almacén de reglas
# ---------------------------------------------------------------------------


class RuleStore:
    """Almacén de reglas (la 'metacognición' del motor).

    Acepta cualquier subclase de `Rule` (Liskov). Internamente deduplica
    por el `name` (que se considera identificador único).
    """

    def __init__(self) -> None:
        self._rules: dict[str, Rule] = {}

    def add(self, rule: Rule) -> bool:
        """Añade una regla. Devuelve False si ya existía con ese nombre."""
        if rule.name in self._rules:
            return False
        self._rules[rule.name] = rule
        return True

    def all(self) -> list[Rule]:
        return list(self._rules.values())

    def rules_involving(self, atom: AtomId) -> list[Rule]:
        return [r for r in self._rules.values() if r.involves(atom)]

    def __len__(self) -> int:
        return len(self._rules)


# ---------------------------------------------------------------------------
# Estrategias de inferencia
# ---------------------------------------------------------------------------


class RuleInferenceStrategy(ABC):
    """Estrategia de inferencia (OCP)."""

    @abstractmethod
    def infer(
        self,
        memory: "AssociationMemory",
        registry: "AtomRegistry",
    ) -> list[Rule]:
        """Inspecciona la memoria y devuelve reglas candidatas (no almacenadas)."""
        raise NotImplementedError


class CoActivationRuleInference(RuleInferenceStrategy):
    """Si dos átomos disparan >=K átomos en común, se declaran intercambiables.

    Versión v2: trabaja sobre la clave `frozenset[AtomId]`. Reglas
    producidas son siempre de un único dominio (no cruzamos dominios).
    """

    def __init__(
        self,
        activation_threshold: int = 3,
        overlap_threshold: int = 2,
        same_domain_only: bool = True,
    ) -> None:
        if activation_threshold < 0 or overlap_threshold < 1:
            raise ValueError("Umbrales fuera de rango válido.")
        self._activation_threshold = activation_threshold
        self._overlap_threshold = overlap_threshold
        self._same_domain_only = same_domain_only

    def infer(
        self,
        memory: "AssociationMemory",
        registry: "AtomRegistry",
    ) -> list[Rule]:
        # 1) Indexar asociaciones fuertes por átomo.
        #    atom -> set(conjuntos de co-ocurrencia)
        atom_to_contexts: dict[AtomId, set[frozenset[AtomId]]] = defaultdict(set)
        for atoms, m in memory.strong_associations(self._activation_threshold):
            for atom in atoms:
                atom_to_contexts[atom].add(atoms)

        new_rules: list[Rule] = []
        atoms_list = list(atom_to_contexts.keys())

        # 2) Pares (o N-tuplas) con intersección significativa.
        for i in range(len(atoms_list)):
            for j in range(i + 1, len(atoms_list)):
                a, b = atoms_list[i], atoms_list[j]
                if self._same_domain_only and a.domain != b.domain:
                    continue
                contexts_a = atom_to_contexts[a]
                contexts_b = atom_to_contexts[b]
                shared = contexts_a & contexts_b
                # El "contexto compartido" se interpreta como la unión de
                # los átomos que aparecen en las asociaciones comunes,
                # distintos de a y b. Esto preserva la idea original de
                # "componentes predicado compartidos" sin acoplar a un eje.
                shared_atoms: set[AtomId] = set()
                for ctx in shared:
                    shared_atoms.update(ctx)
                shared_atoms.discard(a)
                shared_atoms.discard(b)
                if len(shared_atoms) >= self._overlap_threshold:
                    rule = InferredRule(
                        name=f"CONCEPTO_ABSTRACTO_{a.short()}_{b.short()}",
                        component_a=a,
                        component_b=b,
                        shared_context=frozenset(shared_atoms),
                    )
                    new_rules.append(rule)
        return new_rules
