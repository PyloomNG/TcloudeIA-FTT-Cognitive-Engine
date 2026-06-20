"""
Predictor Estructural generalizado.

Dado un payload (texto, imagen, número o multimodal), predice qué
bibliotecas se activan y con qué intensidad. La salida es una lista de
`Library` ordenadas por medida descendente — ya no un único objeto
"predicción del predicado".

Algoritmo (tres fases):
    1. Activación directa: para cada asociación en memoria que sea
       SUPERSET de los átomos del input, se suma su medida a la
       activación del átomo.
    2. Herencia simbólica (reglas): si una regla conecta un átomo del
       input con un átomo hermano, también se acumula la activación de
       ese hermano.
    3. Filtrado y ranking: para cada biblioteca, se calcula la suma de
       medidas de sus miembros predichos y se ordena descendente.

El bucle `for s: for p:` de v1 (sesgo a dos roles) se reemplaza por una
consulta `memory.supersets_of(input_atoms, threshold)`. Misma
complejidad, sin asumir un eje.

Principio SOLID aplicado:
    - Single Responsibility: solo predice.
    - Dependency Inversion: depende de abstracciones (Memory, Registry,
      RuleStore, LibraryIndex).
    - Open/Closed: la lógica de ranking es sustituible vía subclases.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Mapping

from .atom import AtomId
from .association import AssociationMemory
from .extractor import (
    BigramExtractor,
    ImageEdgeExtractor,
    MultimodalFusionExtractor,
    NumberFactorExtractor,
    RealImageExtractor,
)
from .library import Library
from .quantization import Quantizer
from .registry import AtomRegistry
from .rules import Rule, RuleStore


class GeneralizedStructuralPredictor:
    """Predice las bibliotecas activadas por un payload."""

    def __init__(
        self,
        memory: AssociationMemory,
        registry: AtomRegistry,
        rule_store: RuleStore,
        libraries: Mapping[str, Library] | None = None,
        activation_floor: int = 0,
    ) -> None:
        self._memory = memory
        self._registry = registry
        self._rule_store = rule_store
        self._libraries: dict[str, Library] = dict(libraries) if libraries else {}
        self._floor = activation_floor

    # --- API principal ---

    def predict(self, payload: Mapping[str, Any]) -> list[Library]:
        """Predice las bibliotecas activadas por el payload.

        Devuelve la lista de bibliotecas (de las declaradas en el
        constructor) cuya activación total supera el `activation_floor`,
        ordenadas por medida descendente. Bibliotecas sin activación
        suficiente no aparecen en la lista.
        """
        input_atoms = self._extract_atoms(payload)
        if not input_atoms:
            return []
        activation = self._compute_activation(input_atoms)
        return self._rank_libraries(activation)

    def predict_from_atoms(self, input_atoms: set[AtomId]) -> list[Library]:
        """Variante que opera directamente sobre átomos pre-extraídos."""
        if not input_atoms:
            return []
        activation = self._compute_activation(input_atoms)
        return self._rank_libraries(activation)

    # --- Administración de bibliotecas ---

    def register_library(self, library: Library) -> None:
        """Registra/actualiza una biblioteca en el índice del predictor."""
        self._libraries[library.name] = library

    def unregister_library(self, library_name: str) -> None:
        """Quita una biblioteca del índice del predictor.

        Esencial cuando el motor disuelve o renombra un cluster: si el
        predictor conserva bibliotecas fantasma, las predicciones PRED(X)
        apuntarán a X que ya no existe en el motor.
        """
        self._libraries.pop(library_name, None)

    def known_libraries(self) -> dict[str, Library]:
        return dict(self._libraries)

    # --- Internos ---

    def _extract_atoms(self, payload: Mapping[str, Any]) -> set[AtomId]:
        """Despacha al extractor apropiado por cada valor del payload.

        Para que `supersets_of(input)` encuentre la asociación entrenada,
        el `input` debe ser un subconjunto de los átomos reforzados. En
        concreto: extraemos los átomos estructurales (parches, bordes,
        bigramas, primos, ...) y APLICAMOS el mismo role assignment que
        el trainer (HEAD ∪ MODIFIER), pero OMITIMOS el átomo-concepto
        (W:*, N:*, I:*) del input.

        ¿Por qué omitir el concept? En texto/números, el concept es la
        identidad de la palabra/número y aparece en la asociación. Como
        el input contiene el MISMO concept, el match funciona. En
        IMAGEN, el concept es un hash por rejilla: dos rejillas
        distintas producen concepts distintos y el match fallaría. Al
        omitir el concept del input, la generalización se hace a través
        de los átomos estructurales (P2x2:*, HEDGE:*, VEDGE:*) que SÍ
        son compartidos entre rejillas de la misma clase.
        """
        # Mapa rápido tipo -> extractor.
        type_map: dict[type, Any] = {
            str: BigramExtractor(),
            int: NumberFactorExtractor(),
        }
        type_map[list] = RealImageExtractor()
        type_map[tuple] = RealImageExtractor()
        type_map[Mapping] = MultimodalFusionExtractor()
        # Mapa de roles por dominio.
        from .role import (
            ImageSpatialRoleAssignment,
            NumberRoleAssignment,
            PositionalRoleAssignment,
            Role,
        )
        role_map: dict[type, Any] = {
            str: PositionalRoleAssignment(),
            int: NumberRoleAssignment(),
        }
        role_map[list] = ImageSpatialRoleAssignment()
        role_map[tuple] = ImageSpatialRoleAssignment()

        head_modifier: set[AtomId] = set()
        for _key, value in payload.items():
            extractor = type_map.get(type(value))
            if extractor is None and isinstance(value, (list, tuple)):
                extractor = type_map[list]
            if extractor is None and isinstance(value, Mapping):
                extractor = type_map[Mapping]
            if extractor is None:
                continue
            atoms = extractor.extract(value, self._registry)
            role_assigner = role_map.get(type(value))
            if role_assigner is not None:
                roles = role_assigner.assign(
                    value, atoms, surface_of=self._registry.surface_of
                )
                # El predictor NO incluye el concept-atom (W:*, N:*, I:*)
                # en su input. Esto preserva la generalización: dos
                # inputs del mismo tipo (p.ej. dos rejillas VSTRIPE)
                # producen el mismo input de patches+edges aunque sus
                # concept-atoms sean distintos.
                head_modifier.update(roles[Role.HEAD])
                head_modifier.update(roles[Role.MODIFIER])
            else:
                head_modifier.update(atoms)
        return head_modifier

    def _compute_activation(
        self, input_atoms: set[AtomId]
    ) -> dict[AtomId, int]:
        """Suma la activación directa + herencia de reglas.

        Algoritmo (agnóstico al dominio, sin asumir eje s/p):
            Para cada átomo h del input, y para cada átomo p conocido:
                activation[p] += suma de medidas de asociaciones
                                 que contienen {h, p}.
        Esto es el análogo v2 de `for s: for p: memory.get(s,p)` de v1:
        mide "cuánto co-ocurre p con cualquier h del input". La consulta
        `superset_of({h, p}, threshold)` evita enumerar todos los
        pares del universo.
        """
        activation: dict[AtomId, int] = defaultdict(int)

        # Fase 1: activación directa.
        for h in input_atoms:
            for p in self._registry.known_ids():
                pair = frozenset({h, p})
                # Medida agregada: suma de todas las asociaciones que
                # contienen exactamente el par {h, p}. Esto es la
                # generalización de v1's `memory.get(s, p)` a la
                # estructura de frozenset.
                agg = self._aggregate_measure(pair)
                if agg > self._floor:
                    activation[p] += agg

        # Fase 2: herencia simbólica (reglas).
        for rule in self._rule_store.all():
            partner = self._rule_partner(rule, input_atoms)
            if partner is None:
                continue
            for p in self._registry.known_ids():
                pair = frozenset({partner, p})
                agg = self._aggregate_measure(pair)
                if agg > self._floor:
                    activation[p] += agg

        return dict(activation)

    def _aggregate_measure(self, atoms: frozenset[AtomId]) -> int:
        """Suma de medidas de TODAS las asociaciones que son superset
        de `atoms`. Esto es la consulta que reemplaza al `memory.get`
        de v1: en v2 un "par" no es solo un par ordenado, puede ser un
        conjunto arbitrario; medimos la co-ocurrencia agregando todas
        las asociaciones que lo contienen."""
        total = 0
        for assoc, m in self._memory.supersets_of(atoms, -9):
            # supersets_of devuelve con threshold; aquí queremos TODAS,
            # así que filtramos por medida explícitamente.
            if m > 0:
                total += m
        return total

    @staticmethod
    def _rule_partner(rule: Rule, input_atoms: set[AtomId]) -> AtomId | None:
        """Si exactamente uno de los átomos de la regla está en input,
        devuelve el otro. Si hay 0 o >1, devuelve None."""
        matches = [a for a in rule.atoms if a in input_atoms]
        if len(matches) != 1:
            return None
        for a in rule.atoms:
            if a not in input_atoms:
                return a
        return None

    def _rank_libraries(
        self, activation: dict[AtomId, int]
    ) -> list[Library]:
        """Construye Libraries "predichas" a partir de la activación."""
        predicted_set = {
            atom for atom, act in activation.items() if act > self._floor
        }
        if not predicted_set:
            return []

        results: list[Library] = []
        for name, lib in self._libraries.items():
            intersect = lib.members & predicted_set
            if not intersect:
                continue
            # Medida predicha: suma de activaciones de los miembros que
            # están tanto en la biblioteca como en el set predicho. Se
            # cuantiza a 4 bits antes de almacenarse en la Library.
            total = sum(activation[atom] for atom in intersect)
            if total <= self._floor:
                continue
            quantized = Quantizer.quantize(total)
            results.append(
                Library(
                    name=f"PRED({name})",
                    members=intersect,
                    measure=quantized,
                    parents=(),
                )
            )
        # Orden por medida descendente, desempate por nombre.
        results.sort(key=lambda lib: (-lib.measure, lib.name))
        return results
