"""
Memoria de Asociaciones: tabla frozenset[AtomId] -> medida firmada.

Pilar 2: almacenamiento discreto. Toda escritura pasa por el Quantizer,
garantizando que NINGUNA medida se almacene fuera de [-8, +7].

Pilar 4: el aprendizaje es one-shot. Las co-ocurrencias actualizan la
medida del conjunto en +/-1 y se congelan. No hay backprop, no hay gradientes.

Cambio estructural respecto a v1:
    - Clave: v1 era `tuple[int, int]` (sujeto, predicado). v2 es
      `frozenset[AtomId]`. Esto elimina la asimetría sujeto/predicado y
      hace que la co-ocurrencia sea conmutativa.
    - Cuantización: se aplica al VALOR (la medida), no a la clave (los IDs
      de átomos pueden crecer sin límite; son solo etiquetas de identidad).
    - Nueva consulta: `supersets_of(atoms, threshold)` para que el predictor
      agnóstico pueda preguntar "¿qué asociaciones más grandes contienen
      estos átomos?" sin asumir un eje.

Principio SOLID aplicado:
    - Single Responsibility: la memoria SOLO persiste y consulta medidas.
    - Dependency Inversion: depende del Quantizer (abstracción inyectable).
"""

from __future__ import annotations

from typing import Iterable

from .atom import AtomId
from .quantization import Quantizer


class AssociationMemory:
    """Diccionario discreto de medidas con signo. Cuantización en cada escritura."""

    def __init__(self, quantizer: type[Quantizer] | None = None) -> None:
        self._quantizer = quantizer if quantizer is not None else Quantizer
        self._measures: dict[frozenset[AtomId], int] = {}

    # --- Operaciones básicas ---

    def get(self, atoms: Iterable[AtomId]) -> int:
        """Lee la medida de un conjunto de átomos. 0 si nunca se observó."""
        key = frozenset(atoms)
        if not key:
            return 0
        return self._measures.get(key, 0)

    def reinforce(self, atoms: Iterable[AtomId], delta: int) -> int:
        """Aplica ±delta a la medida y la cuantiza antes de persistir.

        Retorna la nueva medida (útil para depuración y propagación).
        Un conjunto vacío no se almacena nunca (no tiene semántica).
        """
        key = frozenset(atoms)
        if not key:
            return 0
        current = self._measures.get(key, 0)
        new_value = self._quantizer.quantize(current + delta)
        if new_value == 0:
            # Podamos el cero: una medida exactamente cero no aporta
            # información Hebbiana y mantiene la memoria compacta.
            self._measures.pop(key, None)
        else:
            self._measures[key] = new_value
        return new_value

    def all_measures(self) -> dict[frozenset[AtomId], int]:
        """Copia superficial del estado (para inspección)."""
        return dict(self._measures)

    def __len__(self) -> int:
        return len(self._measures)

    # --- Consultas estructurales (núcleo de v2) ---

    def strong_associations(
        self, threshold: int
    ) -> list[tuple[frozenset[AtomId], int]]:
        """Devuelve [(conjunto, medida), ...] con medida estrictamente por encima del umbral."""
        return [
            (atoms, m)
            for atoms, m in self._measures.items()
            if m > threshold
        ]

    def supersets_of(
        self, atoms: Iterable[AtomId], threshold: int
    ) -> list[tuple[frozenset[AtomId], int]]:
        """Devuelve asociaciones SUPERSET de `atoms` con medida por encima del umbral.

        Esta consulta es la pieza clave del predictor agnóstico: dado un
        conjunto de átomos (p.ej. los de un input), devuelve todas las
        asociaciones co-ocurrentes más grandes cuya medida supera el umbral.
        Sin asumir un eje.
        """
        target = frozenset(atoms)
        if not target:
            return []
        return [
            (atoms, m)
            for atoms, m in self._measures.items()
            if m > threshold and target <= atoms
        ]

    def measures_in_range(
        self, min_value: int, max_value: int
    ) -> list[tuple[frozenset[AtomId], int]]:
        """Devuelve asociaciones con medida en [min_value, max_value]."""
        return [
            (atoms, m)
            for atoms, m in self._measures.items()
            if min_value <= m <= max_value
        ]
