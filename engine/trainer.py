"""
Entrenador Hebbiano generalizado (Pilar 4: aprendizaje one-shot).

Pilar 4: el aprendizaje es One-Shot. Las co-ocurrencias actualizan la
medida del conjunto en +/-1 y se congelan. Evitamos el olvido
catastrófico: lo nuevo no sobrescribe lo anterior.

A diferencia de v1 (que solo operaba sobre oraciones tokenizadas), el
`GeneralizedHebbianTrainer` de v2 recibe un `payload` que es un dict
`{domain_key: raw_payload}`. Para cada par (key, value):
    1. Despacha al extractor del dominio del valor.
    2. Pasa los átomos resultantes al `RoleAssignment` del dominio.
    3. Refuerza en +1 la asociación `frozenset(HEAD ∪ MODIFIER)`.
    4. Aplica Anti-Hebb selectivo: solo inhibe átomos del mismo dominio
       que PREVIAMENTE tuvieron medida positiva con el conjunto actual.

El Anti-Hebb sigue siendo selectivo (no inundamos todo el espacio), lo
que preserva la información y respeta la cuasi-saturación de 4 bits.

Principio SOLID aplicado:
    - Single Responsibility: el entrenador SOLO coordina actualizaciones.
    - Open/Closed: nuevos dominios no requieren modificar el trainer.
    - Dependency Inversion: extractor y role_assignment se inyectan.
"""

from __future__ import annotations

from typing import Any, Mapping

from .atom import AtomDomain, AtomId
from .association import AssociationMemory
from .extractor import (
    AtomExtractor,
    BigramExtractor,
    ImageEdgeExtractor,
    MultimodalFusionExtractor,
    NumberFactorExtractor,
    RealImageExtractor,
)
from .registry import AtomRegistry
from .role import (
    ImageSpatialRoleAssignment,
    NumberRoleAssignment,
    PositionalRoleAssignment,
    Role,
    RoleAssignment,
)


def default_extractors() -> dict[AtomDomain, AtomExtractor]:
    """Conjunto por defecto de extractores por dominio."""
    return {
        AtomDomain.TEXT: BigramExtractor(),
        AtomDomain.IMAGE: RealImageExtractor(),
        AtomDomain.NUMBER: NumberFactorExtractor(),
        AtomDomain.MULTIMODAL: MultimodalFusionExtractor(),
    }


def default_role_assignments() -> dict[AtomDomain, RoleAssignment]:
    """Conjunto por defecto de role assignments por dominio."""
    return {
        AtomDomain.TEXT: PositionalRoleAssignment(),
        AtomDomain.IMAGE: ImageSpatialRoleAssignment(),
        AtomDomain.NUMBER: NumberRoleAssignment(),
        AtomDomain.MULTIMODAL: PositionalRoleAssignment(),  # fallback
    }


class GeneralizedHebbianTrainer:
    """Entrenador one-shot agnóstico al dominio."""

    def __init__(
        self,
        memory: AssociationMemory,
        registry: AtomRegistry,
        extractors: Mapping[AtomDomain, AtomExtractor] | None = None,
        role_assignments: Mapping[AtomDomain, RoleAssignment] | None = None,
    ) -> None:
        self._memory = memory
        self._registry = registry
        self._extractors: dict[AtomDomain, AtomExtractor] = (
            dict(extractors) if extractors is not None else default_extractors()
        )
        self._role_assignments: dict[AtomDomain, RoleAssignment] = (
            dict(role_assignments) if role_assignments is not None
            else default_role_assignments()
        )

    # --- API principal ---

    def train(self, payload: Mapping[str, Any]) -> None:
        """Observa un payload multimodal.

        `payload` es un dict donde la clave es arbitraria (típicamente
        'text', 'image', 'number') y el valor es el dato crudo de ese
        dominio.

        Modos de operación:
            - Multi-clave (2+ entradas): se trata como UNA observación
              multimodal. Se invoca el MultimodalFusionExtractor con
              el dict completo, lo que crea un átomo ancla CO_OCC:* en
              el dominio MULTIMODAL.
            - Mono-clave (1 entrada): se procesa por el extractor de
              su dominio. Para texto, el valor se tokeniza en palabras
              (split por espacios); cada palabra se procesa por
              separado para evitar bigramas a través de fronteras.
        """
        if not payload:
            return
        if len(payload) >= 2:
            # Observación genuinamente multimodal.
            self._train_one(dict(payload), AtomDomain.MULTIMODAL)
            return
        # Mono-clave.
        for _key, value in payload.items():
            domain = self._infer_domain(value)
            if domain is None:
                continue
            if domain == AtomDomain.TEXT and isinstance(value, str):
                words = [w for w in value.lower().split() if w]
                for word in words:
                    self._train_one(word, domain)
            else:
                self._train_one(value, domain)

    def train_one(self, value: Any, domain: AtomDomain | None = None) -> None:
        """Variante de un solo input."""
        if domain is None:
            domain = self._infer_domain(value)
        if domain is None:
            return
        self._train_one(value, domain)

    # --- Extracción sin escritura (usado por auto_cluster) ---

    def structural_atoms(
        self, payload: Mapping[str, Any]
    ) -> tuple[set[AtomId], AtomId | None]:
        """Devuelve (átomos_estructurales, átomo_concepto) de un payload.

        "Estructurales" son los átomos que NO son concept-atom (W:*, N:*,
        I:*, CO_OCC:*). Son los átomos compartidos entre instancias de
        la misma clase, así que son los que se usan para agrupar (Jaccard).

        Si el payload tiene 2+ claves, se trata como multimodal: el
        concept-atom resultante es el CO_OCC:* del fusion extractor y los
        estructurales incluyen los átomos estructurales de cada sub-dominio.
        """
        from .role import Role
        if not payload:
            return set(), None
        # Caso multimodal: fusion extractor.
        if len(payload) >= 2:
            extractor = self._extractors.get(AtomDomain.MULTIMODAL)
            role_assigner = self._role_assignments.get(AtomDomain.MULTIMODAL)
            if extractor is None or role_assigner is None:
                return set(), None
            atoms = extractor.extract(dict(payload), self._registry)
            roles = role_assigner.assign(
                dict(payload), atoms, surface_of=self._registry.surface_of
            )
            # En multimodal, el concept-atom es el CO_OCC:* recién creado.
            concept: AtomId | None = None
            structural: set[AtomId] = set()
            for a in roles[Role.HEAD] + roles[Role.MODIFIER]:
                surf = self._registry.surface_of(a)
                if surf.startswith("CO_OCC:") and concept is None:
                    concept = a
                else:
                    structural.add(a)
            return structural, concept
        # Mono-clave.
        for _key, value in payload.items():
            domain = self._infer_domain(value)
            if domain is None:
                continue
            if domain == AtomDomain.TEXT and isinstance(value, str):
                words = [w for w in value.lower().split() if w]
                structural_all: set[AtomId] = set()
                concept_id: AtomId | None = None
                for word in words:
                    atoms = self._extractors[domain].extract(word, self._registry)
                    concept_surf = self._concept_atom(word, domain)
                    this_concept: AtomId | None = None
                    if concept_surf is not None:
                        this_concept = self._registry.register(concept_surf, domain)
                    roles = self._role_assignments[domain].assign(
                        word, atoms, surface_of=self._registry.surface_of
                    )
                    head_modifier = set(roles[Role.HEAD]) | set(roles[Role.MODIFIER])
                    if this_concept is not None:
                        head_modifier.discard(this_concept)
                    structural_all.update(head_modifier)
                    if this_concept is not None and concept_id is None:
                        concept_id = this_concept
                return structural_all, concept_id
            # Número o imagen.
            extractor = self._extractors.get(domain)
            role_assigner = self._role_assignments.get(domain)
            if extractor is None or role_assigner is None:
                return set(), None
            atoms = extractor.extract(value, self._registry)
            roles = role_assigner.assign(
                value, atoms, surface_of=self._registry.surface_of
            )
            concept_surf = self._concept_atom(value, domain)
            concept_id = None
            if concept_surf is not None:
                concept_id = self._registry.register(concept_surf, domain)
            head_modifier = set(roles[Role.HEAD]) | set(roles[Role.MODIFIER])
            if concept_id is not None:
                head_modifier.discard(concept_id)
            return head_modifier, concept_id
        return set(), None

    # --- Internos ---

    def _train_one(self, value: Any, domain: AtomDomain) -> None:
        extractor = self._extractors.get(domain)
        role_assigner = self._role_assignments.get(domain)
        if extractor is None or role_assigner is None:
            return
        atoms = extractor.extract(value, self._registry)
        if not atoms:
            return
        # Átomo-concepto: identidad de superficie del input. Es el átomo
        # que el cliente usa para declarar pertenencias a bibliotecas
        # ("perro" ∈ ANIMAL, "12" ∈ EVEN, etc.). Lo añadimos al HEAD
        # para que aparezca en TODA asociación que refuerce este input.
        concept_surface = self._concept_atom(value, domain)
        concept_id: AtomId | None = None
        if concept_surface is not None:
            concept_id = self._registry.register(concept_surface, domain)
            atoms.add(concept_id)
        roles = role_assigner.assign(
            value, atoms, surface_of=self._registry.surface_of
        )
        # El concept-atom siempre va al HEAD (es el núcleo semántico).
        if concept_id is not None:
            roles[Role.HEAD] = [concept_id] + [
                a for a in roles[Role.HEAD] if a != concept_id
            ]
        # Fase 1: Hebb puro. Refuerza HEAD ∪ MODIFIER en +1.
        head_modifier = set(roles[Role.HEAD]) | set(roles[Role.MODIFIER])
        if not head_modifier:
            return
        self._memory.reinforce(head_modifier, +1)

        # Fase 2: Anti-Hebb selectivo. Solo inhibimos átomos del mismo
        # dominio que previamente tuvieron asociación POSITIVA con el
        # conjunto actual.
        current = frozenset(head_modifier)
        current_measure = self._memory.get(current)
        if current_measure <= 0:
            return
        for atom in self._registry.known_in_domain(domain):
            if atom in current:
                continue
            # Comprobamos si `atom` aparece en alguna asociación
            # previamente positiva con el `current`.
            had_positive = self._had_positive_association_with(atom, current)
            if had_positive:
                self._memory.reinforce(current | {atom}, -1)

    def _had_positive_association_with(
        self, atom: AtomId, current: frozenset[AtomId]
    ) -> bool:
        """True si el conjunto EXACTO `current ∪ {atom}` tuvo medida > 0
        previamente. Esta es la versión v2 del Anti-Hebb selectivo de v1:
        solo inhibimos asociaciones que fueron reforzadas con EXACTAMENTE
        los mismos átomos (no cualquier superset)."""
        target = current | {atom}
        return self._memory.get(target) > 0

    @staticmethod
    def _infer_domain(value: Any) -> AtomDomain | None:
        """Inferencia de dominio por tipo del valor."""
        if isinstance(value, bool):
            return None  # bool no es número cognitivo
        if isinstance(value, str):
            return AtomDomain.TEXT
        if isinstance(value, int):
            return AtomDomain.NUMBER
        if isinstance(value, (list, tuple)):
            return AtomDomain.IMAGE
        if isinstance(value, Mapping):
            return AtomDomain.MULTIMODAL
        return None

    @staticmethod
    def _concept_atom(value: Any, domain: AtomDomain) -> str | None:
        """Devuelve la surface-form del átomo-concepto de un input.

        Convenciones (prefijos para evitar colisión con sub-átomos):
            - TEXT:    "W:<palabra>" (e.g. "W:perro"), minúsculas.
            - NUMBER:  "N:<n>" (e.g. "N:12"), con signo explícito.
            - IMAGE:   "I:<hash8>" sobre la rejilla cuantizada.
            - MULTIMODAL: no se crea aquí (lo hace el fusion extractor).
        """
        if domain == AtomDomain.TEXT and isinstance(value, str):
            cleaned = value.strip().lower()
            return f"W:{cleaned}" if cleaned else None
        if domain == AtomDomain.NUMBER and isinstance(value, int):
            return f"N:{value}"
        if domain == AtomDomain.IMAGE and isinstance(value, (list, tuple)):
            try:
                # Hash determinista sobre los valores cuantizados.
                flat = []
                for row in value:
                    for v in row:
                        flat.append(int(v) >> 4)
                payload = ",".join(str(x) for x in flat).encode("utf-8")
                import hashlib
                return f"I:{hashlib.sha1(payload).hexdigest()[:8]}"
            except (TypeError, ValueError):
                return None
        return None
