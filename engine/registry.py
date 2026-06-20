"""
Registro bijectivo de átomos (Pilar 1: identidad portable de Fodor).

El registro de v1 usaba un único contador global de enteros. Eso bastaba
porque solo había texto. Para soportar múltiples dominios (texto, imagen,
número, multimodal) sin colisiones cross-domain, v2 mantiene un contador
por dominio: el `AtomId` resultante es (domain, local_id).

La biyección se preserva DENTRO de cada dominio:
    surface_form <-> AtomId   (único, idempotente)

Principio SOLID aplicado:
    - Single Responsibility: el registro solo mapea superficies a IDs y
      mantiene la información inversa.
    - Open/Closed: añadir un nuevo dominio NO requiere modificar el registro;
      basta con pasar un `AtomDomain` nuevo al `register`.
"""

from __future__ import annotations

from .atom import Atom, AtomDomain, AtomId


class AtomRegistry:
    """Registro bijectivo de átomos, con espacios disjuntos por dominio."""

    def __init__(self) -> None:
        # Doble índice por dominio:
        #   surface -> AtomId
        #   AtomId  -> surface
        self._surface_to_id: dict[tuple[AtomDomain, str], AtomId] = {}
        self._id_to_surface: dict[AtomId, str] = {}
        # Contador independiente por dominio.
        self._counters: dict[AtomDomain, int] = {d: 0 for d in AtomDomain}

    # --- API principal ---

    def register(self, surface_form: str, domain: AtomDomain) -> AtomId:
        """Devuelve el AtomId de `surface_form` en `domain`, creándolo si hace falta.

        Idempotente: registrar la misma superficie dos veces devuelve el
        mismo AtomId.
        """
        if not isinstance(surface_form, str) or not surface_form:
            raise ValueError("surface_form debe ser string no vacío.")
        if not isinstance(domain, AtomDomain):
            raise TypeError(f"domain debe ser AtomDomain, recibí {type(domain).__name__}")
        key = (domain, surface_form)
        existing = self._surface_to_id.get(key)
        if existing is not None:
            return existing
        local_id = self._counters[domain]
        self._counters[domain] = local_id + 1
        atom_id = AtomId(domain=domain, local_id=local_id)
        self._surface_to_id[key] = atom_id
        self._id_to_surface[atom_id] = surface_form
        return atom_id

    def register_atom(self, atom: Atom) -> AtomId:
        """Variante que acepta un `Atom` completo."""
        return self.register(atom.surface_form, atom.domain)

    # --- Consultas ---

    def surface_of(self, atom_id: AtomId) -> str:
        """Devuelve la forma de superficie de un AtomId, o '<??>' si no existe."""
        return self._id_to_surface.get(atom_id, "<??>")

    def known_ids(self) -> set[AtomId]:
        """Conjunto completo de AtomIds registrados (todos los dominios)."""
        return set(self._id_to_surface.keys())

    def known_in_domain(self, domain: AtomDomain) -> set[AtomId]:
        """Conjunto de AtomIds registrados en un dominio concreto."""
        return {aid for aid in self._id_to_surface if aid.domain == domain}

    def known_surfaces(self) -> dict[AtomId, str]:
        """Copia superficial del índice inverso (para inspección)."""
        return dict(self._id_to_surface)

    def count_in_domain(self, domain: AtomDomain) -> int:
        """Cuántos átomos hay registrados en un dominio."""
        return self._counters.get(domain, 0)

    def __len__(self) -> int:
        """Total de átomos registrados (suma de todos los dominios)."""
        return sum(self._counters.values())
