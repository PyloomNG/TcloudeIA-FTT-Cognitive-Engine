"""
FTTCognitiveEngine v2: Fachada que orquesta los subsistemas.

La fachada NO contiene lógica cognitiva. Su único trabajo es coordinar
y presentar una API pública coherente con los 5 pilares:
    1. Fodor:    observe(**inputs) acepta átomos de cualquier dominio.
    2. Tee&Taylor: stats() / print_state() verifican rango 4 bits.
    3. Conjuntos: declare_library / intersect / union / is_in / libraries_of.
    4. Hebb:     observe() delega al trainer (one-shot).
    5. Meta:     infer_rules() produce reglas desde la memoria.

API extendida (auto-clustering + bucle de validación):
    - auto_cluster(payloads, threshold): agrupa observaciones por
      similitud de átomos estructurales (Jaccard). El motor CREA los
      conjuntos (CLUSTER_0, CLUSTER_1, ...) sin que el usuario les ponga
      nombre. El usuario luego les puede poner contexto y reglas.
    - set_library_context(name, description=, context=): adjunta
      definición en lenguaje natural y reglas arbitrarias a un cluster.
    - classify(payload): predice el cluster ganador y devuelve su
      descripción para que el motor "diga" qué es.
    - graph(): imprime la salida del entrenamiento como un GRAFO de
      conjuntos (Pilar 3: teoría de conjuntos, no matrices).
    - confirm_cluster / move_atom / dissolve_cluster: bucle de
      validación humana -> Hebb positivo o negativo -> re-clustering
      automático.

Principio SOLID aplicado:
    - Dependency Inversion: TODAS las dependencias se inyectan.
    - Interface Segregation: solo métodos públicos relevantes.
    - SRP por método: cada método hace UNA cosa (declarar, observar,
      consultar, inspeccionar).
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Iterable, Mapping

from .association import AssociationMemory
from .atom import Atom, AtomDomain, AtomId
from .library import Library
from .predictor import GeneralizedStructuralPredictor
from .quantization import Quantizer
from .registry import AtomRegistry
from .rules import Rule, RuleInferenceStrategy, RuleStore


class FTTCognitiveEngine:
    """Fachada principal del motor FTT v2 (agnóstico al dominio)."""

    def __init__(
        self,
        trainer: Any,  # GeneralizedHebbianTrainer (evitamos import circular)
        predictor: GeneralizedStructuralPredictor,
        rule_strategy: RuleInferenceStrategy,
        rule_store: RuleStore,
        registry: AtomRegistry,
        memory: AssociationMemory,
        libraries: Mapping[str, Library] | None = None,
        *,
        with_default_libraries: bool = True,
    ) -> None:
        self._trainer = trainer
        self._predictor = predictor
        self._rule_strategy = rule_strategy
        self._rule_store = rule_store
        self._registry = registry
        self._memory = memory
        # Índice interno de bibliotecas por nombre. La fuente de verdad
        # para el predictor también es este índice.
        self._libraries: dict[str, Library] = dict(libraries) if libraries else {}
        # Sincronizamos el predictor con las bibliotecas declaradas.
        for lib in self._libraries.values():
            self._predictor.register_library(lib)
        # Bibliotecas predeterminadas (letras y dígitos): el motor
        # nace con conocimiento "innato" sobre el alfabeto y los números.
        if with_default_libraries:
            self._install_default_libraries()

    def _install_default_libraries(self) -> None:
        """Declara las bibliotecas LETTERS, VOWELS, DIGITS, etc.

        Se ejecuta una sola vez al construir el motor. No sobrescribe
        bibliotecas pre-existentes que el usuario haya pasado al
        constructor.
        """
        from .defaults import default_atoms_by_library
        atoms_by_lib = default_atoms_by_library()
        for name, atoms in atoms_by_lib.items():
            if name in self._libraries:
                # Si el usuario pasó una versión custom, la respetamos.
                continue
            atom_ids: set[AtomId] = set()
            for atom in atoms:
                atom_ids.add(self._registry.register_atom(atom))
            library = Library(
                name=name,
                members=frozenset(atom_ids),
                measure=0,
                parents=(),
                description=None,
                context=(),
            )
            self._libraries[name] = library
            self._predictor.register_library(library)
        # Ahora añadimos las jerarquías (parents) sobre las bibliotecas
        # recién creadas. LETTERS contiene todos los átomos (unión de
        # LETTERS_UPPER y LETTERS_LOWER por transitividad).
        from .defaults import UPPERCASE_LETTERS, LOWERCASE_LETTERS, VOWELS_UPPER, VOWELS_LOWER
        all_letter_atoms = (
            [Atom(c, AtomDomain.TEXT) for c in UPPERCASE_LETTERS]
            + [Atom(c, AtomDomain.TEXT) for c in LOWERCASE_LETTERS]
        )
        if "LETTERS" not in self._libraries:
            letter_ids = {self._registry.register_atom(a) for a in all_letter_atoms}
            self._libraries["LETTERS"] = Library(
                name="LETTERS",
                members=frozenset(letter_ids),
                measure=0,
                parents=(),
                description="Conjunto de todas las letras del alfabeto (A-Z, a-z).",
                context=(("domain", "alphabet"), ("cardinality", "52")),
            )
            self._predictor.register_library(self._libraries["LETTERS"])
        # Aplicamos los parents: LETTERS_UPPER ⊂ LETTERS, etc.
        parent_map = {
            "LETTERS_UPPER": "LETTERS",
            "LETTERS_LOWER": "LETTERS",
            "VOWELS_UPPER": "LETTERS_UPPER",
            "VOWELS_LOWER": "LETTERS_LOWER",
            "CONSONANTS_UPPER": "LETTERS_UPPER",
        }
        for child_name, parent_name in parent_map.items():
            if child_name in self._libraries and parent_name in self._libraries:
                self._libraries[child_name] = self._libraries[child_name].with_parents(
                    (parent_name,)
                )
                self._predictor.register_library(self._libraries[child_name])

    # =============================================================
    # 1) Bibliotecas y multi-membresía
    # =============================================================

    def declare_library(
        self,
        name: str,
        atoms: Iterable[Atom] = (),
        *,
        parents: Iterable[str] = (),
    ) -> Library:
        """Crea (o reemplaza) una biblioteca con un conjunto de átomos.

        Los `Atom` se registran primero en el registry (idempotente).
        Devuelve la `Library` resultante y la registra en el índice
        interno y en el predictor.
        """
        atom_ids: set[AtomId] = set()
        for atom in atoms:
            if not isinstance(atom, Atom):
                raise TypeError(f"declare_library espera Atom, recibí {type(atom).__name__}")
            atom_ids.add(self._registry.register_atom(atom))
        library = Library.of(name, atom_ids, parents=parents)
        self._libraries[name] = library
        self._predictor.register_library(library)
        return library

    def add_membership(self, library: Library, atom: Atom) -> Library:
        """Añade un átomo a una biblioteca existente. Devuelve la NUEVA Library."""
        if library.name not in self._libraries:
            raise KeyError(f"Biblioteca desconocida: {library.name!r}")
        if not isinstance(atom, Atom):
            raise TypeError(f"add_membership espera Atom, recibí {type(atom).__name__}")
        atom_id = self._registry.register_atom(atom)
        new_lib = library.add(atom_id)
        self._libraries[library.name] = new_lib
        self._predictor.register_library(new_lib)
        return new_lib

    def remove_membership(self, library: Library, atom: Atom) -> Library:
        """Quita un átomo de una biblioteca existente. Devuelve la NUEVA Library."""
        if library.name not in self._libraries:
            raise KeyError(f"Biblioteca desconocida: {library.name!r}")
        if not isinstance(atom, Atom):
            raise TypeError(f"remove_membership espera Atom, recibí {type(atom).__name__}")
        atom_id = self._registry.register_atom(atom)
        new_lib = library.remove(atom_id)
        self._libraries[library.name] = new_lib
        self._predictor.register_library(new_lib)
        return new_lib

    def rename_library(self, old_name: str, new_name: str) -> Library:
        """Renombra una biblioteca existente. Devuelve la NUEVA Library."""
        if old_name not in self._libraries:
            raise KeyError(f"rename_library: desconocida {old_name!r}")
        if new_name in self._libraries:
            raise ValueError(f"rename_library: destino {new_name!r} ya existe")
        old = self._libraries.pop(old_name)
        renamed = Library(
            name=new_name,
            members=old.members,
            measure=old.measure,
            parents=old.parents,
            description=old.description,
            context=old.context,
        )
        self._libraries[new_name] = renamed
        # Sincronizar predictor: quitar la vieja, registrar la nueva.
        self._predictor.unregister_library(old_name)
        self._predictor.register_library(renamed)
        # También actualizamos las references a old_name en los parents
        # de otras bibliotecas.
        for name, lib in self._libraries.items():
            if old_name in lib.parents:
                new_parents = tuple(
                    new_name if p == old_name else p for p in lib.parents
                )
                self._libraries[name] = lib.with_parents(new_parents)
                self._predictor.register_library(self._libraries[name])
        return renamed

    # =============================================================
    # 2) Observación (agnóstica al dominio)
    # =============================================================

    def observe(self, **inputs: Any) -> None:
        """Observa un payload multimodal. Las claves son arbitrarias.

        Ejemplos:
            engine.observe(text="el perro come")
            engine.observe(image=rejilla_4x4)
            engine.observe(number=12)
            engine.observe(text="perro", image=grid, number=4)
        """
        if not inputs:
            return
        self._trainer.train(inputs)

    # =============================================================
    # 3) Metacognición
    # =============================================================

    def infer_rules(self) -> list[Rule]:
        """Ejecuta la metacognición: analiza la memoria y crea reglas nuevas."""
        new_rules = self._rule_strategy.infer(self._memory, self._registry)
        added: list[Rule] = []
        for rule in new_rules:
            if self._rule_store.add(rule):
                added.append(rule)
        return added

    # =============================================================
    # 4) Consultas (Pilar 3: teoría de conjuntos)
    # =============================================================

    def is_in(self, atom: Atom, library_name: str) -> bool:
        """∈ transitivo: True si `atom` está directa o transitivamente
        en la biblioteca `library_name` (siguiendo la cadena de parents)."""
        if not isinstance(atom, Atom):
            raise TypeError(f"is_in espera Atom, recibí {type(atom).__name__}")
        atom_id = self._registry.register_atom(atom)
        if library_name not in self._libraries:
            return False
        visited: set[str] = set()
        stack: list[str] = [library_name]
        while stack:
            current = stack.pop()
            if current in visited:
                continue
            visited.add(current)
            lib = self._libraries.get(current)
            if lib is None:
                continue
            if atom_id in lib.members:
                return True
            for parent in lib.parents:
                if parent not in visited:
                    stack.append(parent)
        return False

    def libraries_of(self, atom: Atom) -> list[Library]:
        """Multi-membresía: devuelve TODAS las bibliotecas en las que
        `atom` es miembro, transitivamente."""
        if not isinstance(atom, Atom):
            raise TypeError(f"libraries_of espera Atom, recibí {type(atom).__name__}")
        atom_id = self._registry.register_atom(atom)
        found: list[Library] = []
        seen: set[str] = set()
        # Para cada biblioteca, comprobamos transitivamente si contiene al átomo.
        for start_name, start_lib in self._libraries.items():
            if start_name in seen:
                continue
            visited: set[str] = set()
            stack: list[str] = [start_name]
            contains = False
            while stack:
                current = stack.pop()
                if current in visited:
                    continue
                visited.add(current)
                lib = self._libraries.get(current)
                if lib is None:
                    continue
                if atom_id in lib.members:
                    contains = True
                    break
                for parent in lib.parents:
                    if parent not in visited:
                        stack.append(parent)
            if contains:
                # Marcar todos los visitados como "ya procesados" para
                # no repetir la búsqueda desde cada uno.
                seen.update(visited)
                found.append(start_lib)
        # Orden estable: por nombre de biblioteca.
        found.sort(key=lambda lib: lib.name)
        return found

    def predict_libraries(self, **inputs: Any) -> list[Library]:
        """Predice las bibliotecas activadas por un payload multimodal.

        Devuelve la lista de `Library` (PRED(...)) ordenadas por medida
        descendente.
        """
        if not inputs:
            return []
        return self._predictor.predict(inputs)

    def intersect(self, lib_a: str, lib_b: str) -> Library:
        """Intersección ∩ de dos bibliotecas declaradas."""
        a = self._libraries.get(lib_a)
        b = self._libraries.get(lib_b)
        if a is None or b is None:
            raise KeyError(f"intersect: biblioteca desconocida ({lib_a!r}, {lib_b!r})")
        return a.intersects(b)

    def union(self, lib_a: str, lib_b: str) -> Library:
        """Unión ∪ de dos bibliotecas declaradas."""
        a = self._libraries.get(lib_a)
        b = self._libraries.get(lib_b)
        if a is None or b is None:
            raise KeyError(f"union: biblioteca desconocida ({lib_a!r}, {lib_b!r})")
        return a.union(b)

    # =============================================================
    # 5) Auto-clustering (Pilar 1 + Pilar 3: agrupación automática
    #    de observaciones por similitud de conjuntos de átomos)
    # =============================================================

    def auto_cluster(
        self,
        payloads: Iterable[Mapping[str, Any]],
        *,
        threshold: float = 0.5,
        observe: bool = True,
        discriminative_only: bool = True,
    ) -> list[Library]:
        """Agrupa observaciones en bibliotecas (CLUSTER_0, CLUSTER_1, ...).

        Algoritmo (Hebra + Teoría de Conjuntos, sin matrices):
            1. Para cada payload, se extrae el conjunto de átomos
               ESTRUCTURALES (no el concept-atom único) usando la misma
               pipeline del trainer.
            2. Si `observe=True`, el payload se observa primero: el motor
               aprende Hebbianamente (memoria + cuantización 4 bits).
            3. (Opcional) Si `discriminative_only=True`, se filtran los
               átomos que aparecen en >80% de los payloads (átomos
               "comunes" como LUM:MID, ASPECT:SQUARE) porque tienen
               poder discriminante cero. Esto evita que todas las
               imágenes caigan en un solo cluster.
            4. Se calcula la similitud de Jaccard entre cada par de
               payloads: |A ∩ B| / |A ∪ B|.
            5. Se agrupa por single-linkage: dos payloads caen en el
               mismo cluster si jaccard >= threshold (o transitivamente).
            6. Cada cluster se materializa como una `Library` con nombre
               CLUSTER_<n> y miembros = concept-atoms de los payloads
               que cayeron en él. Se eliminan clusters previos
               (CLUSTER_*) y se reinferencen las reglas.

        Devuelve la lista de clusters creados (en orden de aparición).

        El usuario NO necesita nombrar nada: el motor crea los conjuntos.
        Luego puede llamar `set_library_context(...)` para adjuntar
        descripción en lenguaje natural y reglas de inferencia.
        """
        # Materializamos la lista para permitir iteración múltiple.
        payload_list = [dict(p) for p in payloads]
        if not payload_list:
            return []
        # 0) Limpiamos clusters previos generados por el motor. Los
        #    que el usuario haya declarado con declare_library se
        #    preservan.
        self._purge_auto_clusters()

        # 1) Observamos cada payload (Hebb one-shot) y extraemos su
        #    firma estructural.
        signatures: list[tuple[set[AtomId], AtomId | None]] = []
        for p in payload_list:
            if observe:
                self._trainer.train(p)
            struct, concept = self._trainer.structural_atoms(p)
            signatures.append((struct, concept))

        # 2) Filtrado de átomos "comunes" (sin poder discriminante).
        if discriminative_only and signatures:
            # Calculamos la frecuencia de cada átomo estructural.
            freq: dict[AtomId, int] = defaultdict(int)
            for struct, _ in signatures:
                for a in struct:
                    freq[a] += 1
            n = len(signatures)
            # Umbral: un átomo es "común" si aparece en >= 80% de los
            # payloads. Estos átomos no discriminan entre clases.
            common_threshold = 0.8 * n
            filtered: list[tuple[set[AtomId], AtomId | None]] = []
            for struct, concept in signatures:
                filtered_struct = {a for a in struct if freq[a] < common_threshold}
                filtered.append((filtered_struct, concept))
            signatures = filtered

        # 3) Single-linkage clustering por Jaccard de átomos
        #    estructurales.
        n = len(payload_list)
        parent = list(range(n))

        def find(x: int) -> int:
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def union(a: int, b: int) -> None:
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[rb] = ra

        for i in range(n):
            for j in range(i + 1, n):
                sim = self._jaccard(signatures[i][0], signatures[j][0])
                if sim >= threshold:
                    union(i, j)

        # 4) Agrupamos por raíz y construimos las Libraries.
        groups: dict[int, list[int]] = defaultdict(list)
        for i in range(n):
            groups[find(i)].append(i)

        created: list[Library] = []
        for root, members in groups.items():
            cluster_id = len(created)
            name = f"CLUSTER_{cluster_id}"
            # Miembros del cluster: concept-atoms (identidad) UNION
            # átomos estructurales (firma para matching por similitud).
            # Sin los estructurales, el predictor no podría clasificar
            # un nuevo payload en este cluster.
            member_ids: set[AtomId] = set()
            concept_ids: set[AtomId] = set()
            for idx in members:
                struct, concept = signatures[idx]
                member_ids.update(struct)
                if concept is not None:
                    member_ids.add(concept)
                    concept_ids.add(concept)
            # Almacenamos los concept-atoms como contexto para que
            # `print_state` y `graph` puedan distinguirlos.
            library = Library(
                name=name,
                members=frozenset(member_ids),
                measure=0,
                parents=(),
                description=None,
                context=tuple(sorted(("concept", repr(self._registry.surface_of(a)))
                                     for a in concept_ids)),
            )
            self._libraries[name] = library
            self._predictor.register_library(library)
            created.append(library)

        # 5) Reinferimos reglas para que el motor se entere de los
        #    nuevos concept-atoms que vio en este batch.
        try:
            self._rule_strategy.infer(self._memory, self._registry)
        except Exception:
            # La inferencia puede fallar si no hay solapamiento; lo
            # toleramos en auto_cluster para no romper el flujo.
            pass
        return created

    def _purge_auto_clusters(self) -> None:
        """Elimina las bibliotecas cuyo nombre empieza por 'CLUSTER_'.

        Las bibliotecas declaradas por el usuario (declare_library) o
        con nombre custom se preservan.
        """
        auto_names = [n for n in self._libraries if n.startswith("CLUSTER_")]
        for n in auto_names:
            del self._libraries[n]
        # Sincronizamos el predictor: re-registramos las restantes.
        # (No hay método 'unregister' público; recreamos el índice.)
        # Usamos el truco de reasignar las bibliotecas restantes.
        for name, lib in list(self._libraries.items()):
            self._predictor.register_library(lib)

    @staticmethod
    def _jaccard(a: set[AtomId], b: set[AtomId]) -> float:
        if not a and not b:
            return 1.0  # dos firmas vacías son "iguales"
        if not a or not b:
            return 0.0
        inter = len(a & b)
        union = len(a | b)
        return inter / union if union else 0.0

    def cluster_for(self, payload: Mapping[str, Any]) -> Library | None:
        """Devuelve la biblioteca que mejor coincide con el payload.

        A diferencia de `predict_libraries` (que devuelve todas las
        bibliotecas activadas), esta devuelve SOLO la ganadora (mayor
        medida predicha). Funciona con cualquier biblioteca
        (CLUSTER_*, DIGIT_*, ANIMALES, ...), no solo clusters.

        Es defensiva: si la predicción apunta a una biblioteca que ya
        no existe en el motor (p.ej. tras un dissolve), la ignora.

        Para desempatar (varias bibliotecas con la misma medida
        cuantizada), elige la de MAYOR cantidad de átomos de la
        biblioteca presentes en el input (mayor solapamiento real, no
        solo la medida).
        """
        predicted = self.predict_libraries(**payload)
        if not predicted:
            return None
        input_atoms = self._predictor._extract_atoms(payload)
        best: tuple[Library, int] | None = None
        for winner in predicted:
            if not winner.name.startswith("PRED("):
                continue
            original_name = winner.name[len("PRED("):-1]
            if original_name not in self._libraries:
                continue
            original = self._libraries[original_name]
            overlap = len(original.members & input_atoms)
            if best is None or (overlap, winner.measure) > (best[1], 0):
                best = (original, overlap)
        return best[0] if best else None

    # =============================================================
    # 6) Contexto y lenguaje natural
    # =============================================================

    def set_library_context(
        self,
        library_name: str,
        *,
        description: str | None = None,
        context: Mapping[str, Any] | None = None,
    ) -> Library:
        """Adjunta descripción y/o contexto a una biblioteca existente.

        - `description`: definición en lenguaje natural
          (p.ej. "cara: parte frontal de la cabeza humana...").
        - `context`: dict libre para reglas, etiquetas, sinónimos, etc.
        """
        if library_name not in self._libraries:
            raise KeyError(f"Biblioteca desconocida: {library_name!r}")
        current = self._libraries[library_name]
        new_lib = current
        if description is not None:
            new_lib = new_lib.with_description(description)
        if context is not None:
            new_lib = new_lib.with_context(dict(context))
        self._libraries[library_name] = new_lib
        self._predictor.register_library(new_lib)
        return new_lib

    def classify(self, payload: Mapping[str, Any]) -> tuple[Library | None, str]:
        """Predice el cluster del payload y devuelve (cluster, descripción).

        Si el cluster ganador tiene una `description` adjunta, la
        devuelve. Si no, devuelve una frase genérica indicando que el
        motor no tiene definición para ese cluster.
        """
        if not payload:
            return None, ""
        cluster = self.cluster_for(payload)
        if cluster is None:
            return None, "(sin clasificación)"
        desc = cluster.description or "(sin descripción)"
        return cluster, desc

    def explain(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        """Devuelve la JUSTIFICACIÓN machine-readable de la clasificación.

        El motor "explica" por qué un payload cae en un cluster y no en
        otro. La salida es un dict con:
            - predicted_cluster: nombre del cluster ganador.
            - discriminator_atoms: átomos estructurales que están TANTO
              en el input como en el cluster ganador, y que NO están en
              los demás clusters candidatos. Son la "firma" que
              diferencia este cluster.
            - rule: una `Rule` (o None) que el motor infirió que explica
              la membresía.
            - alternatives: lista de otros clusters con su medida
              predicha y su solapamiento.
        """
        if not payload:
            return {"error": "payload vacío"}
        cluster = self.cluster_for(payload)
        if cluster is None:
            return {"predicted_cluster": None, "error": "sin clasificación"}
        input_atoms = self._predictor._extract_atoms(payload)
        # Atómos que están en el cluster ganador Y en el input.
        in_cluster = cluster.members & input_atoms
        # Atómos en el cluster ganador pero NO en otros clusters CLUSTER_*.
        other_cluster_atoms: set[AtomId] = set()
        for n, lib in self._libraries.items():
            if n != cluster.name and n.startswith("CLUSTER_"):
                other_cluster_atoms |= lib.members
        discriminators = sorted(
            (self._registry.surface_of(a) for a in in_cluster
             if a not in other_cluster_atoms)
        )
        # Reglas inferidas que mencionan átomos del input.
        rule = None
        for r in self._rule_store.all():
            if any(a in input_atoms for a in r.atoms):
                rule = r
                break
        # Alternativas
        alternatives = []
        for p in self.predict_libraries(**payload):
            if p.name.startswith("PRED(CLUSTER_") and p.name != f"PRED({cluster.name})":
                alt_name = p.name[len("PRED("):-1]
                alt_lib = self._libraries.get(alt_name)
                if alt_lib is not None:
                    overlap = len(alt_lib.members & input_atoms)
                    alternatives.append({
                        "name": alt_name,
                        "predicted_measure": p.measure,
                        "overlap_with_input": overlap,
                    })
        return {
            "predicted_cluster": cluster.name,
            "description": cluster.description,
            "input_atom_count": len(input_atoms),
            "cluster_atom_count": len(cluster.members),
            "overlap_count": len(in_cluster),
            "discriminator_atoms": discriminators,
            "rule": rule.name if rule else None,
            "alternatives": alternatives,
        }

    def operate(
        self,
        operation: str,
        *payloads: Mapping[str, Any],
    ) -> dict[str, Any]:
        """Operación aritmética entre payloads clasificados.

        Uso típico:
            engine.observe(image=image_of_1)
            engine.set_library_context("CLUSTER_X", context={"value": 1})
            engine.observe(image=image_of_2)
            engine.set_library_context("CLUSTER_Y", context={"value": 2})
            result = engine.operate("+", {"image": image_of_1}, {"image": image_of_1})
            # result = {"operands": [1, 1], "operator": "+", "result": 2}

        Las operaciones soportadas (aritmética hecha en Python, el motor
        solo CLASIFICA):
            +, -, *, /, sum, diff, product, division, max, min

        Si un operando no se puede clasificar o su cluster no tiene
        `value` en su contexto, se devuelve un error explicativo.
        """
        OPS: dict[str, Any] = {
            "+": lambda xs: sum(xs),
            "-": lambda xs: xs[0] - sum(xs[1:]),
            "*": lambda xs: __import__("functools").reduce(lambda a, b: a * b, xs),
            "/": lambda xs: xs[0] / xs[1] if len(xs) == 2 and xs[1] != 0 else float("inf"),
            "sum": lambda xs: sum(xs),
            "diff": lambda xs: xs[0] - xs[1] if len(xs) == 2 else 0,
            "product": lambda xs: __import__("functools").reduce(lambda a, b: a * b, xs),
            "division": lambda xs: xs[0] / xs[1] if len(xs) == 2 and xs[1] != 0 else float("inf"),
            "max": lambda xs: max(xs),
            "min": lambda xs: min(xs),
        }
        if operation not in OPS:
            return {
                "error": f"operador desconocido: {operation!r}. "
                f"Disponibles: {sorted(OPS.keys())}"
            }
        operand_values: list[float] = []
        operand_clusters: list[str] = []
        errors: list[str] = []
        for i, p in enumerate(payloads):
            cluster = self.cluster_for(p)
            if cluster is None:
                errors.append(f"operando {i+1}: no clasificado")
                continue
            ctx = cluster.context_dict()
            if "value" not in ctx:
                errors.append(
                    f"operando {i+1}: cluster {cluster.name!r} no tiene 'value' en su contexto. "
                    f"Use set_library_context(name, context={{'value': N}})."
                )
                continue
            try:
                operand_values.append(float(ctx["value"]))
                operand_clusters.append(cluster.name)
            except (ValueError, TypeError):
                errors.append(f"operando {i+1}: 'value' no es numérico: {ctx['value']!r}")
        if errors or not operand_values:
            return {
                "operator": operation,
                "operands": operand_values,
                "operand_clusters": operand_clusters,
                "errors": errors,
                "result": None,
            }
        try:
            result = OPS[operation](operand_values)
        except Exception as e:
            return {
                "operator": operation,
                "operands": operand_values,
                "operand_clusters": operand_clusters,
                "errors": [str(e)],
                "result": None,
            }
        return {
            "operator": operation,
            "operands": operand_values,
            "operand_clusters": operand_clusters,
            "errors": [],
            "result": result,
        }

    # =============================================================
    # 7) Bucle de validación humana -> Hebb + re-clustering
    # =============================================================

    def confirm_cluster(self, library_name: str) -> None:
        """El usuario confirma el cluster. Hebb positivo: refuerza +1
        la asociación de los concept-atoms del cluster."""
        lib = self._libraries.get(library_name)
        if lib is None:
            raise KeyError(f"confirm_cluster: desconocida {library_name!r}")
        if not lib.members:
            return
        # Reforzamos la asociación entre los concept-atoms del cluster
        # y los átomos estructurales que el motor vio. Como ya están
        # reforzados del train, sumar +1 satura la cuantización 4 bits
        # pero es la semántica correcta de "confirmado por humano".
        for atom in lib.members:
            # Buscamos asociaciones existentes que contengan este átomo
            # y reforzamos. Esto es Hebb: el humano "ve" el cluster y
            # confirma que esos átomos van juntos.
            self._memory.reinforce(lib.members, +1)

    def move_atom(
        self,
        atom: Atom,
        from_name: str,
        to_name: str,
    ) -> None:
        """Mueve un átomo de un cluster a otro. El usuario está diciendo:
        'este elemento estaba mal agrupado'."""
        if not isinstance(atom, Atom):
            raise TypeError(f"move_atom espera Atom, recibí {type(atom).__name__}")
        atom_id = self._registry.register_atom(atom)
        src = self._libraries.get(from_name)
        dst = self._libraries.get(to_name)
        if src is None or dst is None:
            raise KeyError(f"move_atom: biblioteca desconocida ({from_name!r} o {to_name!r})")
        if atom_id not in src.members:
            # El átomo no estaba en src; lo añadimos a dst igualmente
            # para reflejar la intención del usuario.
            self.add_membership(dst, atom)
            return
        # Quitamos de src, añadimos a dst, y reforzamos.
        self.remove_membership(src, atom)
        self.add_membership(dst, atom)
        # Hebb: refuerzo POSITIVO en dst, NEGATIVO en src para la
        # asociación entre los miembros previos de src. El Anti-Hebb
        # selectivo del trainer se encarga de podar.
        if src.members:
            self._memory.reinforce(src.members, -1)
        if dst.members:
            self._memory.reinforce(dst.members, +1)

    def differentiate(self, library_a: str, library_b: str) -> dict[str, Any]:
        """Calcula la diferencia estructural entre dos bibliotecas.

        Devuelve los átomos que están en A pero NO en B (lo que hace a A
        único), y viceversa. Es la base de las REGLAS MACHINE-READABLE
        que el motor puede usar para explicar por qué algo es de una
        clase y no de otra.

        El resultado es un dict con listas de superficies, no una frase
        en lenguaje natural: la máquina usa estos átomos discriminantes
        directamente en la búsqueda de asociaciones.
        """
        lib_a = self._libraries.get(library_a)
        lib_b = self._libraries.get(library_b)
        if lib_a is None or lib_b is None:
            raise KeyError(
                f"differentiate: biblioteca desconocida "
                f"({library_a!r} o {library_b!r})"
            )
        set_a = set(lib_a.members)
        set_b = set(lib_b.members)
        only_in_a = sorted(self._registry.surface_of(a) for a in set_a - set_b)
        only_in_b = sorted(self._registry.surface_of(a) for a in set_b - set_a)
        shared = sorted(self._registry.surface_of(a) for a in set_a & set_b)
        return {
            "library_a": library_a,
            "library_b": library_b,
            "only_in_a": only_in_a,
            "only_in_b": only_in_b,
            "shared": shared,
            "jaccard": (
                len(set_a & set_b) / len(set_a | set_b)
                if (set_a | set_b) else 0.0
            ),
        }

    def dissolve_cluster(self, library_name: str) -> None:
        """El usuario rechaza un cluster entero. Se elimina y se re-clustering.

        Pasos:
            1. Hebb negativo: refuerza -1 la asociación de los miembros.
            2. Elimina la biblioteca del índice interno Y del predictor
               (para evitar bibliotecas fantasma en futuras predicciones).
            3. Re-clustering: agrupa los concept-atoms previamente
               en este cluster usando `_recluster`.
        """
        lib = self._libraries.get(library_name)
        if lib is None:
            raise KeyError(f"dissolve_cluster: desconocida {library_name!r}")
        members = lib.members
        if members:
            self._memory.reinforce(members, -1)
        del self._libraries[library_name]
        # Importante: también quitamos del predictor, sino quedan
        # bibliotecas fantasma y PRED(X) apunta a X inexistente.
        self._predictor.unregister_library(library_name)
        # Re-clustering: re-observamos los concept-atoms huérfanos.
        self._recluster(library_name, dissolved_members=members)

    def _recluster(
        self,
        dissolved_name: str,
        dissolved_members: frozenset[AtomId],
    ) -> None:
        """Re-asigna los concept-atoms de un cluster disuelto.

        Estrategia: para cada miembro disuelto, lo añadimos al cluster
        existente cuyo solapamiento de átomos estructurales sea MAYOR.
        Si ninguno supera un umbral mínimo, se crea un nuevo cluster
        CLUSTER_X.
        """
        # Sin observaciones nuevas: usamos la memoria para reconstruir
        # las firmas estructurales. Como esto es costoso y el motor ya
        # tiene predict_libraries, delegamos en él: predecimos para
        # cada miembro y vemos en qué cluster cae por defecto.
        if not dissolved_members:
            return
        # Calculamos clusters existentes (post-purgue del disuelto).
        existing = [
            (name, lib)
            for name, lib in self._libraries.items()
            if name.startswith("CLUSTER_")
        ]
        next_id = len(self._libraries)
        for atom_id in dissolved_members:
            # Firma estructural tentativa: los átomos que co-ocurren
            # con este concept en la memoria. Si la memoria está
            # vacía, lo metemos en un nuevo cluster "huérfano".
            siblings = self._structural_siblings_of(atom_id)
            best_name: str | None = None
            best_sim: float = 0.0
            for cname, clib in existing:
                sim = self._jaccard(siblings, set(clib.members))
                if sim > best_sim:
                    best_sim = sim
                    best_name = cname
            if best_name is not None and best_sim > 0.0:
                target = self._libraries[best_name]
                new_lib = target.add(atom_id)
                self._libraries[best_name] = new_lib
                self._predictor.register_library(new_lib)
            else:
                new_name = f"CLUSTER_{next_id}"
                next_id += 1
                new_lib = Library(
                    name=new_name,
                    members=frozenset({atom_id}),
                    measure=0,
                    parents=(),
                )
                self._libraries[new_name] = new_lib
                self._predictor.register_library(new_lib)
                existing.append((new_name, new_lib))

    def _structural_siblings_of(self, atom: AtomId) -> set[AtomId]:
        """Devuelve los átomos que co-ocurren con `atom` en la memoria.

        Sirve como firma estructural "tardía" para re-asignar
        concept-atoms huérfanos a clusters existentes.
        """
        siblings: set[AtomId] = set()
        for assoc, _measure in self._memory.supersets_of(frozenset({atom}), -9):
            siblings.update(assoc)
        siblings.discard(atom)
        return siblings

    # =============================================================
    # 8) Salida gráfica (Pilar 3 como grafo de conjuntos)
    # =============================================================

    def to_graph(self) -> "CognitiveGraph":
        """Devuelve el estado del motor como un CognitiveGraph modificable.

        El usuario puede:
            - Agregar / quitar / renombrar nodos y aristas.
            - Editar descripciones (lenguaje natural).
            - Mover átomos entre nodos.
            - Exportar a texto, DOT (Graphviz) o JSON.
        Y luego re-aplicarlo al motor con `apply_graph(graph)`.
        """
        # Import local para evitar circular: graph importa engine.
        from .graph import CognitiveGraph
        return CognitiveGraph.from_engine(self)

    def apply_graph(self, graph: "CognitiveGraph") -> None:
        """Aplica un CognitiveGraph modificado al motor.

        Refleja los cambios del grafo (rename, remove_node, move_atom,
        set_label, etc.) en el estado del motor.
        """
        from .graph import CognitiveGraph  # noqa: F401 (import de tipo)
        current_names = set(self._libraries.keys())
        graph_names = {n.name for n in graph.nodes()}
        # Nodos eliminados en el grafo
        for name in current_names - graph_names:
            del self._libraries[name]
        # Nodos nuevos en el grafo
        engine_node_names = {n.name for n in self.to_graph().nodes()}
        for node in graph.nodes():
            if node.name not in self._libraries:
                # Crear biblioteca vacía nueva
                atom_ids: set[AtomId] = set()
                for surf in node.members:
                    # Intentar registrar en TEXT por defecto.
                    atom_ids.add(
                        self._registry.register(surf, AtomDomain.TEXT)
                    )
                self._libraries[node.name] = Library(
                    name=node.name,
                    members=frozenset(atom_ids),
                    measure=0,
                    parents=(),
                )
        # Sincronizar miembros, descripciones y parents
        for node in graph.nodes():
            if node.name not in self._libraries:
                continue
            current = self._libraries[node.name]
            atom_ids = set(current.members)
            current_surf = {self._registry.surface_of(a) for a in atom_ids}
            target_surf = set(node.members)
            # Añadir átomos nuevos
            for surf in target_surf - current_surf:
                atom_ids.add(self._registry.register(surf, AtomDomain.TEXT))
            # Quitar átomos eliminados
            to_remove = current_surf - target_surf
            if to_remove:
                remove_ids = {a for a in atom_ids
                              if self._registry.surface_of(a) in to_remove}
                atom_ids -= remove_ids
            # Parents: derivar de aristas 'parent'
            parents = tuple(
                e.target for e in graph.edges()
                if e.source == node.name and e.kind == "parent"
            )
            new_lib = Library(
                name=node.name,
                members=frozenset(atom_ids),
                measure=current.measure,
                parents=parents,
                description=node.description or current.description,
                context=current.context,
            )
            self._libraries[node.name] = new_lib
            self._predictor.register_library(new_lib)

    def graph(self) -> str:
        """Atajo: devuelve la versión textual del CognitiveGraph."""
        return self.to_graph().to_text()

    # =============================================================
    # 9) Inspección
    # =============================================================

    def print_state(self) -> None:
        """Imprime el estado del motor en formato genérico (agnóstico al dominio)."""
        print("\n--- ESTADO DE MEDIDAS (no-cero) ---")
        measures = self._memory.all_measures()
        if not measures:
            print("  (memoria vacía)")
        else:
            for atoms, m in sorted(
                measures.items(), key=lambda kv: (-kv[1], len(kv[0]))
            ):
                surfaces = sorted(self._registry.surface_of(a) for a in atoms)
                domains = sorted({a.domain.value for a in atoms})
                print(
                    f"  {{{', '.join(surfaces)}}} "
                    f"<{','.join(domains)}> | medida: {m:+d}"
                )

        print("\n--- BIBLIOTECAS DECLARADAS ---")
        if not self._libraries:
            print("  (ninguna)")
        else:
            for name, lib in sorted(self._libraries.items()):
                print(f"  {lib}")

        print("\n--- REGLAS INFERIDAS ---")
        rules = self._rule_store.all()
        if not rules:
            print("  (ninguna)")
        else:
            for rule in rules:
                atoms_surf = sorted(
                    self._registry.surface_of(a) for a in rule.atoms
                )
                ctx_surf = sorted(
                    self._registry.surface_of(a) for a in rule.shared_context
                )
                print(f"  {rule.name}: {atoms_surf}  ⇆  {ctx_surf}")

    def stats(self) -> dict[str, int]:
        """Estadísticas agregadas. Útil para asserts de aceptación."""
        measures = list(self._memory.all_measures().values())
        all_in_range = all(Quantizer.is_within_range(m) for m in measures)
        return {
            "atoms_total": len(self._registry),
            "atoms_text": self._registry.count_in_domain(AtomDomain.TEXT),
            "atoms_image": self._registry.count_in_domain(AtomDomain.IMAGE),
            "atoms_number": self._registry.count_in_domain(AtomDomain.NUMBER),
            "atoms_multimodal": self._registry.count_in_domain(AtomDomain.MULTIMODAL),
            "associations": len(self._memory),
            "libraries": len(self._libraries),
            "rules": len(self._rule_store),
            "all_measures_in_range": int(all_in_range),
        }

    # --- Acceso de solo lectura (para tests y depuración) ---

    @property
    def registry(self) -> AtomRegistry:
        return self._registry

    @property
    def memory(self) -> AssociationMemory:
        return self._memory

    @property
    def rule_store(self) -> RuleStore:
        return self._rule_store

    @property
    def libraries(self) -> Mapping[str, Library]:
        return dict(self._libraries)
