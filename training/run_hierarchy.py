"""
Caso de uso 4: JERARQUIA TRANSITIVA.

Demostracion focal del caso "perro" como multi-miembro de PERRO,
MAMIFERO y ANIMAL. La jerarquia se construye con la declaracion
EXPLICITA de parents (no se infiere como side-effect). Verifica que:
    - libraries_of(W:perro) devuelve las 3 bibliotecas.
    - is_in(W:perro, "ANIMAL") == True (por transitividad).
    - add_membership / remove_membership devuelven NUEVAS Library.
"""

from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from engine import (  # noqa: E402
    AssociationMemory,
    Atom,
    AtomDomain,
    AtomRegistry,
    CoActivationRuleInference,
    FTTCognitiveEngine,
    GeneralizedHebbianTrainer,
    GeneralizedStructuralPredictor,
    Quantizer,
    RuleStore,
)


def build_engine() -> FTTCognitiveEngine:
    registry = AtomRegistry()
    memory = AssociationMemory(Quantizer)
    rule_store = RuleStore()
    trainer = GeneralizedHebbianTrainer(memory, registry)
    predictor = GeneralizedStructuralPredictor(memory, registry, rule_store)
    strategy = CoActivationRuleInference()
    return FTTCognitiveEngine(
        trainer=trainer,
        predictor=predictor,
        rule_strategy=strategy,
        rule_store=rule_store,
        registry=registry,
        memory=memory,
    )


def main() -> None:
    engine = build_engine()

    # === Declarar la jerarquia explícita ===
    perro = Atom("W:perro", AtomDomain.TEXT)
    gato = Atom("W:gato", AtomDomain.TEXT)
    ballena = Atom("W:ballena", AtomDomain.TEXT)

    # 1) Nivel mas especifico: PERRO contiene solo a perro
    perro_lib = engine.declare_library("PERRO", {perro})

    # 2) Nivel medio: MAMIFERO contiene a perro, gato y ballena
    mamifero_lib = engine.declare_library(
        "MAMIFERO", {perro, gato, ballena}, parents=()
    )

    # 3) Nivel mas general: ANIMAL contiene a MAMIFERO (transitivamente)
    animal_lib = engine.declare_library(
        "ANIMAL", {gato, ballena}, parents=("MAMIFERO",)
    )

    # === Multi-membresia transitiva ===
    libs_perro = engine.libraries_of(perro)
    names_perro = sorted(l.name for l in libs_perro)
    print("=== libraries_of(W:perro) ===")
    for lib in libs_perro:
        print(f"  - {lib.name} (|C|={lib.size()}, parents={lib.parents})")
    assert "PERRO" in names_perro, f"perro debe estar en PERRO, libs={names_perro}"
    assert "MAMIFERO" in names_perro, f"perro debe estar en MAMIFERO, libs={names_perro}"
    assert "ANIMAL" in names_perro, (
        f"perro debe estar en ANIMAL por transitividad, libs={names_perro}"
    )

    # === is_in transitivo ===
    assert engine.is_in(perro, "PERRO"), "perro debe estar en PERRO (directo)"
    assert engine.is_in(perro, "MAMIFERO"), "perro debe estar en MAMIFERO (directo)"
    assert engine.is_in(perro, "ANIMAL"), (
        "perro debe estar en ANIMAL por transitividad (via MAMIFERO)"
    )

    # === Casos negativos ===
    assert not engine.is_in(perro, "INEXISTENTE"), (
        "is_in a biblioteca inexistente debe ser False"
    )
    ballena = Atom("W:ballena", AtomDomain.TEXT)
    assert engine.is_in(ballena, "MAMIFERO"), "ballena debe estar en MAMIFERO (directo)"
    assert engine.is_in(ballena, "ANIMAL"), "ballena debe estar en ANIMAL (directo)"

    # === add_membership / remove_membership devuelven NUEVA Library ===
    print("\n=== INMUTABILIDAD (add/remove) ===")
    print(f"  antes: PERRO size = {perro_lib.size()}")
    gato_lib_agregado = engine.add_membership(perro_lib, gato)
    print(f"  add_membership(gato) -> size = {gato_lib_agregado.size()}")
    # La original NO cambia.
    assert perro_lib.size() == 1, (
        f"perro_lib original no debe cambiar, size={perro_lib.size()}"
    )
    # La nueva es diferente y contiene a gato.
    assert gato_lib_agregado.size() == 2, (
        f"gato_lib_agregado debe tener size=2, size={gato_lib_agregado.size()}"
    )

    # Verificar que el motor ve la nueva membresía.
    libs_perro_after = engine.libraries_of(perro)
    # perro debe seguir en PERRO (con la nueva lib que ahora tiene size=2)
    new_perro_lib = next(l for l in libs_perro_after if l.name == "PERRO")
    assert new_perro_lib.size() == 2, (
        f"PERRO ahora debe tener size=2, size={new_perro_lib.size()}"
    )

    # Quitamos a perro de PERRO.
    perro_removed = engine.remove_membership(perro_lib, perro)
    assert perro_removed.size() == 0, "despues de remove, size=0"
    print(f"  remove_membership(perro) -> size = {perro_removed.size()}")

    # === Resumen ===
    print("\n=== RESUMEN ===")
    stats = engine.stats()
    for k, v in stats.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
