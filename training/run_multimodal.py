"""
Caso de uso 5: MULTIMODAL (texto + imagen + número en la misma observación).

Demuestra que una sola instancia de FTTCognitiveEngine ingiere payloads
mixtos y que el MultimodalFusionExtractor crea el átomo ancla CO_OCC:*
en el dominio MULTIMODAL.

Verifica que:
    - Se crea al menos un átomo con AtomDomain.MULTIMODAL.
    - El átomo CO_OCC es consultable.
    - La observación multimodal refuerza asociaciones multimodales.
"""

from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from engine import (  # noqa: E402
    AssociationMemory,
    AtomDomain,
    AtomRegistry,
    CoActivationRuleInference,
    FTTCognitiveEngine,
    GeneralizedHebbianTrainer,
    GeneralizedStructuralPredictor,
    Quantizer,
    RuleStore,
)


# Rejilla "perro" minimalista: patron 2x2 que simula una mancha
# de perro (centro blanco, esquinas negras).
DOG_GRID: list[list[int]] = [
    [0, 200, 200, 0],
    [200, 255, 255, 200],
    [200, 255, 255, 200],
    [0, 200, 200, 0],
]


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

    # === Observación multimodal ===
    # La misma instancia del motor ingiere texto, imagen y número en
    # la misma llamada a observe(**inputs). El MultimodalFusionExtractor
    # crea un átomo ancla CO_OCC:<hash> en el dominio MULTIMODAL.
    print("=== OBSERVACION MULTIMODAL ===")
    engine.observe(text="perro", image=DOG_GRID, number=4)
    engine.observe(text="gato", image=DOG_GRID, number=4)
    engine.observe(text="perro", number=4)
    print("  3 observaciones multimodales registradas")

    # === Verificar que se creó al menos un átomo MULTIMODAL ===
    stats = engine.stats()
    print(f"\n=== ESTADISTICAS ===")
    for k, v in stats.items():
        print(f"  {k}: {v}")
    assert stats["atoms_multimodal"] >= 1, (
        f"Deberia haber al menos 1 atomo MULTIMODAL, stats={stats}"
    )
    assert stats["all_measures_in_range"] == 1, (
        f"Medidas fuera de rango: {stats}"
    )

    # === Inspeccionar los atomos multimodales creados ===
    print("\n=== ATOMOS MULTIMODALES REGISTRADOS ===")
    multi_atoms = engine.registry.known_in_domain(AtomDomain.MULTIMODAL)
    for atom in multi_atoms:
        surface = engine.registry.surface_of(atom)
        print(f"  - {surface}  (AtomId={atom})")

    # Debe haber al menos un CO_OCC:*
    co_occ_atoms = [
        a for a in multi_atoms
        if engine.registry.surface_of(a).startswith("CO_OCC:")
    ]
    assert co_occ_atoms, (
        f"Deberia haber al menos un atomo CO_OCC:*, encontrados: "
        f"{[engine.registry.surface_of(a) for a in multi_atoms]}"
    )

    # === Verificar que el motor aprendio asociaciones multimodales ===
    print("\n=== ASOCIACIONES CON ATOMOS MULTIMODALES ===")
    co_atom = co_occ_atoms[0]
    co_surface = engine.registry.surface_of(co_atom)
    print(f"  buscando asociaciones con {co_surface}")
    for atoms, m in engine.memory.all_measures().items():
        if co_atom in atoms:
            other = sorted(
                engine.registry.surface_of(a) for a in atoms if a != co_atom
            )
            print(f"  medida={m:+d}  {{ {co_surface}, {', '.join(other)} }}")

    # === Predicción multimodal ===
    # Predecir dado (text="perro", image=DOG_GRID, number=4).
    # Las asociaciones multimodales deberian ser las que mas se activen.
    print("\n=== PREDICCION MULTIMODAL ===")
    preds = engine.predict_libraries(text="perro", image=DOG_GRID, number=4)
    if preds:
        for p in preds:
            members = sorted(
                engine.registry.surface_of(a) for a in p.members
            )
            print(f"  -> {p.name} medida={p.measure:+d} members={members}")
    else:
        print("  (sin prediccion)")

    # === Inspección general ===
    print("\n--- ESTADO COMPLETO ---")
    engine.print_state()


if __name__ == "__main__":
    main()
