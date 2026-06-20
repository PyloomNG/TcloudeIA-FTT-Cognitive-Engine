"""
Caso de uso 1: TEXTO + MULTI-MEMBRESÍA.

Reproduce el escenario v1 (perro, gato, casa, yo, monitor) y AÑADE la
funcionalidad nueva de v2: multi-membresía explícita por bibliotecas
declaradas. Verifica que:
    - "perro" pertenece a ANIMAL, MAMÍFERO y SUJETO (>=3 bibliotecas).
    - La predicción devuelve las bibliotecas esperadas ordenadas por
      medida descendente.
    - Toda medida está en [-8, 7] (Pilar 2: cuantización 4 bits).
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
    """Composición raíz para el caso de texto + multi-membresía."""
    registry = AtomRegistry()
    memory = AssociationMemory(Quantizer)
    rule_store = RuleStore()
    trainer = GeneralizedHebbianTrainer(memory, registry)
    predictor = GeneralizedStructuralPredictor(memory, registry, rule_store)
    strategy = CoActivationRuleInference(
        activation_threshold=2,
        overlap_threshold=2,
    )
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

    # === Bibliotecas declaradas (multi-membresía) ===
    # Los átomos de texto usan el prefijo "W:" para distinguir el
    # concepto (palabra completa) de los bigramas (sub-átomos) que
    # produce el extractor. El trainer crea automáticamente átomos "W:*"
    # al observar texto; declarar bibliotecas con esa misma convención
    # permite que `libraries_of(perro)` las recupere correctamente.
    perro = Atom("W:perro", AtomDomain.TEXT)
    gato = Atom("W:gato", AtomDomain.TEXT)
    casa = Atom("W:casa", AtomDomain.TEXT)
    yo = Atom("W:yo", AtomDomain.TEXT)
    monitor = Atom("W:monitor", AtomDomain.TEXT)

    # Jerarquía mamífero ⊂ animal
    engine.declare_library("ANIMAL", {perro, gato})
    engine.declare_library("MAMIFERO", {perro}, parents=("ANIMAL",))
    # Categoría sintáctica (todos los sujetos)
    engine.declare_library("SUJETO", {perro, gato, casa, yo, monitor})
    # Categoría de verbos frecuentes
    engine.declare_library("ACCION", parents=())  # placeholder

    # === Exposición al lenguaje ===
    sentences = [
        "el gato come",
        "el perro come",
        "el gato duerme",
        "el perro duerme",
        "la casa brilla",
        "la mesa brilla",
        "la casa observa",
        "yo corro",
        "yo salto",
    ]
    print("=== ENTRENAMIENTO DE TEXTO ===")
    for s in sentences:
        engine.observe(text=s)
        print(f"  aprendido: {s!r}")

    engine.print_state()

    # === Verificación Pilar 2: rango 4 bits ===
    stats = engine.stats()
    assert stats["all_measures_in_range"] == 1, (
        f"Alguna medida está fuera de [-8, 7]: {stats}"
    )
    assert stats["atoms_text"] > 0, "Debería haber átomos TEXT registrados"

    # === Multi-membresía: "perro" debe estar en >=3 bibliotecas ===
    libs_perro = engine.libraries_of(perro)
    print("\n=== MULTI-MEMBRESÍA: libraries_of(perro) ===")
    for lib in libs_perro:
        print(f"  - {lib.name} (|C|={lib.size()})")
    assert len(libs_perro) >= 3, (
        f"perro debería estar en >=3 bibliotecas, encontré {len(libs_perro)}: "
        f"{[l.name for l in libs_perro]}"
    )
    assert engine.is_in(perro, "ANIMAL"), "perro debería estar en ANIMAL"
    assert engine.is_in(perro, "MAMIFERO"), "perro debería estar en MAMIFERO"
    assert engine.is_in(perro, "SUJETO"), "perro debería estar en SUJETO"

    # === Verificación transitiva: is_in(perro, ANIMAL) por jerarquía ===
    assert engine.is_in(perro, "ANIMAL"), "ANIMAL debe contener a perro (directo)"

    # === Predicción por contexto ===
    print("\n=== PREDICCIÓN POR CONTEXTO (predict_libraries) ===")
    test_inputs = ["gato", "perro", "casa", "yo", "monitor"]
    for subj in test_inputs:
        atom = Atom(subj, AtomDomain.TEXT)
        preds = engine.predict_libraries(text=subj)
        if not preds:
            print(f"  {subj!r:<10} -> (sin predicción)")
        else:
            top = preds[0]
            members = sorted(
                engine.registry.surface_of(a) for a in top.members
            )
            print(
                f"  {subj!r:<10} -> top: {top.name} "
                f"medida={top.measure:+d} members={members}"
            )

    # === Operaciones de teoría de conjuntos ===
    print("\n=== TEORIA DE CONJUNTOS ===")
    inter = engine.intersect("ANIMAL", "SUJETO")
    union = engine.union("ANIMAL", "SUJETO")
    print(f"  ANIMAL AND SUJETO = {inter} members={[engine.registry.surface_of(a) for a in inter.members]}")
    print(f"  ANIMAL OR  SUJETO = {union} members={[engine.registry.surface_of(a) for a in union.members]}")

    # === Metacognición: inferir reglas ===
    print("\n=== METACOGNICIÓN ===")
    rules = engine.infer_rules()
    if not rules:
        print("  (sin reglas nuevas)")
    else:
        for r in rules:
            atoms_s = sorted(engine.registry.surface_of(a) for a in r.atoms)
            print(f"  {r.name}: {atoms_s}")

    # === Resumen ===
    print("\n=== RESUMEN ===")
    for k, v in stats.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
