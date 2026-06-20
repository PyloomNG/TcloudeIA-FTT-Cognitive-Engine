"""
Caso de uso 2: NUMEROS.

Ingiere la secuencia 1..30 declarando bibliotecas aritmeticas
(EVEN, ODD, MULTIPLE_OF_3, PRIME) y demuestra que:
    - libraries_of(N:12) devuelve las bibliotecas correctas.
    - libraries_of(N:7) devuelve PRIME (no EVEN, no MULTIPLE_OF_3).
    - Toda medida se mantiene en [-8, 7].
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

    # === Declarar bibliotecas aritmeticas ===
    # Para numeros, el "concepto" tiene prefijo "N:" (definido en el trainer).
    # Las bibliotecas de CATEGORIA (EVEN, ODD, ...) se declaran usando
    # los atomos MAG:*, PRIME:*, DIG:* que produce el NumberFactorExtractor.
    evens = {Atom(f"N:{n}", AtomDomain.NUMBER) for n in range(2, 31, 2)}
    odds = {Atom(f"N:{n}", AtomDomain.NUMBER) for n in range(1, 31, 2)}
    multiples_of_3 = {Atom(f"N:{n}", AtomDomain.NUMBER) for n in range(3, 31, 3)}
    # Primos <= 30
    primes_set = {2, 3, 5, 7, 11, 13, 17, 19, 23, 29}
    primes = {Atom(f"N:{n}", AtomDomain.NUMBER) for n in primes_set}

    engine.declare_library("EVEN", evens)
    engine.declare_library("ODD", odds)
    engine.declare_library("MULTIPLE_OF_3", multiples_of_3)
    engine.declare_library("PRIME", primes)

    # === Exposición: numeros 1..30 ===
    print("=== ENTRENAMIENTO NUMERICO (1..30) ===")
    for n in range(1, 31):
        engine.observe(number=n)
    print(f"  {len(range(1, 31))} numeros observados")

    # === Verificación Pilar 2: rango 4 bits ===
    stats = engine.stats()
    assert stats["all_measures_in_range"] == 1, stats
    assert stats["atoms_number"] > 0, "Deberia haber atomos NUMBER"

    # === Multi-membresia: libraries_of(N:12) ===
    n12 = Atom("N:12", AtomDomain.NUMBER)
    libs_12 = engine.libraries_of(n12)
    names_12 = sorted(l.name for l in libs_12)
    print("\n=== libraries_of(N:12) ===")
    for lib in libs_12:
        print(f"  - {lib.name} (|C|={lib.size()})")
    assert "EVEN" in names_12, f"N:12 deberia estar en EVEN, obtuve {names_12}"
    assert "MULTIPLE_OF_3" in names_12, (
        f"N:12 deberia estar en MULTIPLE_OF_3, obtuve {names_12}"
    )
    assert "ODD" not in names_12, f"N:12 NO deberia estar en ODD, obtuve {names_12}"
    assert "PRIME" not in names_12, f"N:12 NO deberia estar en PRIME, obtuve {names_12}"

    # === libraries_of(N:7) ===
    n7 = Atom("N:7", AtomDomain.NUMBER)
    libs_7 = engine.libraries_of(n7)
    names_7 = sorted(l.name for l in libs_7)
    print("\n=== libraries_of(N:7) ===")
    for lib in libs_7:
        print(f"  - {lib.name} (|C|={lib.size()})")
    assert "PRIME" in names_7, f"N:7 deberia estar en PRIME, obtuve {names_7}"
    assert "EVEN" not in names_7, f"N:7 NO deberia estar en EVEN, obtuve {names_7}"
    assert "ODD" in names_7, f"N:7 deberia estar en ODD, obtuve {names_7}"

    # === Prediccion ===
    print("\n=== PREDICCION NUMERICA ===")
    for n in (12, 7, 15, 8, 4, 9, 2, 1):
        atom = Atom(f"N:{n}", AtomDomain.NUMBER)
        preds = engine.predict_libraries(number=n)
        if not preds:
            print(f"  N:{n:<3} -> (sin prediccion)")
        else:
            top = preds[0]
            members = sorted(engine.registry.surface_of(a) for a in top.members)
            print(
                f"  N:{n:<3} -> top: {top.name} medida={top.measure:+d} "
                f"members={members}"
            )

    # === Resumen ===
    print("\n=== RESUMEN ===")
    for k, v in stats.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
