"""
Demo: reconocimiento de digitos por imagen + aritmetica por conjuntos.

Caso de uso:
    1. Enviamos 20 imagenes: cada digito 0-9 en DOS tipografias.
    2. El motor las AGRUPA AUTOMATICAMENTE: 10 clusters (uno por
       digito), porque los rasgos visuales son compartidos entre
       tipografias de la misma cifra.
    3. El usuario (o este script) pone nombre y VALOR NUMERICO a cada
       cluster: set_library_context(name, context={'value': N}).
    4. Se demuestra la operacion aritmetica:
            engine.operate('+', {image: digit_1}, {image: digit_1}) -> 2
            engine.operate('*', {image: digit_3}, {image: digit_4}) -> 12
       El motor CLASIFICA cada imagen, busca el valor en el contexto
       de la biblioteca, y Python hace la aritmetica.
    5. El motor "explica" por que una imagen es un 1 y no un 7
       (atoms discriminantes).
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
    CognitiveGraph,
    CoActivationRuleInference,
    FTTCognitiveEngine,
    GeneralizedHebbianTrainer,
    GeneralizedStructuralPredictor,
    Quantizer,
    RuleStore,
)


SAMPLES = _ROOT / "training" / "samples"


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


def payload_for(digit: int, style: str) -> dict:
    return {
        "image": str(SAMPLES / f"digit_{digit}_{style}.png"),
        "text": f"digit {digit}",
    }


def main() -> None:
    print("=" * 70)
    print("DEMO: Reconocimiento de digitos + aritmetica por conjuntos")
    print("=" * 70)
    engine = build_engine()

    # === 1. Construir 20 observaciones (10 digitos x 2 tipografias) ===
    print(f"\n=== PASO 1: 20 imagenes ingestadas (digitos 0-9, 2 tipografias) ===")
    payloads = []
    for d in range(10):
        for style in ("serif", "sans"):
            payloads.append(payload_for(d, style))
            print(f"  - digit_{d}_{style}.png")

    # === 2. Auto-clustering ===
    print(f"\n=== PASO 2: auto_cluster (threshold=0.4) ===")
    clusters = engine.auto_cluster(payloads, threshold=0.4, discriminative_only=True)
    print(f"Clusters creados: {len(clusters)}")
    for c in clusters:
        concepts = [v for k, v in c.context if k == "concept"]
        print(f"  {c.name}: {len(c.members)} atomos, conceptos={concepts[:3]}")

    # === 3. Renombrar clusters a DIGIT_<N> con su valor numerico ===
    print(f"\n=== PASO 3: renombrar clusters y asignar valor numerico ===")
    # Como no sabemos que cluster corresponde a que digito, hacemos una
    # pasada: clasificamos una imagen canonica de cada digito, y el
    # cluster ganador se renombra.
    for d in range(10):
        # Usamos la imagen serif como referencia
        test = payload_for(d, "serif")
        cluster = engine.cluster_for(test)
        if cluster is not None:
            new_name = f"DIGIT_{d}"
            # Renombrar la biblioteca directamente
            engine.rename_library(cluster.name, new_name)
            # Adjuntamos descripcion y value al contexto
            engine.set_library_context(
                new_name,
                description=f"digito {d}: patrones visuales de la cifra {d}",
                context={"value": d, "type": "digit"},
            )
            print(f"  digito {d} -> {cluster.name} renombrado a {new_name}, value={d}")
        else:
            print(f"  digito {d} -> (sin clasificar)")

    # === 4. explain: por que esto es un 1 y no un 7 ===
    print(f"\n=== PASO 4: explain (por que esto es un 1?) ===")
    test1 = payload_for(1, "sans")
    explanation = engine.explain(test1)
    print(f"  Payload: digit_1_sans.png")
    print(f"  Cluster predicho: {explanation['predicted_cluster']}")
    print(f"  Solapamiento: {explanation['overlap_count']}/{explanation['cluster_atom_count']}")
    print(f"  Atomos discriminantes: {explanation['discriminator_atoms'][:5]}")
    if explanation.get("alternatives"):
        print(f"  Alternativas: {explanation['alternatives'][:3]}")

    # === 5. Aritmetica: 1 + 1 = 2 ===
    print(f"\n=== PASO 5: aritmetica (1 + 1 = 2) ===")
    result = engine.operate("+", payload_for(1, "serif"), payload_for(1, "sans"))
    print(f"  1 + 1 = {result['result']}")
    print(f"  operandos: {result['operands']} (clusters: {result['operand_clusters']})")
    assert result["result"] == 2, f"esperado 2, obtuve {result['result']}"

    result = engine.operate("+", payload_for(1, "serif"), payload_for(2, "serif"))
    print(f"  1 + 2 = {result['result']}")
    assert result["result"] == 3, f"esperado 3, obtuve {result['result']}"

    result = engine.operate("*", payload_for(3, "serif"), payload_for(4, "serif"))
    print(f"  3 * 4 = {result['result']}")
    assert result["result"] == 12, f"esperado 12, obtuve {result['result']}"

    result = engine.operate("sum", payload_for(5, "serif"), payload_for(7, "serif"), payload_for(2, "serif"))
    print(f"  sum(5, 7, 2) = {result['result']}")
    assert result["result"] == 14

    result = engine.operate("max", payload_for(9, "serif"), payload_for(3, "serif"))
    print(f"  max(9, 3) = {result['result']}")

    # === 6. differentiate: por que 1 y 7 son diferentes? ===
    print(f"\n=== PASO 6: differentiate (atomos unicos de 1 vs 7) ===")
    diff = engine.differentiate("DIGIT_1", "DIGIT_7")
    print(f"  Solo en DIGIT_1: {diff['only_in_a'][:5]}")
    print(f"  Solo en DIGIT_7: {diff['only_in_b'][:5]}")
    print(f"  Compartidos: {len(diff['shared'])} atomos")
    print(f"  Jaccard: {diff['jaccard']:.2f}")

    # === 7. Resumen ===
    print(f"\n=== RESUMEN ===")
    stats = engine.stats()
    for k, v in stats.items():
        print(f"  {k}: {v}")
    print(f"\nOperacion clave: el motor NO hace la aritmetica (la hace Python).")
    print(f"Lo que el motor hace es CLASIFICAR cada imagen en su cluster,")
    print(f"y leer el 'value' que el usuario asoció al cluster.")


if __name__ == "__main__":
    main()
