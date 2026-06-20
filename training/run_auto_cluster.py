"""
Demo: auto-clustering + bucle de validacion humana + salida como grafo.

Caso de uso:
    1. Cargamos 12 imagenes: 3 botellas de plastico, 3 de vidrio,
       2 caras, 1 mano, 1 cuerpo, 2 latas.
    2. El motor las AGRUPA AUTOMATICAMENTE por similitud de atomos
       estructurales (sin que le digamos las clases).
    3. Imprimimos el grafo de bibliotecas (Pilar 3 como salida).
    4. El usuario itera: confirma, mueve atomos, disuelve clusters.
    5. El motor re-clustera despues de cada intervencion.
    6. Probamos `classify` para ver que el motor "dice" que es un
       nuevo payload, adjuntando la descripcion en lenguaje natural.
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


def payload_for(image_name: str) -> dict:
    """Crea un payload con la imagen + el nombre como texto."""
    return {
        "image": str(SAMPLES / image_name),
        "text": image_name.replace(".png", "").replace("_", " "),
    }


def main() -> None:
    print("=" * 70)
    print("DEMO: auto-clustering + grafo modificable + bucle de validacion")
    print("=" * 70)

    engine = build_engine()
    print(f"\nBibliotecas predeterminadas (Pilar 3 - conocimiento innato):")
    for name in ["LETTERS", "VOWELS_UPPER", "CONSONANTS_UPPER", "DIGITS"]:
        lib = engine.libraries[name]
        print(f"  {name}: |C|={len(lib.members)}, desc={lib.description!r}")

    # === 1. Construir el batch de observaciones ===
    images = [
        "bottle_plastic_red.png", "bottle_plastic_green.png", "bottle_plastic_blue.png",
        "bottle_glass_green.png", "bottle_glass_clear.png", "bottle_glass_amber.png",
        "can1.png", "can2.png",
        "face1.png", "face2.png",
        "hand.png", "body.png",
    ]
    payloads = [payload_for(name) for name in images]
    print(f"\n=== PASO 1: {len(payloads)} imagenes ingestadas ===")
    for p in payloads:
        print(f"  - {Path(p['image']).name}")

    # === 2. Auto-clustering ===
    print(f"\n=== PASO 2: auto_cluster (threshold=0.3) ===")
    clusters = engine.auto_cluster(payloads, threshold=0.3, discriminative_only=True)
    print(f"Clusters creados: {len(clusters)}")
    for c in clusters:
        # Extraemos los concept-atoms del contexto.
        concepts = [v for k, v in c.context if k == "concept"]
        print(f"  {c.name}: {len(c.members)} atomos, conceptos={concepts[:3]}")

    # === 3. Adjuntar contexto a los clusters ===
    print(f"\n=== PASO 3: set_library_context (lenguaje natural) ===")
    # Buscamos un cluster que contenga concept-atoms relacionados con botellas.
    for c in clusters:
        concepts = [v for k, v in c.context if k == "concept"]
        joined = " ".join(concepts).lower()
        if "plastic" in joined or "botella_plastic" in joined or any(
            "plastic" in v.lower() for v in concepts
        ):
            engine.set_library_context(
                c.name,
                description="Botella de plastico: cuerpo opaco, tapa ancha coloreada.",
                context={"material": "plastic", "shape": "wide_body"},
            )
        elif "glass" in joined or any("glass" in v.lower() for v in concepts):
            engine.set_library_context(
                c.name,
                description="Botella de vidrio: cuerpo transparente, cuello estrecho, tapa pequena.",
                context={"material": "glass", "shape": "narrow_neck"},
            )
        elif "can" in joined or any("can" in v.lower() for v in concepts):
            engine.set_library_context(
                c.name,
                description="Lata de refresco: cuerpo cilindrico corto y ancho, con lengüeta.",
                context={"material": "metal", "shape": "short_cylinder"},
            )
        elif "face" in joined or any("face" in v.lower() for v in concepts):
            engine.set_library_context(
                c.name,
                description="Cara humana: rasgos faciales, parte frontal de la cabeza.",
                context={"body_part": "head_front"},
            )
        elif "hand" in joined or any("hand" in v.lower() for v in concepts):
            engine.set_library_context(
                c.name,
                description="Mano: extremidad con cinco dedos.",
                context={"body_part": "upper_limb_end"},
            )
        elif "body" in joined or any("body" in v.lower() for v in concepts):
            engine.set_library_context(
                c.name,
                description="Cuerpo humano: torso completo que incluye cara y extremidades.",
                context={"body_part": "torso_full"},
            )

    # === 4. Salida como grafo modificable ===
    print(f"\n=== PASO 4: SALIDA COMO GRAFO (modificable por el usuario) ===")
    graph = engine.to_graph()
    # Filtramos las bibliotecas predeterminadas para que el output sea
    # mas legible (las predeterminadas no cambian en este demo).
    user_nodes = [n for n in graph.nodes() if not (
        n.name in ("LETTERS", "LETTERS_UPPER", "LETTERS_LOWER",
                   "VOWELS_UPPER", "VOWELS_LOWER", "CONSONANTS_UPPER", "DIGITS")
    )]
    print(f"Nodos de usuario: {len(user_nodes)}")
    for n in user_nodes:
        desc = f' "{n.description}"' if n.description else ""
        print(f"  [{n.name}]({n.node_kind}){desc}")

    print(f"\nAristas:")
    for e in graph.edges():
        if e.source.startswith("CLUSTER_") or e.target.startswith("CLUSTER_"):
            print(f"  {e.source} --[{e.kind}]--> {e.target}  w={e.weight:.2f}  {e.label}")

    # === 5. Modificar el grafo: el usuario disuelve un cluster mal formado ===
    if len(clusters) >= 5:
        bad_cluster = clusters[0].name
        print(f"\n=== PASO 5: el usuario disuelve {bad_cluster} (agrupamiento mal) ===")
        # Simulamos la decision humana: este cluster no es coherente.
        engine.dissolve_cluster(bad_cluster)
        # Tras dissolve_cluster, el motor re-clustera los concept-atoms.
        print("Re-clustering automatico completado.")
        print("Estado actual de clusters:")
        for name, lib in engine.libraries.items():
            if name.startswith("CLUSTER_"):
                concepts = [v for k, v in lib.context if k == "concept"]
                print(f"  {name}: {len(lib.members)} atomos, conceptos={concepts[:3]}")

    # === 6. classify: el motor "dice" que es un nuevo payload ===
    print(f"\n=== PASO 6: classify (motor 'habla' en lenguaje natural) ===")
    test_payload = payload_for("bottle_glass_green.png")
    cluster, desc = engine.classify(test_payload)
    print(f"Mostrando al motor: {Path(test_payload['image']).name}")
    print(f"  -> {cluster.name if cluster else 'sin clasificar'}: {desc}")

    test_payload2 = payload_for("bottle_plastic_red.png")
    cluster2, desc2 = engine.classify(test_payload2)
    print(f"Mostrando al motor: {Path(test_payload2['image']).name}")
    print(f"  -> {cluster2.name if cluster2 else 'sin clasificar'}: {desc2}")

    test_payload3 = payload_for("face1.png")
    cluster3, desc3 = engine.classify(test_payload3)
    print(f"Mostrando al motor: {Path(test_payload3['image']).name}")
    print(f"  -> {cluster3.name if cluster3 else 'sin clasificar'}: {desc3}")

    # === 7. Exportar el grafo en formato DOT (Graphviz) ===
    print(f"\n=== PASO 7: exportar a DOT (Graphviz) ===")
    dot = graph.to_dot()
    # Guardamos el .dot a disco
    out_path = _ROOT / "training" / "graph_output.dot"
    out_path.write_text(dot, encoding="utf-8")
    print(f"Grafo DOT guardado en: {out_path}")
    print(f"Lineas: {len(dot.splitlines())}")

    # === 8. Resumen final ===
    print(f"\n=== RESUMEN FINAL ===")
    stats = engine.stats()
    for k, v in stats.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
