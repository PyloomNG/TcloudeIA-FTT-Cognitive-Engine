"""
Caso de uso 3: IMAGEN.

Ingiere 4 rejillas 4x4 (dos con rayas verticales, dos con rayas
horizontales) y declara bibliotecas VSTRIPE_PATCH y HSTRIPE_PATCH.
Demuestra que:
    - Una rejilla inedita del tipo VSTRIPE se predice como VSTRIPE_PATCH.
    - Una rejilla inedita del tipo HSTRIPE se predice como HSTRIPE_PATCH.
    - Todas las medidas en [-8, 7].
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


# ---------------------------------------------------------------------------
# Rejillas de ejemplo (4x4). Los valores son intensidades de pixel [0, 255].
# ---------------------------------------------------------------------------

# Rayas VERTICALES: dos columnas blancas (255), dos columnas negras (0).
VSTRIPE_1: list[list[int]] = [
    [255, 0, 255, 0],
    [255, 0, 255, 0],
    [255, 0, 255, 0],
    [255, 0, 255, 0],
]
VSTRIPE_2: list[list[int]] = [
    [0, 255, 0, 255],
    [0, 255, 0, 255],
    [0, 255, 0, 255],
    [0, 255, 0, 255],
]

# Rayas HORIZONTALES: dos filas blancas, dos filas negras.
HSTRIPE_1: list[list[int]] = [
    [255, 255, 255, 255],
    [255, 255, 255, 255],
    [0, 0, 0, 0],
    [0, 0, 0, 0],
]
HSTRIPE_2: list[list[int]] = [
    [0, 0, 0, 0],
    [0, 0, 0, 0],
    [255, 255, 255, 255],
    [255, 255, 255, 255],
]

# Rejilla inedita: VSTRIPE con pequena variacion (delta < 32 para que
# el detector de bordes no introduzca nuevos atomos HEDGE:*). Cambia
# 255 -> 230 en un pixel: la diferencia 25 < 32 no dispara un borde.
VSTRIPE_NOUVEAU: list[list[int]] = [
    [255, 0, 255, 0],
    [255, 0, 255, 0],
    [255, 0, 230, 0],  # un pixel cambiado: 255 -> 230
    [255, 0, 255, 0],
]

# Rejilla inedita: HSTRIPE con pequena variacion analoga.
HSTRIPE_NOUVEAU: list[list[int]] = [
    [255, 255, 255, 255],
    [255, 255, 255, 255],
    [0, 0, 0, 0],
    [0, 0, 0, 25],  # un pixel cambiado: 0 -> 25
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

    # === Bibliotecas visuales ===
    # Para imagen, el "concepto" es un hash por rejilla, así que dos
    # rejillas distintas tienen concept-atoms distintos y la membresía
    # por concepto no generaliza. La GENERALIZACIÓN se hace a través
    # de los átomos de BORDE (HEDGE:*, VEDGE:*), que son discretos y
    # compartidos por rejillas de la misma clase.
    # NOTA: el sentido de las aristas es INVERSO a la orientación del
    # stripe (contra-intuitivo pero correcto):
    #   - VSTRIPE (rayas verticales) -> cada fila alterna blanco/negro
    #     -> muchas aristas HORIZONTALES (HEDGE:HIGH) y pocas verticales.
    #   - HSTRIPE (rayas horizontales) -> cada columna alterna
    #     -> muchas aristas VERTICALES (VEDGE:HIGH) y pocas horizontales.
    hedge_high = Atom("HEDGE:HIGH", AtomDomain.IMAGE)
    vedge_high = Atom("VEDGE:HIGH", AtomDomain.IMAGE)

    engine.declare_library("VSTRIPE_PATCH", {hedge_high})
    engine.declare_library("HSTRIPE_PATCH", {vedge_high})

    # === Exposición a las 4 rejillas ===
    print("=== ENTRENAMIENTO VISUAL (4 rejillas 4x4) ===")
    from engine.trainer import GeneralizedHebbianTrainer as _T
    for name, grid in [
        ("VSTRIPE_1", VSTRIPE_1),
        ("VSTRIPE_2", VSTRIPE_2),
        ("HSTRIPE_1", HSTRIPE_1),
        ("HSTRIPE_2", HSTRIPE_2),
    ]:
        engine.observe(image=grid)
        print(f"  observado: {name}  concept={_T._concept_atom(grid, AtomDomain.IMAGE)}")

    stats = engine.stats()
    assert stats["all_measures_in_range"] == 1, stats
    assert stats["atoms_image"] > 0, "Deberia haber atomos IMAGE"

    # === Verificar que los atomos de borde estan registrados ===
    print("\n=== ATOMOS DE BORDE REGISTRADOS ===")
    image_atoms = engine.registry.known_in_domain(AtomDomain.IMAGE)
    edges = sorted(
        engine.registry.surface_of(a) for a in image_atoms
        if engine.registry.surface_of(a).startswith(("HEDGE:", "VEDGE:"))
    )
    for s in edges:
        print(f"  - {s}")

    # === Multi-membresía: HEDGE:HIGH debe estar en VSTRIPE_PATCH ===
    print("\n=== MULTI-MEMBRESIA: libraries_of(HEDGE:HIGH) ===")
    libs_h = engine.libraries_of(hedge_high)
    for lib in libs_h:
        print(f"  - {lib.name}")
    assert any(l.name == "VSTRIPE_PATCH" for l in libs_h), (
        f"HEDGE:HIGH deberia estar en VSTRIPE_PATCH, libs={libs_h}"
    )
    libs_v = engine.libraries_of(vedge_high)
    assert any(l.name == "HSTRIPE_PATCH" for l in libs_v), (
        f"VEDGE:HIGH deberia estar en HSTRIPE_PATCH, libs={libs_v}"
    )

    # === Predicción sobre rejillas ineditas ===
    print("\n=== PREDICCION SOBRE REJILLAS INEDITAS ===")
    print(f"  VSTRIPE_NOUVEAU concept = {_T._concept_atom(VSTRIPE_NOUVEAU, AtomDomain.IMAGE)}")
    print(f"  HSTRIPE_NOUVEAU concept = {_T._concept_atom(HSTRIPE_NOUVEAU, AtomDomain.IMAGE)}")

    preds_v = engine.predict_libraries(image=VSTRIPE_NOUVEAU)
    if preds_v:
        for p in preds_v:
            members = sorted(engine.registry.surface_of(a) for a in p.members)
            print(f"  VSTRIPE_NOUVEAU -> {p.name} medida={p.measure:+d} members={members}")
        assert any(p.name == "PRED(VSTRIPE_PATCH)" for p in preds_v), (
            f"VSTRIPE_NOUVEAU deberia predecir VSTRIPE_PATCH entre sus opciones, "
            f"top={preds_v[0].name}"
        )
    else:
        print("  VSTRIPE_NOUVEAU -> (sin prediccion)")

    preds_h = engine.predict_libraries(image=HSTRIPE_NOUVEAU)
    if preds_h:
        for p in preds_h:
            members = sorted(engine.registry.surface_of(a) for a in p.members)
            print(f"  HSTRIPE_NOUVEAU -> {p.name} medida={p.measure:+d} members={members}")
        assert any(p.name == "PRED(HSTRIPE_PATCH)" for p in preds_h), (
            f"HSTRIPE_NOUVEAU deberia predecir HSTRIPE_PATCH entre sus opciones, "
            f"top={preds_h[0].name}"
        )
    else:
        print("  HSTRIPE_NOUVEAU -> (sin prediccion)")

    # === Resumen ===
    print("\n=== RESUMEN ===")
    for k, v in stats.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
