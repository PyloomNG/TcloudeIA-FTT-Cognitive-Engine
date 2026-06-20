"""
Motor FTT (Fodor-TeeTaylor) v2 — paquete de inteligencia artificial cognitiva.

API pública: re-exports planos para que el cliente use:
    from engine import FTTCognitiveEngine, Atom, Library, ...
sin necesidad de conocer la estructura interna de submódulos.

v2 es agnóstico al dominio: el mismo motor ingiere texto, imágenes y
números. Las decisiones arquitectónicas clave están documentadas en cada
submódulo; este archivo solo expone la superficie pública.
"""

from .association import AssociationMemory
from .atom import Atom, AtomDomain, AtomId
from .engine import FTTCognitiveEngine
from .extractor import (
    AtomExtractor,
    ImageEdgeExtractor,
    MultimodalFusionExtractor,
    NumberFactorExtractor,
    RealImageExtractor,
)
from .graph import CognitiveGraph, GraphEdge, GraphNode
# BigramExtractor existe en v2 (subclase de AtomExtractor) pero no se
# reexporta en __all__ para no colisionar con el nombre v1. Los clientes
# pueden accederlo via `from engine.extractor import BigramExtractor`.
from .library import Library
from .predictor import GeneralizedStructuralPredictor
from .quantization import Quantizer
from .registry import AtomRegistry
from .role import (
    ImageSpatialRoleAssignment,
    NumberRoleAssignment,
    PositionalRoleAssignment,
    Role,
    RoleAssignment,
)
from .rules import (
    CoActivationRuleInference,
    InferredRule,
    NaryInferredRule,
    Rule,
    RuleInferenceStrategy,
    RuleStore,
)
from .trainer import (
    GeneralizedHebbianTrainer,
    default_extractors,
    default_role_assignments,
)

__all__ = [
    # --- Cuantización (Pilar 2) ---
    "Quantizer",
    # --- Átomos (Pilar 1) ---
    "AtomDomain",
    "AtomId",
    "Atom",
    "AtomRegistry",
    # --- Memoria de asociaciones (Pilar 2 + 4) ---
    "AssociationMemory",
    # --- Bibliotecas medibles (Pilar 3) ---
    "Library",
    # --- Extracción de átomos ---
    "AtomExtractor",
    "ImageEdgeExtractor",
    "RealImageExtractor",
    "NumberFactorExtractor",
    "MultimodalFusionExtractor",
    # --- Asignación de roles ---
    "Role",
    "RoleAssignment",
    "PositionalRoleAssignment",
    "ImageSpatialRoleAssignment",
    "NumberRoleAssignment",
    # --- Reglas / Metacognición (Pilar 5) ---
    "Rule",
    "InferredRule",
    "NaryInferredRule",
    "RuleStore",
    "RuleInferenceStrategy",
    "CoActivationRuleInference",
    # --- Entrenador y predictor generalizados ---
    "GeneralizedHebbianTrainer",
    "GeneralizedStructuralPredictor",
    "default_extractors",
    "default_role_assignments",
    # --- Grafo de conjuntos modificable (Pilar 3 como salida editable) ---
    "CognitiveGraph",
    "GraphNode",
    "GraphEdge",
    # --- Fachada ---
    "FTTCognitiveEngine",
]
