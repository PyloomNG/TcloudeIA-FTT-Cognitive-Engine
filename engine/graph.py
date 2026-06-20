"""
Grafo de Conjuntos Medibles (salida modificable del entrenamiento).

El entrenamiento produce GRAPH, no matrices densas. Este módulo define
una estructura de datos GRAFO que el usuario puede editar:
    - add_node / remove_node
    - rename_node
    - add_edge / remove_edge (jerarquía, similitud, inferencia)
    - set_label (descripción en lenguaje natural)
    - move_atom (mover un átomo entre nodos)
    - Exports: to_text(), to_dot(), to_json()

Los nodos llevan 'payloads_repr': ejemplos que cayeron en ese cluster
(útil para mostrar imágenes en el grafo).

Principio SOLID aplicado:
    - SRP: el grafo SOLO modela la estructura, no consulta memoria.
    - OCP: nuevos tipos de aristas se añaden como string 'kind'.
    - DIP: la fachada traduce entre Library y GraphNode.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Iterable


@dataclass
class GraphNode:
    """Nodo del grafo: una biblioteca / cluster / conjunto."""

    name: str
    members: tuple[str, ...] = ()
    description: str | None = None
    context: dict[str, Any] = field(default_factory=dict)
    node_kind: str = "library"  # "library" | "cluster" | "atom" | "default"
    examples: tuple[str, ...] = ()  # paths o strings de ejemplos


@dataclass
class GraphEdge:
    """Arista del grafo: relación entre dos nodos."""

    source: str
    target: str
    kind: str  # "subset" | "similarity" | "inferred" | "parent"
    weight: float = 1.0
    label: str = ""


class CognitiveGraph:
    """Grafo modificable de bibliotecas, clusters y relaciones.

    Es la salida del entrenamiento en formato de teoría de conjuntos
    (no matrices). El usuario puede modificar el grafo y devolverlo al
    motor con `engine.apply_graph(graph)`.
    """

    def __init__(self) -> None:
        self._nodes: dict[str, GraphNode] = {}
        self._edges: list[GraphEdge] = []

    # --- Fábrica desde el estado de un motor ---

    @classmethod
    def from_engine(cls, engine: Any) -> "CognitiveGraph":
        """Construye el grafo a partir de un FTTCognitiveEngine.

        Itera las bibliotecas declaradas y crea nodos. Las aristas
        vienen de:
            - jerarquía `parents` (subset)
            - similitud Jaccard entre CLUSTER_* (similarity)
            - reglas inferidas que conectan átomos de bibliotecas
              distintas (inferred)
        """
        graph = cls()
        for name, lib in engine.libraries.items():
            members_surf = tuple(sorted(
                engine.registry.surface_of(a) for a in lib.members
            ))
            kind = "cluster" if name.startswith("CLUSTER_") else "library"
            examples = lib.context_dict().get("examples", [])
            if isinstance(examples, str):
                examples = [examples]
            graph._nodes[name] = GraphNode(
                name=name,
                members=members_surf,
                description=lib.description,
                context=lib.context_dict(),
                node_kind=kind,
                examples=tuple(str(e) for e in examples),
            )
        # Aristas: jerarquía
        for name, lib in engine.libraries.items():
            for parent in lib.parents:
                if parent in graph._nodes:
                    graph._edges.append(
                        GraphEdge(
                            source=name, target=parent,
                            kind="parent", weight=1.0, label="⊂",
                        )
                    )
        # Aristas: similitud entre clusters
        cluster_names = [
            n for n, node in graph._nodes.items() if node.node_kind == "cluster"
        ]
        for i, n1 in enumerate(cluster_names):
            for n2 in cluster_names[i + 1:]:
                node1 = graph._nodes[n1]
                node2 = graph._nodes[n2]
                set1 = set(node1.members)
                set2 = set(node2.members)
                sim = CognitiveGraph._jaccard(set1, set2)
                if sim > 0:
                    graph._edges.append(
                        GraphEdge(
                            source=n1, target=n2,
                            kind="similarity", weight=sim,
                            label=f"J={sim:.2f}",
                        )
                    )
        return graph

    # --- Modificación (usuario) ---

    def add_node(
        self,
        name: str,
        *,
        description: str | None = None,
        members: Iterable[str] = (),
        kind: str = "library",
        examples: Iterable[str] = (),
    ) -> GraphNode:
        if name in self._nodes:
            raise ValueError(f"add_node: ya existe {name!r}")
        node = GraphNode(
            name=name,
            members=tuple(sorted(members)),
            description=description,
            context={},
            node_kind=kind,
            examples=tuple(examples),
        )
        self._nodes[name] = node
        return node

    def remove_node(self, name: str) -> None:
        if name not in self._nodes:
            raise KeyError(f"remove_node: desconocido {name!r}")
        del self._nodes[name]
        # También quitamos aristas incidentes.
        self._edges = [
            e for e in self._edges
            if e.source != name and e.target != name
        ]

    def rename_node(self, old_name: str, new_name: str) -> None:
        if old_name not in self._nodes:
            raise KeyError(f"rename_node: desconocido {old_name!r}")
        if new_name in self._nodes:
            raise ValueError(f"rename_node: destino {new_name!r} ya existe")
        node = self._nodes.pop(old_name)
        node.name = new_name
        self._nodes[new_name] = node
        for edge in self._edges:
            if edge.source == old_name:
                edge.source = new_name
            if edge.target == old_name:
                edge.target = new_name

    def set_label(self, name: str, description: str | None) -> None:
        if name not in self._nodes:
            raise KeyError(f"set_label: desconocido {name!r}")
        self._nodes[name].description = description

    def add_edge(
        self,
        source: str,
        target: str,
        kind: str = "parent",
        weight: float = 1.0,
        label: str = "",
    ) -> GraphEdge:
        if source not in self._nodes:
            raise KeyError(f"add_edge: source {source!r} desconocido")
        if target not in self._nodes:
            raise KeyError(f"add_edge: target {target!r} desconocido")
        edge = GraphEdge(
            source=source, target=target,
            kind=kind, weight=weight, label=label,
        )
        self._edges.append(edge)
        return edge

    def remove_edge(self, source: str, target: str, kind: str | None = None) -> int:
        """Quita aristas entre source y target. Si kind es None, quita todas."""
        original = len(self._edges)
        self._edges = [
            e for e in self._edges
            if not (
                (e.source == source and e.target == target)
                or (e.source == target and e.target == source)
            ) or (kind is not None and e.kind != kind)
        ]
        return original - len(self._edges)

    def move_atom(self, atom: str, from_node: str, to_node: str) -> None:
        """Mueve un átomo (string) de un nodo a otro."""
        if from_node not in self._nodes:
            raise KeyError(f"move_atom: {from_node!r} desconocido")
        if to_node not in self._nodes:
            raise KeyError(f"move_atom: {to_node!r} desconocido")
        src = self._nodes[from_node]
        dst = self._nodes[to_node]
        if atom not in src.members:
            return
        new_src_members = tuple(m for m in src.members if m != atom)
        new_dst_members = tuple(sorted(set(dst.members) | {atom}))
        self._nodes[from_node] = GraphNode(
            name=src.name,
            members=new_src_members,
            description=src.description,
            context=src.context,
            node_kind=src.node_kind,
            examples=src.examples,
        )
        self._nodes[to_node] = GraphNode(
            name=dst.name,
            members=new_dst_members,
            description=dst.description,
            context=dst.context,
            node_kind=dst.node_kind,
            examples=dst.examples,
        )

    def merge_nodes(self, target: str, sources: Iterable[str]) -> None:
        """Fusiona varios nodos en `target`."""
        for s in sources:
            if s == target:
                continue
            if s not in self._nodes:
                continue
            src = self._nodes[s]
            tgt = self._nodes[target]
            new_members = tuple(sorted(set(tgt.members) | set(src.members)))
            self._nodes[target] = GraphNode(
                name=target,
                members=new_members,
                description=tgt.description or src.description,
                context={**tgt.context, **src.context},
                node_kind=tgt.node_kind,
                examples=tgt.examples + src.examples,
            )
            self.remove_node(s)

    # --- Inspección ---

    def nodes(self) -> list[GraphNode]:
        return list(self._nodes.values())

    def edges(self) -> list[GraphEdge]:
        return list(self._edges)

    def get_node(self, name: str) -> GraphNode | None:
        return self._nodes.get(name)

    # --- Exports ---

    def to_text(self) -> str:
        lines: list[str] = []
        lines.append("=== GRAFO DE CONJUNTOS MEDIBLES (modificable) ===")
        if not self._nodes:
            lines.append("  (vacío)")
            return "\n".join(lines)
        # Nodos
        lines.append("\n[ NODOS ]")
        for node in sorted(self._nodes.values(), key=lambda n: n.name):
            desc = f' "{node.description}"' if node.description else ""
            members_preview = ", ".join(node.members[:5])
            more = f" (+{len(node.members) - 5} más)" if len(node.members) > 5 else ""
            examples = ""
            if node.examples:
                ex_preview = ", ".join(node.examples[:3])
                examples = f"  ej={ex_preview}"
            lines.append(
                f"  [{node.name}]({node.node_kind}){desc}  "
                f"|C|={len(node.members)}  {{{members_preview}{more}}}{examples}"
            )
        # Aristas
        lines.append("\n[ ARISTAS ]")
        if not self._edges:
            lines.append("  (ninguna)")
        else:
            for e in sorted(self._edges, key=lambda x: (x.source, x.target, x.kind)):
                lines.append(
                    f"  {e.source} --[{e.kind}]--> {e.target}  "
                    f"w={e.weight:.2f}  {e.label}"
                )
        return "\n".join(lines)

    def to_dot(self) -> str:
        """Exporta en formato DOT (Graphviz) para visualización."""
        lines = ["digraph CognitiveGraph {"]
        lines.append("  rankdir=LR;")
        lines.append('  node [shape=box, style="rounded,filled", fontname="Helvetica"];')
        for node in self._nodes.values():
            color = {
                "library": "#cce5ff",
                "cluster": "#ffe5cc",
                "atom": "#e5ffcc",
                "default": "#f0f0f0",
            }.get(node.node_kind, "#ffffff")
            label = node.name
            if node.description:
                label += f"\\n{node.description[:30]}"
            members_preview = ", ".join(node.members[:3])
            if members_preview:
                label += f"\\n[{members_preview}{'...' if len(node.members) > 3 else ''}]"
            if node.examples:
                label += f"\\n ej: {node.examples[0]}"
            lines.append(
                f'  "{node.name}" [label="{label}", fillcolor="{color}"];'
            )
        for edge in self._edges:
            attrs = f'label="{edge.label}"' if edge.label else ""
            if edge.weight != 1.0:
                attrs += f', penwidth="{max(0.5, edge.weight * 3):.2f}"'
            lines.append(
                f'  "{edge.source}" -> "{edge.target}" [{attrs}];'
            )
        lines.append("}")
        return "\n".join(lines)

    def to_json(self) -> str:
        return json.dumps(
            {
                "nodes": [
                    {
                        "name": n.name,
                        "members": list(n.members),
                        "description": n.description,
                        "context": n.context,
                        "node_kind": n.node_kind,
                        "examples": list(n.examples),
                    }
                    for n in self._nodes.values()
                ],
                "edges": [
                    {
                        "source": e.source,
                        "target": e.target,
                        "kind": e.kind,
                        "weight": e.weight,
                        "label": e.label,
                    }
                    for e in self._edges
                ],
            },
            ensure_ascii=False,
            indent=2,
        )

    @staticmethod
    def _jaccard(a: set[str], b: set[str]) -> float:
        if not a and not b:
            return 0.0
        if not a or not b:
            return 0.0
        return len(a & b) / len(a | b)
