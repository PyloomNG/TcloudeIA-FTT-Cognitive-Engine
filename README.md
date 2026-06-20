# TcloudeIA — FTT Cognitive Engine

[Status](https://img.shields.io/badge/status-research%20%2F%20pre--alpha-orange) ![Python](https://img.shields.io/badge/python-3.10%2B-blue) ![License](https://img.shields.io/badge/license-MIT-green)

> A domain-agnostic, Hebbian learning engine that uses **set-theory operations** (intersection, union, complement, Jaccard similarity) as its primary inference primitive — and produces an **editable graph of measurable sets** as output, not dense weight matrices.

This repository contains a from-scratch implementation of a small **cognitive architecture** that learns from multimodal input (text, images, integers, and joint multimodal observations) using principles inspired by Hebbian learning, Fodorian language of thought, Tee & Taylor quantization, and set-theoretic computation.

The engine ingests examples one at a time, stores **discrete co-occurrences** in a quantized memory, and answers questions about them using **set operations** rather than matrix multiplication.

---

## 1. Why this project exists

Most modern AI systems are built on dense vector spaces and gradient descent. The model in this project explores an alternative that has very different properties:

- **No floating-point weights.** All association strengths are integers in `[-8, +7]` (4-bit signed).
- **No backpropagation.** Learning is one-shot: every observation moves a single measure up or down by 1.
- **No opaque parameters.** The trained model is a *graph of named sets*. The user can read it, edit it, and feed it back.
- **Domain-agnostic by construction.** The same engine ingests text, images, integers, and joint multimodal observations.
- **Set theory is the API.** `∪`, `∩`, `¬`, `∈`, and Jaccard similarity are first-class operations on the learned state.

The name **FTT** stands for *Fodor–Tee–Taylor*, the three theoretical pillars the engine is built on (see Section 2).

---

## 2. Five architectural pillars

### Pillar 1 — Fodor: combinatorially structured, portable atoms
Mental representations are combinatorial. The smallest indivisible unit in the system is the **`Atom`**, an opaque identity that can be combined into sets, queried, and reasoned about. Atoms are identified by `(domain, local_id)` pairs so that, e.g., the bigram `"er"` in text cannot collide with the edge code `"er"` in an image. Domains are `TEXT`, `IMAGE`, `NUMBER`, and `MULTIMODAL`.

### Pillar 2 — Tee & Taylor: 4-bit quantized storage
All long-term state is stored as integers in the closed range `[-8, +7]`. Saturation is hard: any value that would fall outside the range is clipped to the corresponding extreme. This is a discrete, deterministic, fully reversible form of "memory compression" that prevents the model from accumulating unbounded noise.

### Pillar 3 — Mathematical engine as sets (libraries, not matrices)
Knowledge is represented as a collection of named **`Library`** objects (measurable sets) and the relations between them. Hierarchies are *declarative*: a child library names its parents. Multi-membership is explicit: an atom can belong to many libraries, and the engine can ask "what libraries contain this atom?" with full transitive closure. Inference is set operations on these libraries (`∪`, `∩`, `¬`, `\`).

### Pillar 4 — Hebb: one-shot co-occurrence learning
"Neurons that fire together wire together." Every time a set of atoms is observed together, the engine increments the measure of that *exact set* by 1. Conversely, when the user signals that two atoms should not be together, the measure is decremented. There are no gradients, no optimizers, no epochs.

### Pillar 5 — Metacognition: the engine inspects its own memory
After training, the engine can scan its own quantized memory and discover higher-level patterns. For example: if two different atoms always co-occur with the same *other* atoms, the engine infers that they are **interchangeable** in that context, and stores this as a `Rule` (binary or n-ary). This is a closed-loop system: learning produces memory, memory produces rules, rules constrain future learning and prediction.

---

## 3. Repository layout

```
TcloudeIACore/
├── README.md                  # this file
├── a.py                       # one-line launcher
├── engine/                    # the cognitive engine (a Python package)
│   ├── __init__.py            # public re-exports
│   ├── atom.py                # Atom, AtomId, AtomDomain
│   ├── registry.py            # bijective surface ↔ AtomId map (per domain)
│   ├── quantization.py        # 4-bit signed saturating quantizer
│   ├── association.py         # discrete co-occurrence memory
│   ├── library.py             # immutable measurable set + set operations
│   ├── defaults.py            # pre-declared LETTERS / VOWELS / DIGITS libraries
│   ├── extractor.py           # domain-specific feature extractors
│   ├── role.py                # HEAD / MODIFIER / CONTEXT role assignment
│   ├── trainer.py             # one-shot Hebbian updater
│   ├── predictor.py           # activation-by-superset-set-membership
│   ├── rules.py               # Rule, InferredRule, NaryInferredRule, inferrer
│   ├── graph.py               # editable CognitiveGraph (TEXT / DOT / JSON)
│   └── engine.py              # FTTCognitiveEngine facade
└── training/                  # end-to-end demos
    ├── run_text.py            # text bigram learning
    ├── run_image.py           # image feature learning
    ├── run_numbers.py         # integer factorization learning
    ├── run_multimodal.py      # text + image joint observation
    ├── run_hierarchy.py       # declarative library hierarchies
    ├── run_auto_cluster.py    # unsupervised clustering + human-in-the-loop
    ├── run_digits_arith.py    # classify handwritten digits, then do arithmetic
    ├── make_bottle_samples.py
    ├── make_digit_samples.py
    └── samples/               # generated sample images used by the demos
```

---

## 4. Module-by-module walkthrough

### 4.1 `atom.py` — the minimal unit
Defines an immutable `Atom` (surface form + domain) and an `AtomId` (`(domain, local_id)`). The two are kept strictly separated: `Atom` is the human-readable face (`"a"`, `"er"`, `"W:perro"`), `AtomId` is the internal identity used as a dictionary key. This split is what makes the engine domain-agnostic without losing identity.

### 4.2 `registry.py` — bijective surface-to-id map
`AtomRegistry` provides a *bijection* between surface forms and `AtomId`s **within each domain**. A separate counter per domain guarantees that the string `"er"` registered in `TEXT` and the edge code `"er"` registered in `IMAGE` get distinct IDs. Registration is idempotent: registering the same surface twice returns the same ID.

### 4.3 `quantization.py` — 4-bit signed clipping
A two-method class with a single responsibility: clamp any integer into `[-8, +7]`. It is pure (no state) and reusable. The trainer writes through it on every reinforcement; the predictor reads its output to ensure no measurement ever escapes the 4-bit range.

### 4.4 `association.py` — discrete co-occurrence memory
The memory is a `dict[frozenset[AtomId], int]`. Every key is an *unordered* set of atoms (commutative by construction), every value is a quantized signed measure. The memory exposes three structural queries that replace dense linear algebra:
- `strong_associations(threshold)` — return all sets with measure above a threshold;
- `supersets_of(atoms, threshold)` — return all stored sets that contain a given set;
- `measures_in_range(min, max)` — return all stored sets with measure in a range.

The `supersets_of` query is the heart of the predictor: rather than dot products, prediction is "what bigger sets contain this one?".

### 4.5 `library.py` — measurable sets + set operations
A `Library` is an immutable, hashable, named set of atoms with a measure and a tuple of parent names. Operations:
- `intersects`, `union`, `complement`, `difference` — return a *new* library;
- `add`, `remove`, `merge` — return a *new* library;
- `transitive_parents(library_index)` — DFS over the parent chain with cycle protection;
- `contains(atom)` — direct membership.

All operations are functional: they do not mutate the receiver. This makes the learned state fully snapshotable and thread-safe.

### 4.6 `defaults.py` — innate knowledge
Pre-declares a small hierarchy of libraries every fresh engine ships with:
```
LETTERS
├── LETTERS_UPPER
│   ├── VOWELS_UPPER
│   └── CONSONANTS_UPPER
└── LETTERS_LOWER
    └── VOWELS_LOWER
DIGITS
```
This is the engine's "innate" alphabet and numeral awareness. The user can extend it, replace it, or delete it.

### 4.7 `extractor.py` — domain-specific atom extractors
A `Strategy` base class with concrete extractors:
- **`BigramExtractor`** (TEXT) — splits a word into overlapping bigrams (with a `#U#`-prefixed unigram fallback for single-letter words).
- **`ImageEdgeExtractor`** (IMAGE) — quantizes a 2D grid of pixels to 4 bits, produces 2×2 patch atoms, counts horizontal/vertical edge transitions, and binarizes them into `LOW / MID / HIGH` atoms.
- **`RealImageExtractor`** (IMAGE) — for real photographs (PNG/JPG via PIL). Computes a perceptual hash (aHash) over the whole image, plus three regional hashes (`top`/`mid`/`bot`), a luminance histogram, an edge density, and an aspect-ratio atom. This is the extractor used to cluster real photos.
- **`NumberFactorExtractor`** (NUMBER) — for integers. Produces atoms for the sign (`POS`/`NEG`/`ZERO`), magnitude bucket (`MAG:SMALL`/`MAG:MEDIUM`/`MAG:LARGE`), prime factors with multiplicity (`PRIME:2`, `PRIME:2`, `PRIME:3` for 12), and digit bigrams.
- **`MultimodalFusionExtractor`** (MULTIMODAL) — composes sub-extractors per payload key (or per Python type), accumulates all atoms, and additionally registers a deterministic `CO_OCC:<sha1_8>` anchor atom that represents the joint observation.

### 4.8 `role.py` — generalized role assignment
A v1-style system used a fixed `(subject, predicate)` pair, which was linguistic. v2 generalizes: every domain's `RoleAssignment` decides which atoms are `HEAD` (the core), `MODIFIER` (companion), or `CONTEXT` (peripheral). Implementations:
- `PositionalRoleAssignment` (TEXT) — first half of atoms are HEAD, rest MODIFIER.
- `ImageSpatialRoleAssignment` (IMAGE) — same positional split, representing top/bottom halves.
- `NumberRoleAssignment` (NUMBER) — uses the surface prefix: `MAG:*` is HEAD, `PRIME:*` MODIFIER, `DIG:*` CONTEXT.

The HEAD ∪ MODIFIER set is what the trainer reinforces in one shot. CONTEXT atoms are observed but not reinforced.

### 4.9 `trainer.py` — one-shot Hebbian updater
The `GeneralizedHebbianTrainer` implements **two phases** per observation:
1. **Hebbian reinforcement.** The set `HEAD ∪ MODIFIER` has its measure incremented by 1, then quantized to 4 bits.
2. **Selective anti-Hebbian inhibition.** For every atom in the same domain that previously had a *positive* association with the current set, the engine decrements the measure of `current ∪ {atom}` by 1.

This is the v2 equivalent of competitive learning: new patterns are carved against previously co-active ones, not against the entire universe. A `structural_atoms` helper also returns the per-payload structural fingerprint used by the auto-clusterer (the set of atoms *without* the unique concept atom).

### 4.10 `predictor.py` — activation by superset lookup
The `GeneralizedStructuralPredictor` answers "which library is this payload?" in three phases:
1. **Direct activation.** For each atom in the input and each known atom `p`, sum the measures of all stored associations that contain the *set* `{h, p}`. This is the v2 analogue of `memory.get(subject, predicate)`.
2. **Rule-driven inheritance.** If an inferred rule links an input atom to a partner not in the input, also sum the partner's activations.
3. **Library ranking.** For each declared library, sum the activations of the atoms it contains, quantize, and sort by measure descending.

The prediction output is a list of `PRED(<library_name>)` libraries ordered by predicted measure, not a single hard label.

### 4.11 `rules.py` — metacognition
A `Rule` connects a tuple of atoms with a `shared_context` (the atoms that all of them co-activate). Two concrete subtypes:
- `InferredRule` — binary, with a canonical atom order.
- `NaryInferredRule` — n-ary, for higher-order abstractions.

The `CoActivationRuleInference` strategy implements the v2 rule inference: for every pair of atoms that share ≥ K co-active associations above an activation threshold, emit a rule. By construction it never crosses domains. Rules are stored in a `RuleStore` keyed by name.

### 4.12 `graph.py` — the editable output
After training, the engine exposes its state as a **`CognitiveGraph`**: a set of `GraphNode`s (libraries, clusters) and `GraphEdge`s (subset, similarity, inferred). The graph supports:
- `add_node`, `remove_node`, `rename_node`, `set_label`;
- `add_edge`, `remove_edge`;
- `move_atom`, `merge_nodes`;
- `to_text()` for human reading;
- `to_dot()` for Graphviz;
- `to_json()` for programmatic use.

The user can edit the graph, then call `engine.apply_graph(graph)` to push the edits back into the engine. This is what makes the system **interactive and human-in-the-loop** in a way that dense neural weights cannot be.

### 4.13 `engine.py` — the public facade
`FTTCognitiveEngine` is the single import users need. It wires the registry, memory, trainer, predictor, rule store, rule strategy, and library index into a coherent API:

- **Construction.** `FTTCognitiveEngine(trainer, predictor, rule_strategy, rule_store, registry, memory, ...)` — with optional default libraries installed automatically.
- **Library management.** `declare_library`, `add_membership`, `remove_membership`, `rename_library`.
- **Observation.** `observe(text=..., image=..., number=...)` — domain-agnostic, supports multiple modalities in one call.
- **Set queries.** `is_in(atom, library)`, `libraries_of(atom)`, `intersect(a, b)`, `union(a, b)`.
- **Prediction.** `predict_libraries(**inputs)`, `cluster_for(payload)`, `classify(payload)`, `explain(payload)`.
- **Metacognition.** `infer_rules()` — runs the rule strategy and stores new rules.
- **Unsupervised learning.** `auto_cluster(payloads, threshold, discriminative_only)` — clusters observations by Jaccard similarity of structural atoms, creates `CLUSTER_n` libraries, and (optionally) prunes non-discriminative atoms.
- **Human-in-the-loop.** `confirm_cluster`, `move_atom`, `dissolve_cluster`, `differentiate(a, b)`.
- **Arithmetic over classified payloads.** `operate('+', a, b)` — classifies each payload into a cluster, reads a user-set `value` from the cluster's context, and applies the operator.
- **Graph round-trip.** `to_graph()`, `apply_graph(graph)`, `graph()`.
- **Inspection.** `print_state()`, `stats()`.

`stats()` is the cheap diagnostic every demo prints at the end: it returns total atom counts per domain, total associations, total libraries, total rules, and a boolean that asserts every measure is in `[-8, +7]`.

---

## 5. End-to-end demos (in `training/`)

The `training/` directory contains nine self-contained scripts. Each one imports the engine, runs a small experiment, and prints the result.

| Script | What it demonstrates |
| --- | --- |
| `run_text.py` | Train on a short text corpus; ask whether a word is in a learned library. |
| `run_numbers.py` | Train on integers; demonstrate prime-factor + magnitude-based reasoning. |
| `run_image.py` | Train on small image grids; visualize learned atoms. |
| `run_multimodal.py` | Train on `text + image` pairs; show that joint atoms emerge. |
| `run_hierarchy.py` | Declare `PERRO ⊂ MAMIFERO ⊂ ANIMAL`; show transitive `is_in`. |
| `run_auto_cluster.py` | Cluster 12 images of bottles / faces / cans unsupervised; the user then `confirm_cluster`, `move_atom`, `dissolve_cluster` to refine. |
| `run_digits_arith.py` | 20 digit images (10 digits × 2 fonts) → 10 clusters → user assigns each cluster a numeric `value` → `operate('+', a, b)`, `operate('*', a, b)`, `operate('sum', a, b, c)` all work end-to-end. |
| `make_bottle_samples.py` | Generates the `samples/bottle_*.png` training images. |
| `make_digit_samples.py` | Generates the `samples/digit_*.png` training images. |

A canonical fingerprint of the project is the digits demo:
1. 20 PNG digits are observed.
2. The engine produces 10 clusters by Jaccard similarity of structural atoms.
3. Each cluster is renamed `DIGIT_n` and given a numeric value.
4. The engine then answers arithmetic questions (`1 + 1 = 2`, `3 * 4 = 12`, `sum(5, 7, 2) = 14`) by *classifying* each image and reading the value from the cluster's context. Arithmetic itself is done in plain Python; the engine's job is classification.

---

## 6. Key design choices

- **Immutability everywhere it matters.** `Atom`, `AtomId`, and `Library` are frozen dataclasses. The only mutable structures are the memory, the registry, and the engine's library index — exactly the structures that have to change to learn.
- **SOLID throughout.** Each module has one responsibility; the trainer, predictor, and rules strategy are all substitutable; new domains are added by subclassing `AtomExtractor` and `RoleAssignment` without touching the engine.
- **Idempotent registration.** Registering the same surface form twice returns the same ID. This is what makes the engine robust to repeated observations.
- **No floating point in the stored state.** All long-term state is integer in `[-8, +7]`. Floating point only appears in transient computations (e.g., Jaccard similarity threshold comparisons, luminance ratios during extraction).
- **Set theory is the API surface.** Users do not call `predict(weights)`, they call `intersect(a, b)`, `union(a, b)`, `differentiate(a, b)`, `is_in(atom, library)`. Inference is set operation, not matrix multiplication.
- **Editable output.** The learned model is a graph the user can rename, edit, and re-apply. The engine is meant to be a *collaborator*, not a black box.
- **Domain-agnostic core.** The same `GeneralizedHebbianTrainer`, `GeneralizedStructuralPredictor`, `AssociationMemory`, and `RuleStore` serve text, images, integers, and multimodal payloads. Only the extractor and the role assignment change per domain.

---

## 7. Running the demos

```bash
# from the repository root
python training/run_text.py
python training/run_numbers.py
python training/run_image.py
python training/run_multimodal.py
python training/run_hierarchy.py
python training/run_auto_cluster.py
python training/run_digits_arith.py
```

Each script is self-contained and prints a final `stats()` block you can use as a regression check.

---

## 8. Conceptual limits and what this engine is *not*

This is a research artifact for studying **discrete, set-theoretic learning**. It is not a replacement for deep learning; it is a deliberate counter-example that:

- does not scale to billions of parameters;
- does not produce probability distributions;
- does not implement backpropagation;
- does not train on GPUs.

What it *does* offer is a small, inspectable, fully editable model whose every piece of state is named, finite, and machine-readable. That property — **interpretability and editability as a first-class output, not a post-hoc analysis** — is the central contribution of the project.
