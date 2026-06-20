"""
Extractores de átomos agnósticos al dominio (Pilar 1: Fodor).

Cada extractor define cómo se descompone un "payload" de un dominio
concreto en átomos combinatorios. Los átomos resultantes se registran
en el `AtomRegistry` antes de devolverse, garantizando identidad
bijectiva.

Cuatro implementaciones concretas:
    - BigramExtractor (TEXT):      portado del v1 con fallback de unigrama.
    - ImageEdgeExtractor (IMAGE):  rejillas 2x2 con píxeles cuantizados a
                                   4 bits + histograma de bordes.
    - NumberFactorExtractor (NUM): signo, magnitud, factores primos con
                                   multiplicidad, bigramas de dígitos.
    - MultimodalFusionExtractor (MULTIMODAL): compone sub-extractores
                                   y crea un átomo ancla CO_OCC.

Principio SOLID aplicado:
    - Open/Closed: añadir un nuevo dominio (audio, grafo...) es
      extender este módulo sin tocar AssociationMemory ni Trainer.
    - Liskov Substitution: todo extractor concreto satisface el mismo
      contrato (extract -> set[AtomId]).
    - Single Responsibility: cada extractor sabe de UN dominio.
"""

from __future__ import annotations

import hashlib
from abc import ABC, abstractmethod
from typing import Any, Iterable, Mapping

from .atom import Atom, AtomDomain, AtomId
from .registry import AtomRegistry


# Umbral para considerar un cambio de intensidad como "borde" en una imagen.
_EDGE_THRESHOLD: int = 32


class AtomExtractor(ABC):
    """Estrategia de extracción de átomos a partir de un payload."""

    DOMAIN: AtomDomain  # atributo de clase: lo redefine cada subclase

    @abstractmethod
    def extract(self, payload: Any, registry: AtomRegistry) -> set[AtomId]:
        """Devuelve el conjunto de AtomIds extraídos del payload.

        El extractor ES RESPONSABLE de registrar cada átomo en el registry
        (vía `registry.register(surface, DOMAIN)`). El registry garantiza
        idempotencia: el mismo `surface_form` en el mismo dominio siempre
        produce el mismo AtomId.
        """
        raise NotImplementedError


# ---------------------------------------------------------------------------
# Texto
# ---------------------------------------------------------------------------


class BigramExtractor(AtomExtractor):
    """Bigramas como átomos de texto. Fallback `#U#` para palabras de 1 carácter."""

    DOMAIN: AtomDomain = AtomDomain.TEXT
    _FALLBACK_PREFIX: str = "#U#"

    def extract(self, payload: Any, registry: AtomRegistry) -> set[AtomId]:
        if not isinstance(payload, str):
            return set()
        word = payload.lower()
        atoms: set[AtomId] = set()
        if len(word) >= 2:
            for i in range(len(word) - 1):
                atoms.add(registry.register(word[i : i + 2], self.DOMAIN))
        if len(word) == 1:
            # Palabras de 1 carácter no producen bigramas; usamos un unigrama
            # con prefijo para no colisionar con ningún bigrama real.
            atoms.add(
                registry.register(f"{self._FALLBACK_PREFIX}{word}", self.DOMAIN)
            )
        return atoms


# ---------------------------------------------------------------------------
# Imagen
# ---------------------------------------------------------------------------


class ImageEdgeExtractor(AtomExtractor):
    """Rejilla de píxeles -> átomos visuales.

    Estrategia (implementación funcional, no esqueleto):
        1. Cuantización de cada píxel a 4 bits (`>> 4`, rango 0-15).
        2. Parches 2x2 como átomos "P2x2:qTL_qTR_qBL_qBR".
        3. Bordes horizontales: transiciones de intensidad > 32 entre
           píxeles vecinos de la misma fila. Se binean en LOW/MID/HIGH.
        4. Bordes verticales: análogo por columna.
    """

    DOMAIN: AtomDomain = AtomDomain.IMAGE

    def extract(self, payload: Any, registry: AtomRegistry) -> set[AtomId]:
        grid = self._coerce_grid(payload)
        if grid is None:
            return set()
        atoms: set[AtomId] = set()

        h_edges = 0
        v_edges = 0
        rows = len(grid)
        cols = len(grid[0]) if rows > 0 else 0

        # 1) Parches 2x2
        for r in range(rows - 1):
            for c in range(cols - 1):
                q_tl = grid[r][c] >> 4
                q_tr = grid[r][c + 1] >> 4
                q_bl = grid[r + 1][c] >> 4
                q_br = grid[r + 1][c + 1] >> 4
                surface = f"P2x2:{q_tl}_{q_tr}_{q_bl}_{q_br}"
                atoms.add(registry.register(surface, self.DOMAIN))

        # 2) Bordes horizontales (dentro de cada fila, comparando con el vecino derecho)
        for r in range(rows):
            for c in range(cols - 1):
                if abs(grid[r][c + 1] - grid[r][c]) > _EDGE_THRESHOLD:
                    h_edges += 1
        # 3) Bordes verticales (dentro de cada columna, comparando con el vecino inferior)
        for c in range(cols):
            for r in range(rows - 1):
                if abs(grid[r + 1][c] - grid[r][c]) > _EDGE_THRESHOLD:
                    v_edges += 1

        atoms.add(registry.register(f"HEDGE:{self._bin_edges(h_edges)}", self.DOMAIN))
        atoms.add(registry.register(f"VEDGE:{self._bin_edges(v_edges)}", self.DOMAIN))
        return atoms

    @staticmethod
    def _coerce_grid(payload: Any) -> list[list[int]] | None:
        """Acepta list[list[int]] o tuplas anidadas. Devuelve None si no se puede."""
        if not isinstance(payload, (list, tuple)) or not payload:
            return None
        first = payload[0]
        if not isinstance(first, (list, tuple)):
            return None
        try:
            return [[int(v) for v in row] for row in payload]
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _bin_edges(count: int) -> str:
        """Binning de cuentas de bordes en categorías discretas."""
        if count <= 1:
            return "LOW"
        if count <= 3:
            return "MID"
        return "HIGH"


# ---------------------------------------------------------------------------
# Número
# ---------------------------------------------------------------------------


class NumberFactorExtractor(AtomExtractor):
    """Entero -> átomos numéricos.

    Estrategia (implementación funcional):
        1. Signo: POS, NEG, ZERO.
        2. Magnitud bucketizada: ZERO, SMALL (1-9), MEDIUM (10-99), LARGE (100+).
        3. Factores primos con multiplicidad: "PRIME:2" aparece DOS veces
           para el número 4 (= 2 * 2). Esto preserva la información de
           potencia sin recurrir a un set (que la perdería).
        4. Bigramas de dígitos: para el número 123 -> "DIG:12", "DIG:23".
    """

    DOMAIN: AtomDomain = AtomDomain.NUMBER

    def extract(self, payload: Any, registry: AtomRegistry) -> set[AtomId]:
        if not isinstance(payload, int) or isinstance(payload, bool):
            return set()
        atoms: set[AtomId] = set()
        atoms.add(registry.register(self._sign_atom(payload), self.DOMAIN))
        atoms.add(registry.register(self._magnitude_atom(payload), self.DOMAIN))
        for surface in self._prime_factors_with_multiplicity(payload):
            atoms.add(registry.register(surface, self.DOMAIN))
        for surface in self._digit_bigrams(payload):
            atoms.add(registry.register(surface, self.DOMAIN))
        return atoms

    @staticmethod
    def _sign_atom(n: int) -> str:
        if n == 0:
            return "ZERO"
        if n > 0:
            return "POS"
        return "NEG"

    @staticmethod
    def _magnitude_atom(n: int) -> str:
        a = abs(n)
        if a == 0:
            return "MAG:ZERO"
        if a < 10:
            return "MAG:SMALL"
        if a < 100:
            return "MAG:MEDIUM"
        return "MAG:LARGE"

    @staticmethod
    def _prime_factors_with_multiplicity(n: int) -> list[str]:
        """Devuelve ['PRIME:2', 'PRIME:2', 'PRIME:3'] para n=12 (=2*2*3)."""
        a = abs(n)
        if a < 2:
            return []
        factors: list[str] = []
        d = 2
        while d * d <= a:
            while a % d == 0:
                factors.append(f"PRIME:{d}")
                a //= d
            d += 1
        if a > 1:
            factors.append(f"PRIME:{a}")
        return factors

    @staticmethod
    def _digit_bigrams(n: int) -> list[str]:
        """Bigramas de dígitos: 123 -> ['DIG:12', 'DIG:23'].

        Para n=0, devuelve ['DIG:0']. Para n negativos, opera sobre |n|.
        """
        s = str(abs(n))
        if len(s) < 2:
            return [f"DIG:{s}"]
        return [f"DIG:{s[i]}{s[i + 1]}" for i in range(len(s) - 1)]


# ---------------------------------------------------------------------------
# Multimodal (composición)
# ---------------------------------------------------------------------------


class MultimodalFusionExtractor(AtomExtractor):
    """Combina varios sub-extractores en un átomo ancla multimodal.

    Payload esperado: `Mapping[str, Any]` (p.ej. {"text": "perro",
    "image": grid, "number": 4}). Para cada par (key, value), se elige
    el extractor adecuado siguiendo este ORDEN de prioridad:
        1. Mapeo explícito por CLAVE (self._by_key) — preferente.
           Ej: {"image": RealImageExtractor, "text": BigramExtractor}.
        2. Mapeo por TIPO Python (self._by_type) — fallback.
        3. Heurística: si el valor es un str que parece ruta de imagen
           (.png, .jpg, .bmp, .jpeg), usar RealImageExtractor.
    Cada sub-extractor se aplica al valor y se acumulan los átomos.
    Adicionalmente se crea un átomo ancla "CO_OCC:<hash>" que es la
    firma del co-ocurrimiento completo: este es el átomo que porta la
    información genuinamente multimodal (no presente en ningún
    sub-dominio).
    """

    DOMAIN: AtomDomain = AtomDomain.MULTIMODAL

    _IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".bmp", ".gif", ".tiff")

    def __init__(
        self,
        extractors_by_type: Mapping[type, AtomExtractor] | None = None,
        extractors_by_key: Mapping[str, AtomExtractor] | None = None,
    ) -> None:
        # Mapeo por defecto: tipo Python -> extractor. El cliente puede
        # sobreescribir pasando uno propio.
        default_map: dict[type, AtomExtractor] = {
            str: BigramExtractor(),
            int: NumberFactorExtractor(),
        }
        default_map[list] = RealImageExtractor()
        default_map[tuple] = RealImageExtractor()
        if extractors_by_type is not None:
            default_map.update(dict(extractors_by_type))
        self._by_type: dict[type, AtomExtractor] = default_map
        # Mapeo por clave (preferente). Default: 'image' -> RealImage.
        default_key_map: dict[str, AtomExtractor] = {
            "image": RealImageExtractor(),
        }
        if extractors_by_key is not None:
            default_key_map.update(dict(extractors_by_key))
        self._by_key: dict[str, AtomExtractor] = default_key_map

    def extract(self, payload: Any, registry: AtomRegistry) -> set[AtomId]:
        if not isinstance(payload, Mapping):
            return set()
        all_atoms: set[AtomId] = set()
        surfaces: list[str] = []
        for key, value in payload.items():
            extractor = self._select_extractor(key, value)
            if extractor is None:
                continue
            sub_atoms = extractor.extract(value, registry)
            all_atoms.update(sub_atoms)
            for a in sub_atoms:
                surfaces.append(registry.surface_of(a))
        # Átomo ancla multimodal: hash determinista del co-ocurrimiento.
        surfaces.sort()
        digest = hashlib.sha1("|".join(surfaces).encode("utf-8")).hexdigest()[:8]
        anchor = f"CO_OCC:{digest}"
        all_atoms.add(registry.register(anchor, self.DOMAIN))
        return all_atoms

    def _select_extractor(
        self, key: str, value: Any
    ) -> AtomExtractor | None:
        # 1) Por clave (preferente).
        if key in self._by_key:
            return self._by_key[key]
        # 2) Por tipo Python.
        extractor = self._by_type.get(type(value))
        if extractor is not None:
            return extractor
        # 3) Heurística: string que parece ruta de imagen.
        if isinstance(value, str) and value.lower().endswith(self._IMAGE_EXTENSIONS):
            return self._by_type.get(list, RealImageExtractor())
        # 4) Fallbacks de tipo.
        if isinstance(value, bool):
            return self._by_type.get(int)
        if isinstance(value, (list, tuple)):
            return self._by_type.get(list)
        return None


# ---------------------------------------------------------------------------
# Imagen real (PIL): hash perceptual + regiones + color + bordes
# ---------------------------------------------------------------------------


def _try_pil():
    """Intenta importar PIL. Devuelve el módulo o None si no está disponible."""
    try:
        from PIL import Image  # type: ignore
        return Image
    except ImportError:
        return None


class RealImageExtractor(AtomExtractor):
    """Extractor para imágenes reales (archivos PNG/JPG/BMP o PIL.Image).

    Estrategia de características (todas en dominio IMAGE):
        1. PHASH global: aHash sobre la imagen reducida a 16x16.
           Compara cada pixel con la media → string de 16 bits.
        2. PHASH por región: la imagen se divide en 3 bandas
           (top/mid/bot) y se hashea cada una. Esto permite que una
           imagen de "cuerpo" comparta el PHASH:top con las imágenes
           de "cara" y el PHASH:bot con imágenes de "brazo".
        3. Histograma de luminancia: 3 bins (oscuro/medio/claro).
        4. Densidad de bordes: discretizada en LOW/MID/HIGH.
        5. Proporción de aspecto: SQUARE/TALL/WIDE.

    El conjunto de átomos resultantes es lo que el motor usa para
    agrupar imágenes por similitud. Imágenes con regiones y PHASH
    similares caen en el mismo cluster; las que no, no.
    """

    DOMAIN: AtomDomain = AtomDomain.IMAGE

    # Tamaño de reducción para el hash perceptual.
    _HASH_SIZE: int = 8  # 8x8 = 64 bits por hash

    def __init__(self, *, prefer_pil: bool = True) -> None:
        self._pil = _try_pil() if prefer_pil else None

    def extract(self, payload: Any, registry: AtomRegistry) -> set[AtomId]:
        grid = self._coerce_to_grid(payload)
        if grid is None:
            return set()
        # Si la imagen es pequeña, el hash puede ser ruidoso; trabajamos
        # sobre la rejilla tal cual o normalizada.
        atoms: set[AtomId] = set()
        h = len(grid)
        w = len(grid[0]) if h else 0
        if h == 0 or w == 0:
            return atoms

        # 1) PHASH global
        g_hash = self._average_hash(grid)
        atoms.add(registry.register(f"PHASH:{g_hash}", self.DOMAIN))

        # 2) PHASH por regiones (3 bandas horizontales)
        for region_name in ("top", "mid", "bot"):
            region = self._slice_region(grid, region_name)
            r_hash = self._average_hash(region)
            atoms.add(registry.register(f"PHASH:{region_name}:{r_hash}", self.DOMAIN))

        # 3) Histograma de luminancia
        for bin_name in self._luminance_histogram(grid):
            atoms.add(registry.register(f"LUM:{bin_name}", self.DOMAIN))

        # 4) Densidad de bordes
        edge_density = self._edge_density(grid)
        atoms.add(registry.register(f"EDGES:{self._bin_edges(edge_density)}", self.DOMAIN))

        # 5) Proporción de aspecto
        atoms.add(registry.register(f"ASPECT:{self._aspect_bin(w, h)}", self.DOMAIN))

        return atoms

    # --- Carga y coerción ---

    def _coerce_to_grid(self, payload: Any) -> list[list[int]] | None:
        """Acepta str (ruta a archivo), PIL.Image, o list[list[int]]."""
        # 1) Ruta a archivo (str)
        if isinstance(payload, str) and self._pil is not None:
            try:
                img = self._pil.open(payload).convert("L")  # grayscale
                # Redimensionar a un tamaño fijo y manejable.
                img = img.resize((32, 32))
                return [[img.getpixel((x, y)) for x in range(32)] for y in range(32)]
            except (OSError, ValueError):
                return None
        # 2) PIL.Image
        if self._pil is not None and isinstance(payload, self._pil.Image):
            try:
                img = payload.convert("L")
                img = img.resize((32, 32))
                return [[img.getpixel((x, y)) for x in range(32)] for y in range(32)]
            except (ValueError, AttributeError):
                return None
        # 3) Lista anidada de enteros
        if isinstance(payload, (list, tuple)) and payload:
            try:
                return [[int(v) for v in row] for row in payload]
            except (TypeError, ValueError):
                return None
        return None

    # --- Hash perceptual (aHash) ---

    def _average_hash(self, grid: list[list[int]]) -> str:
        """aHash: media de los pixeles, bit = 1 si pixel >= media."""
        h = len(grid)
        w = len(grid[0]) if h else 0
        flat = [grid[r][c] for r in range(h) for c in range(w)]
        if not flat:
            return "0" * (self._HASH_SIZE * self._HASH_SIZE)
        mean = sum(flat) / len(flat)
        # Sub-muestrear a HASH_SIZE x HASH_SIZE
        bits: list[str] = []
        step_r = max(1, h // self._HASH_SIZE)
        step_c = max(1, w // self._HASH_SIZE)
        for r in range(0, h, step_r)[:self._HASH_SIZE] if False else \
                [r0 for r0 in range(0, h, max(1, h // self._HASH_SIZE))][:self._HASH_SIZE]:
            for c in range(0, w, max(1, w // self._HASH_SIZE))[:self._HASH_SIZE]:
                val = grid[r][c]
                bits.append("1" if val >= mean else "0")
        # Truncar/rellenar a longitud fija
        target = self._HASH_SIZE * self._HASH_SIZE
        s = "".join(bits)[:target]
        return s.ljust(target, "0")

    def _slice_region(self, grid: list[list[int]], name: str) -> list[list[int]]:
        """Devuelve la región top/mid/bot como una sub-rejilla."""
        h = len(grid)
        third = h // 3
        if name == "top":
            return [row[:] for row in grid[:third]]
        if name == "mid":
            return [row[:] for row in grid[third:2 * third]]
        if name == "bot":
            return [row[:] for row in grid[2 * third:]]
        return grid

    # --- Histograma de luminancia ---

    def _luminance_histogram(self, grid: list[list[int]]) -> list[str]:
        """Devuelve los bins no vacíos: DARK (<85), MID (85-170), LIGHT (>170)."""
        h = len(grid)
        w = len(grid[0]) if h else 0
        dark = mid = light = 0
        total = 0
        for r in range(h):
            for c in range(w):
                v = grid[r][c]
                total += 1
                if v < 85:
                    dark += 1
                elif v < 170:
                    mid += 1
                else:
                    light += 1
        if total == 0:
            return []
        bins: list[str] = []
        # Solo añadimos bins con > 5% de pixeles (evita átomos espurios)
        if dark / total > 0.05:
            bins.append("DARK")
        if mid / total > 0.05:
            bins.append("MID")
        if light / total > 0.05:
            bins.append("LIGHT")
        return bins

    # --- Densidad de bordes ---

    def _edge_density(self, grid: list[list[int]]) -> int:
        """Número de transiciones de intensidad > umbral entre pixeles vecinos."""
        h = len(grid)
        w = len(grid[0]) if h else 0
        count = 0
        for r in range(h):
            for c in range(w - 1):
                if abs(grid[r][c + 1] - grid[r][c]) > 32:
                    count += 1
        for c in range(w):
            for r in range(h - 1):
                if abs(grid[r + 1][c] - grid[r][c]) > 32:
                    count += 1
        return count

    @staticmethod
    def _bin_edges(count: int) -> str:
        if count <= 5:
            return "LOW"
        if count <= 20:
            return "MID"
        return "HIGH"

    # --- Proporción de aspecto ---

    @staticmethod
    def _aspect_bin(w: int, h: int) -> str:
        if h == 0:
            return "SQUARE"
        ratio = w / h
        if 0.85 <= ratio <= 1.15:
            return "SQUARE"
        if ratio < 0.85:
            return "TALL"
        return "WIDE"
