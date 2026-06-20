"""
Genera imágenes de botellas de PLÁSTICO y de VIDRIO en `training/samples/`.

Las imágenes son 32x32 en escala de grises. La idea es que el
RealImageExtractor pueda distinguirlas por:
    - Histograma de luminancia (plástico opaco vs vidrio con reflejos).
    - Densidad de bordes (vidrio tiene más reflejos -> más bordes).
    - Distribución espacial (cuello estrecho en vidrio, ancho en plástico).

Esto demuestra que el motor agrupa automáticamente "todo lo que se
parece" sin que el usuario le diga qué tipo de botella es cada una.
"""

from __future__ import annotations

import random
import sys
from pathlib import Path

from PIL import Image, ImageDraw

_ROOT = Path(__file__).resolve().parent.parent
SAMPLES_DIR = _ROOT / "training" / "samples"
SAMPLES_DIR.mkdir(parents=True, exist_ok=True)


def _save(img: Image.Image, name: str) -> Path:
    path = SAMPLES_DIR / name
    img.save(path)
    return path


def make_plastic_bottle(seed: int, cap_color: tuple[int, int, int]) -> Image.Image:
    """Botella de plástico: cuerpo opaco uniforme, cap grande, etiqueta rectangular."""
    rng = random.Random(seed)
    img = Image.new("RGB", (32, 32), (245, 245, 245))  # fondo claro
    draw = ImageDraw.Draw(img)
    # Cuerpo (opaco, color claro, ancho)
    body_color = (180, 210, 230)  # azul claro opaco
    for y in range(8, 28):
        # Pequeño ruido en el cuerpo (textura plástica)
        for x in range(8, 24):
            noise = rng.randint(-8, 8)
            r = max(0, min(255, body_color[0] + noise))
            g = max(0, min(255, body_color[1] + noise))
            b = max(0, min(255, body_color[2] + noise))
            draw.point((x, y), fill=(r, g, b))
    # Cap (ancho, opaco, color)
    for y in range(3, 8):
        for x in range(10, 22):
            noise = rng.randint(-15, 15)
            r = max(0, min(255, cap_color[0] + noise))
            g = max(0, min(255, cap_color[1] + noise))
            b = max(0, min(255, cap_color[2] + noise))
            draw.point((x, y), fill=(r, g, b))
    # Etiqueta blanca (rectángulo horizontal)
    for x in range(9, 23):
        for y in range(15, 21):
            noise = rng.randint(-5, 5)
            v = max(0, min(255, 250 + noise))
            draw.point((x, y), fill=(v, v, v))
    return img.convert("L")


def make_glass_bottle(seed: int, tint: tuple[int, int, int]) -> Image.Image:
    """Botella de vidrio: cuerpo transparente con reflejos, cuello estrecho, sin etiqueta."""
    rng = random.Random(seed)
    img = Image.new("RGB", (32, 32), (250, 250, 250))  # fondo casi blanco
    draw = ImageDraw.Draw(img)
    # Cuerpo (estrecho, transparente: muestra el fondo, con tinte sutil)
    # Líneas verticales: simulan reflejos
    for x in range(11, 21):
        for y in range(10, 28):
            v = 200 + rng.randint(-15, 15)
            v = max(150, min(255, v))
            # Tinte del vidrio
            r = max(0, min(255, int(v * tint[0] / 255) + 20))
            g = max(0, min(255, int(v * tint[1] / 255) + 20))
            b = max(0, min(255, int(v * tint[2] / 255) + 20))
            draw.point((x, y), fill=(r, g, b))
    # Reflejo vertical brillante (vidrio)
    for y in range(10, 28):
        v = 240 + rng.randint(-10, 10)
        v = max(200, min(255, v))
        draw.point((13, y), fill=(v, v, v))
    # Cuello estrecho
    for y in range(5, 10):
        for x in range(13, 19):
            v = 100 + rng.randint(-10, 10)
            v = max(70, min(255, v))
            draw.point((x, y), fill=(v, v, v))
    # Cap pequeño y oscuro (corcho o metal)
    for y in range(2, 5):
        for x in range(14, 18):
            v = 40 + rng.randint(-5, 5)
            v = max(20, min(255, v))
            draw.point((x, y), fill=(v, v, v))
    return img.convert("L")


def make_other_objects() -> None:
    """Genera también un par de objetos NO botellas para que el clustering
    no agrupe todo. Sirve para que el Jaccard sea discriminante."""
    # Lata de refresco: muy diferente a una botella
    img = Image.new("RGB", (32, 32), (240, 240, 240))
    draw = ImageDraw.Draw(img)
    # Cuerpo cilíndrico corto y ancho con colores brillantes
    for y in range(10, 26):
        for x in range(7, 25):
            # Bandas rojas y blancas
            if 12 <= y <= 14 or 18 <= y <= 20:
                color = (220, 30, 30)  # rojo brillante
            else:
                color = (245, 245, 245)  # blanco
            draw.point((x, y), fill=color)
    # Tab superior
    for y in range(6, 10):
        for x in range(7, 25):
            draw.point((x, y), fill=(120, 120, 130))
    # Lengüeta
    for x in range(20, 24):
        for y in range(4, 7):
            draw.point((x, y), fill=(180, 180, 180))
    _save(img.convert("L"), "can1.png")

    img2 = Image.new("RGB", (32, 32), (240, 240, 240))
    draw2 = ImageDraw.Draw(img2)
    # Mismo concepto, diferente color
    for y in range(10, 26):
        for x in range(7, 25):
            if 12 <= y <= 14 or 18 <= y <= 20:
                color = (30, 30, 220)  # azul
            else:
                color = (240, 240, 240)
            draw2.point((x, y), fill=color)
    for y in range(6, 10):
        for x in range(7, 25):
            draw2.point((x, y), fill=(110, 110, 120))
    for x in range(20, 24):
        for y in range(4, 7):
            draw2.point((x, y), fill=(170, 170, 170))
    _save(img2.convert("L"), "can2.png")


def main() -> None:
    print("Generando muestras de botellas en", SAMPLES_DIR)
    # 3 botellas de plástico (diferente color de cap)
    _save(make_plastic_bottle(seed=1, cap_color=(220, 30, 30)), "bottle_plastic_red.png")
    _save(make_plastic_bottle(seed=2, cap_color=(30, 200, 30)), "bottle_plastic_green.png")
    _save(make_plastic_bottle(seed=3, cap_color=(30, 30, 220)), "bottle_plastic_blue.png")
    # 3 botellas de vidrio (diferente tinte)
    _save(make_glass_bottle(seed=11, tint=(120, 180, 120)), "bottle_glass_green.png")
    _save(make_glass_bottle(seed=12, tint=(180, 180, 220)), "bottle_glass_clear.png")
    _save(make_glass_bottle(seed=13, tint=(200, 160, 120)), "bottle_glass_amber.png")
    # Objetos no-botella (latas) para que el Jaccard sea discriminante
    make_other_objects()
    print("OK. Listo para auto-clustering.")


if __name__ == "__main__":
    main()
