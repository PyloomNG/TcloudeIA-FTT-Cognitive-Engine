"""
Genera imagenes de digitos 0-9 en DOS tipografias diferentes para
demostrar que el motor agrupa por similitud de rasgos, no por la
superficie exacta.

Imagenes resultantes en `training/samples/digit_<N>_<style>.png`:
    digit_0_serif.png   digit_0_sans.png
    digit_1_serif.png   digit_1_sans.png
    ...
"""

from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
SAMPLES_DIR = _ROOT / "training" / "samples"
SAMPLES_DIR.mkdir(parents=True, exist_ok=True)

# Solo PIL.ImageDraw (no fuentes externas: dibujamos los digitos a mano).
# Cada digito se construye con primitivas simples: lineas, rectangulos, arcos.


def _serif_one(size: int = 32) -> list[list[int]]:
    """Digito 1 estilo serif: una barra vertical con serif arriba y base."""
    grid = [[255] * size for _ in range(size)]
    # Serif superior (horizontal)
    for c in range(8, 16):
        grid[5][c] = 0
    # Trazo vertical principal
    for r in range(5, 27):
        grid[r][13] = 0
        grid[r][14] = 0
    # Base (pequena serif abajo)
    for c in range(9, 19):
        grid[27][c] = 0
    return grid


def _sans_one(size: int = 32) -> list[list[int]]:
    """Digito 1 sans-serif: solo el trazo vertical principal, sin serif."""
    grid = [[255] * size for _ in range(size)]
    for r in range(4, 28):
        grid[r][14] = 0
        grid[r][15] = 0
    return grid


def _serif_two(size: int = 32) -> list[list[int]]:
    """Digito 2: curva arriba, diagonal, base."""
    grid = [[255] * size for _ in range(size)]
    # Curva superior
    for c in range(10, 22):
        grid[6][c] = 0
        grid[7][c] = 0
    for r in range(6, 12):
        grid[r][21] = 0
    # Diagonal
    for i, (r, c) in enumerate(zip(range(11, 22), range(20, 7, -1))):
        grid[r][c] = 0
        if i < 12:
            grid[r][c - 1] = 0
    # Base
    for c in range(8, 24):
        grid[25][c] = 0
        grid[26][c] = 0
    return grid


def _sans_two(size: int = 32) -> list[list[int]]:
    """Digito 2 sans: angulos rectos, sin curvas."""
    grid = [[255] * size for _ in range(size)]
    # Linea superior
    for c in range(8, 24):
        grid[5][c] = 0
        grid[6][c] = 0
    # Lateral derecho superior
    for r in range(5, 14):
        grid[r][22] = 0
        grid[r][23] = 0
    # Linea media
    for c in range(8, 22):
        grid[15][c] = 0
        grid[16][c] = 0
    # Lateral izquierdo inferior
    for r in range(15, 25):
        grid[r][8] = 0
        grid[r][9] = 0
    # Base
    for c in range(8, 24):
        grid[25][c] = 0
        grid[26][c] = 0
    return grid


def _make_digit_three(value: int, style: str) -> list[list[int]]:
    """Digitos 3-9: version simplificada con dos segmentos horizontales y laterales."""
    size = 32
    grid = [[255] * size for _ in range(size)]
    # Tres segmentos horizontales
    for c in range(8, 24):
        grid[5][c] = 0
        grid[6][c] = 0
        grid[15][c] = 0
        grid[16][c] = 0
        grid[25][c] = 0
        grid[26][c] = 0
    # Laterales (estilo 3: solo derecha)
    if value in (3,):
        for r in range(5, 26):
            grid[r][22] = 0
            grid[r][23] = 0
    elif value in (4,):
        for r in range(5, 15):
            grid[r][14] = 0
        for r in range(5, 26):
            grid[r][22] = 0
            grid[r][23] = 0
    elif value in (5,):
        for r in range(5, 15):
            grid[r][8] = 0
            grid[r][9] = 0
        for c in range(8, 24):
            grid[5][c] = 0
        # segmento medio
        for c in range(8, 24):
            grid[15][c] = 0
        for r in range(15, 26):
            grid[r][22] = 0
    elif value in (6,):
        for r in range(5, 26):
            grid[r][8] = 0
            grid[r][9] = 0
        for c in range(8, 24):
            grid[15][c] = 0
    elif value in (7,):
        for c in range(8, 24):
            grid[5][c] = 0
        for r in range(5, 16):
            grid[r][22] = 0
        for r in range(15, 26):
            if r > 16 + (r - 16):
                grid[r][22 - (r - 16)] = 0
    elif value in (8,):
        for r in range(5, 26):
            grid[r][8] = 0
            grid[r][9] = 0
            grid[r][22] = 0
            grid[r][23] = 0
    elif value in (9,):
        for r in range(5, 26):
            grid[r][22] = 0
            grid[r][23] = 0
        for c in range(8, 24):
            grid[5][c] = 0
        for r in range(5, 15):
            grid[r][8] = 0
    # Para el estilo "sans", mas grueso (3 pixeles en cada linea)
    if style == "sans":
        for r in range(size):
            for c in range(size):
                if grid[r][c] == 0:
                    # Vecinos: si alguno es 0, pon 0 tambien
                    for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                        nr, nc = r + dr, c + dc
                        if 0 <= nr < size and 0 <= nc < size:
                            grid[nr][nc] = 0
    return grid


def save_digit(value: int, style: str) -> None:
    name = f"digit_{value}_{style}.png"
    if value == 1 and style == "serif":
        grid = _serif_one()
    elif value == 1 and style == "sans":
        grid = _sans_one()
    elif value == 2 and style == "serif":
        grid = _serif_two()
    elif value == 2 and style == "sans":
        grid = _sans_two()
    else:
        grid = _make_digit_three(value, style)
    from PIL import Image
    img = Image.new("L", (32, 32), 255)
    for r, row in enumerate(grid):
        for c, v in enumerate(row):
            img.putpixel((c, r), v)
    img.save(SAMPLES_DIR / name)


def main() -> None:
    print("Generando digitos en", SAMPLES_DIR)
    for value in range(10):
        save_digit(value, "serif")
        save_digit(value, "sans")
    print(f"OK. {10 * 2} imagenes generadas.")


if __name__ == "__main__":
    main()
