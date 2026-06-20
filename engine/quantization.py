"""
Cuantización estricta de 4 bits con signo (Tee & Taylor).

Pilar 2 del sistema: la información se almacena de forma discreta, no continua.
Las medidas de relación solo pueden tomar valores enteros en el rango [-8, +7].
Prohibido el uso de punto flotante para almacenamiento a largo plazo.

El cuantizador no sabe nada de átomos ni de dominios: es una pieza pura
de teoría de números, reutilizable por la AssociationMemory y por
cualquier otra estructura que necesite saturación discreta.

Principio SOLID aplicado:
    - Single Responsibility: esta clase SOLO clamea y redondea al rango de 4 bits.
"""

from __future__ import annotations


class Quantizer:
    """Cuantizador a 4 bits con signo. Inmutable y sin estado."""

    MIN_MEASURE: int = -8
    MAX_MEASURE: int = 7

    @staticmethod
    def quantize(value: int) -> int:
        """Mapea cualquier entero al rango [-8, 7] preservando el signo.

        Saturación dura: cualquier valor fuera del rango se clipea al extremo
        correspondiente. No se permite wrap-around (que destruiría la
        semántica Hebbiana).
        """
        if value > Quantizer.MAX_MEASURE:
            return Quantizer.MAX_MEASURE
        if value < Quantizer.MIN_MEASURE:
            return Quantizer.MIN_MEASURE
        return int(round(value))

    @staticmethod
    def is_within_range(value: int) -> bool:
        return Quantizer.MIN_MEASURE <= value <= Quantizer.MAX_MEASURE

    @staticmethod
    def range() -> tuple[int, int]:
        return Quantizer.MIN_MEASURE, Quantizer.MAX_MEASURE
