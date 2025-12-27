from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

import numpy as np

Vector = np.ndarray


def _to_vector(values: Iterable[float], *, length: int | None = None) -> Vector:
    arr = np.asarray(list(values), dtype=float)
    if arr.ndim != 1:
        raise ValueError("Vector must be 1D")
    if length is not None and arr.size != length:
        raise ValueError(f"Vector must have length {length}")
    return arr


@dataclass
class PointMass:
    entity_id: str
    mass: float
    position: Vector = field(default_factory=lambda: np.zeros(2, dtype=float))
    velocity: Vector = field(default_factory=lambda: np.zeros(2, dtype=float))

    def __post_init__(self) -> None:
        if self.mass <= 0:
            raise ValueError("Mass must be positive")
        self.position = _to_vector(self.position)
        self.velocity = _to_vector(self.velocity, length=self.position.size)

    @property
    def dimension(self) -> int:
        return int(self.position.size)
