from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Iterable, List

import numpy as np

from ..model import PointMass, Vector
from .com import center_of_mass


class FrameChoice(str, Enum):
    WORLD = "world"
    COM = "com"


@dataclass(frozen=True)
class VectorSegment:
    start: Vector
    end: Vector


@dataclass(frozen=True)
class FrameVectors:
    frame: FrameChoice
    entity_ids: List[str]
    masses: List[float]
    positions: List[Vector]
    origin: Vector
    com: Vector
    r_op_segments: List[VectorSegment]
    r_cp_segments: List[VectorSegment]
    r_oc_segment: VectorSegment
    r_oc_world: Vector
    dimension: int

    @classmethod
    def empty(cls, frame: FrameChoice, dimension: int = 2) -> "FrameVectors":
        zero = np.zeros(dimension, dtype=float)
        return cls(
            frame=frame,
            entity_ids=[],
            masses=[],
            positions=[],
            origin=zero,
            com=zero,
            r_op_segments=[],
            r_cp_segments=[],
            r_oc_segment=VectorSegment(start=zero, end=zero),
            r_oc_world=zero,
            dimension=dimension,
        )


def to_frame(point_world: Vector, frame: FrameChoice, r_oc_world: Vector) -> Vector:
    if frame == FrameChoice.WORLD:
        return point_world.copy()
    if frame == FrameChoice.COM:
        return point_world - r_oc_world
    raise ValueError(f"Unsupported frame: {frame}")


def from_frame(point_frame: Vector, frame: FrameChoice, r_oc_world: Vector) -> Vector:
    if frame == FrameChoice.WORLD:
        return point_frame.copy()
    if frame == FrameChoice.COM:
        return point_frame + r_oc_world
    raise ValueError(f"Unsupported frame: {frame}")


def compute_frame_vectors(entities: Iterable[PointMass], frame: FrameChoice) -> FrameVectors:
    entities_list = list(entities)
    if not entities_list:
        return FrameVectors.empty(frame)

    entity_ids = [entity.entity_id for entity in entities_list]
    masses = [entity.mass for entity in entities_list]
    r_op_list = [entity.position for entity in entities_list]
    r_oc_world = center_of_mass(entities_list)
    dimension = int(r_oc_world.size)

    origin_world = np.zeros(dimension, dtype=float)
    origin_frame = to_frame(origin_world, frame, r_oc_world)
    com_frame = to_frame(r_oc_world, frame, r_oc_world)
    positions_frame = [to_frame(pos, frame, r_oc_world) for pos in r_op_list]

    r_op_segments = [VectorSegment(start=origin_frame, end=pos) for pos in positions_frame]
    r_cp_segments = [VectorSegment(start=com_frame, end=pos) for pos in positions_frame]
    r_oc_segment = VectorSegment(start=origin_frame, end=com_frame)

    return FrameVectors(
        frame=frame,
        entity_ids=entity_ids,
        masses=masses,
        positions=positions_frame,
        origin=origin_frame,
        com=com_frame,
        r_op_segments=r_op_segments,
        r_cp_segments=r_cp_segments,
        r_oc_segment=r_oc_segment,
        r_oc_world=r_oc_world,
        dimension=dimension,
    )
