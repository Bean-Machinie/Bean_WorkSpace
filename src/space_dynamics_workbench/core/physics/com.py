from __future__ import annotations

from typing import Iterable

import numpy as np

from ..model import PointMass, Vector


def total_mass(entities: Iterable[PointMass]) -> float:
    masses = [entity.mass for entity in entities]
    return float(np.sum(masses))


def center_of_mass(entities: Iterable[PointMass]) -> Vector:
    entities_list = list(entities)
    if not entities_list:
        raise ValueError("No entities provided")
    masses = np.array([entity.mass for entity in entities_list], dtype=float)
    positions = np.stack([entity.position for entity in entities_list])
    return np.sum(positions * masses[:, None], axis=0) / np.sum(masses)


def center_of_velocity(entities: Iterable[PointMass]) -> Vector:
    entities_list = list(entities)
    if not entities_list:
        raise ValueError("No entities provided")
    masses = np.array([entity.mass for entity in entities_list], dtype=float)
    velocities = np.stack([entity.velocity for entity in entities_list])
    return np.sum(velocities * masses[:, None], axis=0) / np.sum(masses)


def relative_positions(entities: Iterable[PointMass]) -> list[Vector]:
    entities_list = list(entities)
    com = center_of_mass(entities_list)
    return [entity.position - com for entity in entities_list]


def relative_velocities(entities: Iterable[PointMass]) -> list[Vector]:
    entities_list = list(entities)
    cov = center_of_velocity(entities_list)
    return [entity.velocity - cov for entity in entities_list]


def invariant_position_sum(entities: Iterable[PointMass]) -> Vector:
    entities_list = list(entities)
    com = center_of_mass(entities_list)
    return np.sum(
        [entity.mass * (entity.position - com) for entity in entities_list],
        axis=0,
    )


def invariant_velocity_sum(entities: Iterable[PointMass]) -> Vector:
    entities_list = list(entities)
    cov = center_of_velocity(entities_list)
    return np.sum(
        [entity.mass * (entity.velocity - cov) for entity in entities_list],
        axis=0,
    )
