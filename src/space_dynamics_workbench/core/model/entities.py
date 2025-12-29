from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, List, Protocol, Tuple

import numpy as np

Vector = np.ndarray
ComponentIdSeparator = ":"


def _to_vector(values: Iterable[float], *, length: int | None = None) -> Vector:
    arr = np.asarray(list(values), dtype=float)
    if arr.ndim != 1:
        raise ValueError("Vector must be 1D")
    if length is not None and arr.size != length:
        raise ValueError(f"Vector must have length {length}")
    return arr


def _to_scale_vector(values: object) -> Vector:
    if np.isscalar(values):
        scale = float(values)
        return np.array([scale, scale, scale], dtype=float)
    return _to_vector(values, length=3)


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


@dataclass(frozen=True)
class RigidBodyComponent:
    component_id: str
    mass: float
    position_body: Vector

    def __post_init__(self) -> None:
        if self.mass <= 0:
            raise ValueError("Mass must be positive")
        object.__setattr__(self, "position_body", _to_vector(self.position_body, length=3))


@dataclass
class MeshMetadata:
    path: str
    path_is_absolute: bool = False
    scale: Vector = field(default_factory=lambda: np.ones(3, dtype=float))
    offset_body: Vector = field(default_factory=lambda: np.zeros(3, dtype=float))
    rotation_body: Vector = field(default_factory=lambda: np.array([1.0, 0.0, 0.0, 0.0], dtype=float))

    def __post_init__(self) -> None:
        self.scale = _to_scale_vector(self.scale)
        self.offset_body = _to_vector(self.offset_body, length=3)
        self.rotation_body = _normalize_quaternion(_to_vector(self.rotation_body, length=4))


class MassPointLike(Protocol):
    entity_id: str
    mass: float
    position: Vector
    velocity: Vector


@dataclass(frozen=True)
class MassPointState:
    entity_id: str
    mass: float
    position: Vector
    velocity: Vector


def _pad_vector(vec: Vector, length: int) -> Vector:
    arr = np.asarray(vec, dtype=float).reshape(-1)
    if arr.size > length:
        raise ValueError(f"Vector has length {arr.size} larger than {length}")
    if arr.size == length:
        return arr
    padded = np.zeros(length, dtype=float)
    padded[: arr.size] = arr
    return padded


def _normalize_quaternion(q: Vector) -> Vector:
    q_arr = np.asarray(q, dtype=float).reshape(4)
    norm = float(np.linalg.norm(q_arr))
    if norm <= 0.0:
        raise ValueError("Quaternion must be non-zero")
    return q_arr / norm


def _quat_multiply(q_left: Vector, q_right: Vector) -> Vector:
    w1, x1, y1, z1 = np.asarray(q_left, dtype=float).reshape(4)
    w2, x2, y2, z2 = np.asarray(q_right, dtype=float).reshape(4)
    return np.array(
        [
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ],
        dtype=float,
    )


def _quat_to_matrix(q: Vector) -> np.ndarray:
    w, x, y, z = _normalize_quaternion(q)
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=float,
    )


def _integrate_quaternion_body(q: Vector, omega_body: Vector, dt: float) -> Vector:
    omega = np.asarray(omega_body, dtype=float).reshape(3)
    angle = float(np.linalg.norm(omega)) * dt
    if angle <= 1e-12:
        return _normalize_quaternion(q)
    axis = omega / float(np.linalg.norm(omega))
    half = 0.5 * angle
    delta = np.array(
        [np.cos(half), axis[0] * np.sin(half), axis[1] * np.sin(half), axis[2] * np.sin(half)],
        dtype=float,
    )
    # Right-multiply because q maps body -> world and omega is expressed in body frame.
    return _normalize_quaternion(_quat_multiply(q, delta))


@dataclass
class RigidBody:
    entity_id: str
    components: List[RigidBodyComponent]
    com_position: Vector = field(default_factory=lambda: np.zeros(3, dtype=float))
    com_velocity: Vector = field(default_factory=lambda: np.zeros(3, dtype=float))
    orientation: Vector = field(default_factory=lambda: np.array([1.0, 0.0, 0.0, 0.0], dtype=float))
    # Interpreted in the body frame (components aligned with the spacecraft).
    omega_world: Vector = field(default_factory=lambda: np.zeros(3, dtype=float))
    mesh: MeshMetadata | None = None

    def __post_init__(self) -> None:
        if not self.components:
            raise ValueError("RigidBody must have at least one component")
        self.com_position = _to_vector(self.com_position, length=3)
        self.com_velocity = _to_vector(self.com_velocity, length=3)
        self.orientation = _normalize_quaternion(_to_vector(self.orientation, length=4))
        self.omega_world = _to_vector(self.omega_world, length=3)
        if self.mesh is not None and not isinstance(self.mesh, MeshMetadata):
            raise ValueError("mesh must be MeshMetadata or None")

    @property
    def total_mass(self) -> float:
        return float(np.sum([component.mass for component in self.components]))

    def rotation_matrix(self) -> np.ndarray:
        return _quat_to_matrix(self.orientation)

    def component_ids(self) -> List[str]:
        return [component.component_id for component in self.components]

    def component_entity_id(self, component_id: str) -> str:
        return f"{self.entity_id}{ComponentIdSeparator}{component_id}"

    def get_component(self, component_id: str) -> RigidBodyComponent | None:
        for component in self.components:
            if component.component_id == component_id:
                return component
        return None

    def component_positions_world(self) -> List[Vector]:
        rotation = self.rotation_matrix()
        return [self.com_position + rotation @ component.position_body for component in self.components]

    def component_velocities_world(self) -> List[Vector]:
        rotation = self.rotation_matrix()
        rel_positions = [rotation @ component.position_body for component in self.components]
        omega_world = rotation @ self.omega_world
        return [self.com_velocity + np.cross(omega_world, rel_pos) for rel_pos in rel_positions]

    def invariant_position_sum(self) -> Vector:
        rel_positions = self.component_positions_world()
        weighted = np.sum(
            [component.mass * (pos - self.com_position) for component, pos in zip(self.components, rel_positions)],
            axis=0,
        )
        return np.asarray(weighted, dtype=float)

    def pairwise_distances_world(self) -> np.ndarray:
        positions = np.stack(self.component_positions_world())
        count = positions.shape[0]
        distances = np.zeros((count, count), dtype=float)
        for i in range(count):
            for j in range(i + 1, count):
                distances[i, j] = np.linalg.norm(positions[i] - positions[j])
                distances[j, i] = distances[i, j]
        return distances

    def step_kinematic(self, dt: float) -> None:
        if dt <= 0:
            raise ValueError("dt must be positive")
        self.com_position = self.com_position + self.com_velocity * dt
        # TODO: replace with torque-driven attitude integration once torque models are available.
        self.orientation = _integrate_quaternion_body(self.orientation, self.omega_world, dt)


SimEntity = PointMass | RigidBody


def iter_mass_points(entities: Iterable[SimEntity]) -> List[MassPointState]:
    entities_list = list(entities)
    if not entities_list:
        return []
    max_dim = 0
    for entity in entities_list:
        if isinstance(entity, PointMass):
            max_dim = max(max_dim, int(entity.position.size))
        else:
            max_dim = max(max_dim, 3)
    mass_points: List[MassPointState] = []
    for entity in entities_list:
        if isinstance(entity, PointMass):
            position = _pad_vector(entity.position, max_dim)
            velocity = _pad_vector(entity.velocity, max_dim)
            mass_points.append(
                MassPointState(
                    entity_id=entity.entity_id,
                    mass=entity.mass,
                    position=position,
                    velocity=velocity,
                )
            )
        else:
            positions = entity.component_positions_world()
            velocities = entity.component_velocities_world()
            for component, pos, vel in zip(entity.components, positions, velocities):
                mass_points.append(
                    MassPointState(
                        entity_id=entity.component_entity_id(component.component_id),
                        mass=component.mass,
                        position=_pad_vector(pos, max_dim),
                        velocity=_pad_vector(vel, max_dim),
                    )
                )
    return mass_points


def resolve_entity_by_id(
    entities: Iterable[SimEntity], entity_id: str
) -> Tuple[SimEntity | None, RigidBodyComponent | None]:
    for entity in entities:
        if entity.entity_id == entity_id:
            return entity, None
        if isinstance(entity, RigidBody):
            prefix = f"{entity.entity_id}{ComponentIdSeparator}"
            if entity_id.startswith(prefix):
                component_id = entity_id[len(prefix) :]
                component = entity.get_component(component_id)
                if component is not None:
                    return entity, component
    return None, None
