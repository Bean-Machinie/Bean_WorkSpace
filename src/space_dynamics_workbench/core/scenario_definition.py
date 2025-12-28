from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, List

import numpy as np

from .model import MeshMetadata, PointMass, RigidBody, RigidBodyComponent
from .sim import Integrator, Simulation, SymplecticEulerIntegrator

SCENARIO_SCHEMA_VERSION = 1

INTEGRATOR_IDS: dict[str, type[Integrator]] = {
    "symplectic_euler": SymplecticEulerIntegrator,
}


@dataclass
class SimulationDefinition:
    dt: float = 0.05
    integrator: str = "symplectic_euler"


@dataclass(frozen=True)
class RigidBodyComponentDefinition:
    component_id: str
    mass: float
    position_body: np.ndarray


@dataclass
class RigidBodyStateDefinition:
    com_position: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=float))
    com_velocity: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=float))
    orientation: np.ndarray = field(default_factory=lambda: np.array([1.0, 0.0, 0.0, 0.0], dtype=float))
    omega_world: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=float))


@dataclass
class RigidBodyDefinition:
    entity_id: str
    components: List[RigidBodyComponentDefinition]
    state: RigidBodyStateDefinition = field(default_factory=RigidBodyStateDefinition)
    mesh: MeshMetadata | None = None


@dataclass
class ScenarioDefinition:
    name: str
    simulation: SimulationDefinition = field(default_factory=SimulationDefinition)
    entities: List["EntityDefinition"] = field(default_factory=list)
    schema_version: int = SCENARIO_SCHEMA_VERSION
    created_at: str | None = None
    modified_at: str | None = None


def integrator_from_id(integrator_id: str) -> Integrator:
    integrator_cls = INTEGRATOR_IDS.get(integrator_id)
    if integrator_cls is None:
        raise ValueError(f"Unknown integrator id: {integrator_id}")
    return integrator_cls()


def integrator_id_from_instance(integrator: Integrator) -> str:
    for key, integrator_cls in INTEGRATOR_IDS.items():
        if isinstance(integrator, integrator_cls):
            return key
    raise ValueError(f"Unsupported integrator: {type(integrator)}")


def rigid_body_from_definition(defn: RigidBodyDefinition) -> RigidBody:
    components = [
        RigidBodyComponent(
            component_id=component.component_id,
            mass=float(component.mass),
            position_body=np.array(component.position_body, dtype=float),
        )
        for component in defn.components
    ]
    return RigidBody(
        entity_id=defn.entity_id,
        components=components,
        com_position=np.array(defn.state.com_position, dtype=float),
        com_velocity=np.array(defn.state.com_velocity, dtype=float),
        orientation=np.array(defn.state.orientation, dtype=float),
        omega_world=np.array(defn.state.omega_world, dtype=float),
        mesh=None if defn.mesh is None else MeshMetadata(
            path=str(defn.mesh.path),
            path_is_absolute=bool(defn.mesh.path_is_absolute),
            scale=np.array(defn.mesh.scale, dtype=float),
            offset_body=np.array(defn.mesh.offset_body, dtype=float),
            rotation_body=np.array(defn.mesh.rotation_body, dtype=float),
        ),
    )


def point_mass_from_definition(defn: PointMassDefinition) -> PointMass:
    return PointMass(
        entity_id=defn.entity_id,
        mass=float(defn.mass),
        position=np.array(defn.position, dtype=float),
        velocity=np.array(defn.velocity, dtype=float),
    )


def entities_from_definition(entities: Iterable[EntityDefinition]) -> list[PointMass | RigidBody]:
    runtime: list[PointMass | RigidBody] = []
    for entity in entities:
        if isinstance(entity, PointMassDefinition):
            runtime.append(point_mass_from_definition(entity))
        else:
            runtime.append(rigid_body_from_definition(entity))
    return runtime


def simulation_from_definition(defn: ScenarioDefinition) -> Simulation:
    return Simulation(
        entities=entities_from_definition(defn.entities),
        dt=float(defn.simulation.dt),
        integrator=integrator_from_id(defn.simulation.integrator),
    )


def definition_from_simulation(name: str, sim: Simulation) -> ScenarioDefinition:
    entities: list[EntityDefinition] = []
    for entity in sim.entities:
        if isinstance(entity, PointMass):
            entities.append(
                PointMassDefinition(
                    entity_id=entity.entity_id,
                    mass=float(entity.mass),
                    position=np.array(entity.position, dtype=float),
                    velocity=np.array(entity.velocity, dtype=float),
                )
            )
        elif isinstance(entity, RigidBody):
            components = [
                RigidBodyComponentDefinition(
                    component_id=component.component_id,
                    mass=float(component.mass),
                    position_body=np.array(component.position_body, dtype=float),
                )
                for component in entity.components
            ]
            state = RigidBodyStateDefinition(
                com_position=np.array(entity.com_position, dtype=float),
                com_velocity=np.array(entity.com_velocity, dtype=float),
                orientation=np.array(entity.orientation, dtype=float),
                omega_world=np.array(entity.omega_world, dtype=float),
            )
            mesh = None
            if entity.mesh is not None:
                mesh = MeshMetadata(
                    path=str(entity.mesh.path),
                    path_is_absolute=bool(entity.mesh.path_is_absolute),
                    scale=np.array(entity.mesh.scale, dtype=float),
                    offset_body=np.array(entity.mesh.offset_body, dtype=float),
                    rotation_body=np.array(entity.mesh.rotation_body, dtype=float),
                )
            entities.append(
                RigidBodyDefinition(entity_id=entity.entity_id, components=components, state=state, mesh=mesh)
            )
    return ScenarioDefinition(
        name=name,
        simulation=SimulationDefinition(
            dt=float(sim.dt),
            integrator=integrator_id_from_instance(sim.integrator),
        ),
        entities=entities,
    )
@dataclass
class PointMassDefinition:
    entity_id: str
    mass: float
    position: np.ndarray = field(default_factory=lambda: np.zeros(2, dtype=float))
    velocity: np.ndarray = field(default_factory=lambda: np.zeros(2, dtype=float))


EntityDefinition = RigidBodyDefinition | PointMassDefinition
