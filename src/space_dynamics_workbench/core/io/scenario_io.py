from __future__ import annotations

import json
from typing import Any, Dict, List

import numpy as np

from ..model import MeshMetadata
from ..scenario_definition import (
    SCENARIO_SCHEMA_VERSION,
    PointMassDefinition,
    RigidBodyComponentDefinition,
    RigidBodyDefinition,
    RigidBodyStateDefinition,
    ScenarioDefinition,
    SimulationDefinition,
)


def scenario_definition_to_dict(defn: ScenarioDefinition) -> Dict[str, Any]:
    return {
        "schema_version": defn.schema_version,
        "name": defn.name,
        "created_at": defn.created_at,
        "modified_at": defn.modified_at,
        "simulation": {
            "dt": defn.simulation.dt,
            "integrator": defn.simulation.integrator,
        },
        "entities": [_entity_to_dict(entity) for entity in defn.entities],
    }


def _entity_to_dict(entity: RigidBodyDefinition | PointMassDefinition) -> Dict[str, Any]:
    if isinstance(entity, PointMassDefinition):
        return {
            "type": "point_mass",
            "entity_id": entity.entity_id,
            "mass": entity.mass,
            "position": entity.position.tolist(),
            "velocity": entity.velocity.tolist(),
        }
    return _rigid_body_to_dict(entity)


def _rigid_body_to_dict(entity: RigidBodyDefinition) -> Dict[str, Any]:
    payload: Dict[str, Any] = {
        "type": "rigid_body",
        "entity_id": entity.entity_id,
        "state": {
            "com_position": entity.state.com_position.tolist(),
            "com_velocity": entity.state.com_velocity.tolist(),
            "orientation": entity.state.orientation.tolist(),
            "omega_world": entity.state.omega_world.tolist(),
        },
        "components": [
            {
                "component_id": component.component_id,
                "mass": component.mass,
                "position_body": component.position_body.tolist(),
            }
            for component in entity.components
        ],
    }
    if entity.mesh is not None:
        payload["mesh"] = {
            "path": entity.mesh.path,
            "path_is_absolute": bool(entity.mesh.path_is_absolute),
            "scale": entity.mesh.scale.tolist(),
            "offset_body": entity.mesh.offset_body.tolist(),
            "rotation_body": entity.mesh.rotation_body.tolist(),
        }
    return payload


def scenario_definition_from_dict(payload: Dict[str, Any]) -> ScenarioDefinition:
    version = int(payload.get("schema_version", 0))
    if version != SCENARIO_SCHEMA_VERSION:
        raise ValueError(f"Unsupported scenario schema version: {version}")
    simulation_payload = payload.get("simulation", {})
    simulation = SimulationDefinition(
        dt=float(simulation_payload.get("dt", 0.05)),
        integrator=str(simulation_payload.get("integrator", "symplectic_euler")),
    )
    entities: List[RigidBodyDefinition | PointMassDefinition] = []
    for entity_payload in payload.get("entities", []):
        entity_type = entity_payload.get("type")
        if entity_type == "point_mass":
            entities.append(
                PointMassDefinition(
                    entity_id=str(entity_payload.get("entity_id", "")),
                    mass=float(entity_payload.get("mass", 1.0)),
                    position=np.array(entity_payload.get("position", [0.0, 0.0]), dtype=float),
                    velocity=np.array(entity_payload.get("velocity", [0.0, 0.0]), dtype=float),
                )
            )
            continue
        if entity_type not in (None, "rigid_body"):
            raise ValueError(f"Unsupported entity type: {entity_type}")
        state_payload = entity_payload.get("state", {})
        state = RigidBodyStateDefinition(
            com_position=np.array(state_payload.get("com_position", [0.0, 0.0, 0.0]), dtype=float),
            com_velocity=np.array(state_payload.get("com_velocity", [0.0, 0.0, 0.0]), dtype=float),
            orientation=np.array(state_payload.get("orientation", [1.0, 0.0, 0.0, 0.0]), dtype=float),
            omega_world=np.array(state_payload.get("omega_world", [0.0, 0.0, 0.0]), dtype=float),
        )
        components = [
            RigidBodyComponentDefinition(
                component_id=str(component.get("component_id", "")),
                mass=float(component.get("mass", 1.0)),
                position_body=np.array(component.get("position_body", [0.0, 0.0, 0.0]), dtype=float),
            )
            for component in entity_payload.get("components", [])
        ]
        mesh_data = entity_payload.get("mesh")
        mesh = None
        if isinstance(mesh_data, dict):
            path = str(mesh_data.get("path", "")).strip()
            if path:
                mesh = MeshMetadata(
                    path=path,
                    path_is_absolute=bool(mesh_data.get("path_is_absolute", False)),
                    scale=np.array(mesh_data.get("scale", [1.0, 1.0, 1.0]), dtype=float),
                    offset_body=np.array(mesh_data.get("offset_body", [0.0, 0.0, 0.0]), dtype=float),
                    rotation_body=np.array(mesh_data.get("rotation_body", [1.0, 0.0, 0.0, 0.0]), dtype=float),
                )
        entities.append(
            RigidBodyDefinition(
                entity_id=str(entity_payload.get("entity_id", "")),
                components=components,
                state=state,
                mesh=mesh,
            )
        )
    return ScenarioDefinition(
        name=str(payload.get("name", "Untitled Scenario")),
        created_at=payload.get("created_at"),
        modified_at=payload.get("modified_at"),
        simulation=simulation,
        entities=entities,
        schema_version=version,
    )


def serialize_scenario_definition(defn: ScenarioDefinition) -> str:
    return json.dumps(scenario_definition_to_dict(defn), indent=2)


def deserialize_scenario_definition(payload: str) -> ScenarioDefinition:
    return scenario_definition_from_dict(json.loads(payload))
