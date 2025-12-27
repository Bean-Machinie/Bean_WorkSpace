from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List

import numpy as np

from ..core.model import MeshMetadata, PointMass, RigidBody, RigidBodyComponent, SimEntity

SCENE_VERSION = 1


@dataclass
class SceneData:
    entities: List[SimEntity]
    scenario_id: str | None = None
    ui_state: Dict[str, Any] | None = None
    scene_version: int = SCENE_VERSION


def scene_to_dict(scene: SceneData) -> Dict[str, Any]:
    return {
        "scene_version": scene.scene_version,
        "scenario_id": scene.scenario_id,
        "entities": [
            _entity_to_dict(entity) for entity in scene.entities
        ],
        "ui_state": scene.ui_state or {},
    }


def _entity_to_dict(entity: SimEntity) -> Dict[str, Any]:
    if isinstance(entity, PointMass):
        return {
            "type": "point_mass",
            "entity_id": entity.entity_id,
            "mass": entity.mass,
            "position": entity.position.tolist(),
            "velocity": entity.velocity.tolist(),
        }
    if isinstance(entity, RigidBody):
        payload = {
            "type": "rigid_body",
            "entity_id": entity.entity_id,
            "com_position": entity.com_position.tolist(),
            "com_velocity": entity.com_velocity.tolist(),
            "orientation": entity.orientation.tolist(),
            "omega_world": entity.omega_world.tolist(),
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
    raise TypeError(f"Unsupported entity type: {type(entity)}")


def scene_from_dict(payload: Dict[str, Any]) -> SceneData:
    version = int(payload.get("scene_version", 0))
    if version != SCENE_VERSION:
        raise ValueError(f"Unsupported scene version: {version}")
    entities = []
    for entity_data in payload.get("entities", []):
        entity_type = entity_data.get("type", "point_mass")
        if entity_type == "point_mass":
            entities.append(
                PointMass(
                    entity_id=str(entity_data["entity_id"]),
                    mass=float(entity_data["mass"]),
                    position=np.array(entity_data["position"], dtype=float),
                    velocity=np.array(entity_data["velocity"], dtype=float),
                )
            )
        elif entity_type == "rigid_body":
            components = [
                RigidBodyComponent(
                    component_id=str(component["component_id"]),
                    mass=float(component["mass"]),
                    position_body=np.array(component["position_body"], dtype=float),
                )
                for component in entity_data.get("components", [])
            ]
            mesh_data = entity_data.get("mesh")
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
                RigidBody(
                    entity_id=str(entity_data["entity_id"]),
                    components=components,
                    com_position=np.array(entity_data["com_position"], dtype=float),
                    com_velocity=np.array(entity_data["com_velocity"], dtype=float),
                    orientation=np.array(entity_data["orientation"], dtype=float),
                    omega_world=np.array(entity_data["omega_world"], dtype=float),
                    mesh=mesh,
                )
            )
        else:
            raise ValueError(f"Unsupported entity type: {entity_type}")
    return SceneData(
        entities=entities,
        scenario_id=payload.get("scenario_id"),
        ui_state=dict(payload.get("ui_state", {})),
        scene_version=version,
    )


def serialize_scene(scene: SceneData) -> str:
    return json.dumps(scene_to_dict(scene), indent=2)


def deserialize_scene(payload: str) -> SceneData:
    return scene_from_dict(json.loads(payload))


def capture_scene(entities: Iterable[SimEntity], scenario_id: str | None = None) -> SceneData:
    return clone_scene(SceneData(entities=list(entities), scenario_id=scenario_id, ui_state={}))


def clone_scene(scene: SceneData) -> SceneData:
    entities_copy = []
    for entity in scene.entities:
        if isinstance(entity, PointMass):
            entities_copy.append(
                PointMass(
                    entity_id=entity.entity_id,
                    mass=entity.mass,
                    position=np.array(entity.position, dtype=float),
                    velocity=np.array(entity.velocity, dtype=float),
                )
            )
        elif isinstance(entity, RigidBody):
            components_copy = [
                RigidBodyComponent(
                    component_id=component.component_id,
                    mass=component.mass,
                    position_body=np.array(component.position_body, dtype=float),
                )
                for component in entity.components
            ]
            entities_copy.append(
                RigidBody(
                    entity_id=entity.entity_id,
                    components=components_copy,
                    com_position=np.array(entity.com_position, dtype=float),
                    com_velocity=np.array(entity.com_velocity, dtype=float),
                    orientation=np.array(entity.orientation, dtype=float),
                    omega_world=np.array(entity.omega_world, dtype=float),
                    mesh=None
                    if entity.mesh is None
                    else MeshMetadata(
                        path=str(entity.mesh.path),
                        path_is_absolute=bool(entity.mesh.path_is_absolute),
                        scale=np.array(entity.mesh.scale, dtype=float),
                        offset_body=np.array(entity.mesh.offset_body, dtype=float),
                        rotation_body=np.array(entity.mesh.rotation_body, dtype=float),
                    ),
                )
            )
        else:
            raise TypeError(f"Unsupported entity type: {type(entity)}")
    ui_state = dict(scene.ui_state or {})
    return SceneData(
        entities=entities_copy,
        scenario_id=scene.scenario_id,
        ui_state=ui_state,
        scene_version=scene.scene_version,
    )
