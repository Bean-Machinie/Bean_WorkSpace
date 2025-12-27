from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List

import numpy as np

from ..core.model import PointMass

SCENE_VERSION = 1


@dataclass
class SceneData:
    entities: List[PointMass]
    scenario_id: str | None = None
    ui_state: Dict[str, Any] | None = None
    scene_version: int = SCENE_VERSION


def scene_to_dict(scene: SceneData) -> Dict[str, Any]:
    return {
        "scene_version": scene.scene_version,
        "scenario_id": scene.scenario_id,
        "entities": [
            {
                "entity_id": entity.entity_id,
                "mass": entity.mass,
                "position": entity.position.tolist(),
                "velocity": entity.velocity.tolist(),
            }
            for entity in scene.entities
        ],
        "ui_state": scene.ui_state or {},
    }


def scene_from_dict(payload: Dict[str, Any]) -> SceneData:
    version = int(payload.get("scene_version", 0))
    if version != SCENE_VERSION:
        raise ValueError(f"Unsupported scene version: {version}")
    entities = []
    for entity_data in payload.get("entities", []):
        entities.append(
            PointMass(
                entity_id=str(entity_data["entity_id"]),
                mass=float(entity_data["mass"]),
                position=np.array(entity_data["position"], dtype=float),
                velocity=np.array(entity_data["velocity"], dtype=float),
            )
        )
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


def capture_scene(entities: Iterable[PointMass], scenario_id: str | None = None) -> SceneData:
    return clone_scene(SceneData(entities=list(entities), scenario_id=scenario_id, ui_state={}))


def clone_scene(scene: SceneData) -> SceneData:
    entities_copy = [
        PointMass(
            entity_id=entity.entity_id,
            mass=entity.mass,
            position=np.array(entity.position, dtype=float),
            velocity=np.array(entity.velocity, dtype=float),
        )
        for entity in scene.entities
    ]
    ui_state = dict(scene.ui_state or {})
    return SceneData(
        entities=entities_copy,
        scenario_id=scene.scenario_id,
        ui_state=ui_state,
        scene_version=scene.scene_version,
    )
