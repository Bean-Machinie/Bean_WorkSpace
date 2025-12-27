import numpy as np

from space_dynamics_workbench.core.model import PointMass
from space_dynamics_workbench.io.scene_format import SceneData, deserialize_scene, serialize_scene


def test_scene_roundtrip() -> None:
    entities = [
        PointMass(entity_id="P1", mass=1.0, position=np.array([0.0, 0.0]), velocity=np.array([0.1, 0.2])),
        PointMass(entity_id="P2", mass=4.0, position=np.array([2.0, -1.0]), velocity=np.array([-0.2, 0.0])),
    ]

    scene_json = serialize_scene(
        scene=SceneData(entities=entities, scenario_id="test", ui_state={})
    )
    restored = deserialize_scene(scene_json)

    assert restored.scene_version == 1
    assert restored.scenario_id == "test"
    assert len(restored.entities) == len(entities)

    for original, loaded in zip(entities, restored.entities):
        assert original.entity_id == loaded.entity_id
        assert original.mass == loaded.mass
        np.testing.assert_allclose(original.position, loaded.position)
        np.testing.assert_allclose(original.velocity, loaded.velocity)
