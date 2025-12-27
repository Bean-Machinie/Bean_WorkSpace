import numpy as np

from space_dynamics_workbench.core.model import MeshMetadata, PointMass, RigidBody, RigidBodyComponent
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


def test_scene_roundtrip_rigid_body_mesh() -> None:
    components = [
        RigidBodyComponent(component_id="A", mass=2.0, position_body=np.array([1.0, 0.0, 0.0])),
        RigidBodyComponent(component_id="B", mass=1.0, position_body=np.array([-0.5, 0.0, 0.2])),
    ]
    mesh = MeshMetadata(
        path="assets/models/test.glb",
        path_is_absolute=False,
        scale=np.array([1.0, 2.0, 1.5], dtype=float),
        offset_body=np.array([0.1, -0.2, 0.3], dtype=float),
        rotation_body=np.array([1.0, 0.0, 0.0, 0.0], dtype=float),
    )
    rigid_body = RigidBody(
        entity_id="RB-1",
        components=components,
        com_position=np.array([0.0, 0.0, 0.0]),
        com_velocity=np.array([0.0, 0.0, 0.0]),
        orientation=np.array([1.0, 0.0, 0.0, 0.0]),
        omega_world=np.array([0.0, 0.0, 0.0]),
        mesh=mesh,
    )
    scene_json = serialize_scene(SceneData(entities=[rigid_body], scenario_id="mesh-test", ui_state={}))
    restored = deserialize_scene(scene_json)

    assert len(restored.entities) == 1
    loaded = restored.entities[0]
    assert isinstance(loaded, RigidBody)
    assert loaded.mesh is not None
    assert loaded.mesh.path == "assets/models/test.glb"
    assert loaded.mesh.path_is_absolute is False
    np.testing.assert_allclose(loaded.mesh.scale, np.array([1.0, 2.0, 1.5]))
    np.testing.assert_allclose(loaded.mesh.offset_body, np.array([0.1, -0.2, 0.3]))
    np.testing.assert_allclose(loaded.mesh.rotation_body, np.array([1.0, 0.0, 0.0, 0.0]))
