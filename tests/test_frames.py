import numpy as np

from space_dynamics_workbench.core.model import PointMass
from space_dynamics_workbench.core.physics import FrameChoice, compute_frame_vectors


def test_frame_vectors_invariants() -> None:
    entities = [
        PointMass(entity_id="A", mass=2.0, position=np.array([1.0, 0.0]), velocity=np.array([0.5, -0.2])),
        PointMass(entity_id="B", mass=3.0, position=np.array([-1.0, 2.0]), velocity=np.array([-0.1, 0.3])),
        PointMass(entity_id="C", mass=5.0, position=np.array([0.0, -1.5]), velocity=np.array([0.0, -0.1])),
    ]

    world = compute_frame_vectors(entities, FrameChoice.WORLD)
    com = compute_frame_vectors(entities, FrameChoice.COM)

    r_cp_world = [seg.end - seg.start for seg in world.r_cp_segments]
    r_cp_com = [seg.end - seg.start for seg in com.r_cp_segments]

    for v_world, v_com in zip(r_cp_world, r_cp_com):
        np.testing.assert_allclose(v_world, v_com)

    weighted_sum = np.sum([m * r for m, r in zip(world.masses, r_cp_world)], axis=0)
    assert np.linalg.norm(weighted_sum) < 1e-9
