import numpy as np

from space_dynamics_workbench.core.model import PointMass
from space_dynamics_workbench.core.physics import invariant_position_sum, invariant_velocity_sum


def test_com_invariants_hold() -> None:
    entities = [
        PointMass(entity_id="A", mass=2.0, position=np.array([1.0, 0.0]), velocity=np.array([0.5, -0.2])),
        PointMass(entity_id="B", mass=3.0, position=np.array([-1.0, 2.0]), velocity=np.array([-0.1, 0.3])),
        PointMass(entity_id="C", mass=5.0, position=np.array([0.0, -1.5]), velocity=np.array([0.0, -0.1])),
    ]

    pos_sum = invariant_position_sum(entities)
    vel_sum = invariant_velocity_sum(entities)

    assert np.linalg.norm(pos_sum) < 1e-9
    assert np.linalg.norm(vel_sum) < 1e-9
