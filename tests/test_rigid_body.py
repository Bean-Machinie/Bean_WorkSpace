import numpy as np

from space_dynamics_workbench.core.model import RigidBody, RigidBodyComponent, iter_mass_points
from space_dynamics_workbench.core.physics import center_of_mass


def test_rigid_body_world_positions() -> None:
    components = [
        RigidBodyComponent(component_id="A", mass=2.0, position_body=np.array([1.0, 0.0, 0.0])),
        RigidBodyComponent(component_id="B", mass=2.0, position_body=np.array([-1.0, 0.0, 0.0])),
    ]
    angle = np.deg2rad(90.0)
    orientation = np.array([np.cos(angle / 2.0), 0.0, 0.0, np.sin(angle / 2.0)], dtype=float)
    body = RigidBody(
        entity_id="RB-1",
        components=components,
        com_position=np.array([2.0, 3.0, 0.0]),
        com_velocity=np.zeros(3),
        orientation=orientation,
        omega_world=np.zeros(3),
    )
    positions = body.component_positions_world()
    np.testing.assert_allclose(positions[0], np.array([2.0, 4.0, 0.0]))
    np.testing.assert_allclose(positions[1], np.array([2.0, 2.0, 0.0]))


def test_rigid_body_com_invariant() -> None:
    components = [
        RigidBodyComponent(component_id="A", mass=1.0, position_body=np.array([2.0, 0.0, 0.0])),
        RigidBodyComponent(component_id="B", mass=2.0, position_body=np.array([-1.0, 0.0, 0.0])),
        RigidBodyComponent(component_id="C", mass=1.0, position_body=np.array([0.0, 1.0, 0.0])),
        RigidBodyComponent(component_id="D", mass=1.0, position_body=np.array([0.0, -1.0, 0.0])),
    ]
    body = RigidBody(
        entity_id="RB-2",
        components=components,
        com_position=np.array([-1.0, 0.5, 0.0]),
        com_velocity=np.zeros(3),
        orientation=np.array([1.0, 0.0, 0.0, 0.0]),
        omega_world=np.zeros(3),
    )
    invariant = body.invariant_position_sum()
    np.testing.assert_allclose(invariant, np.zeros(3), atol=1e-9)

    mass_points = iter_mass_points([body])
    com = center_of_mass(mass_points)
    np.testing.assert_allclose(com, body.com_position)


def test_rigid_body_distances_constant() -> None:
    components = [
        RigidBodyComponent(component_id="A", mass=1.0, position_body=np.array([1.2, 0.0, 0.0])),
        RigidBodyComponent(component_id="B", mass=1.5, position_body=np.array([-0.6, 0.8, 0.0])),
        RigidBodyComponent(component_id="C", mass=0.7, position_body=np.array([-0.4, -0.7, 0.0])),
    ]
    body = RigidBody(
        entity_id="RB-3",
        components=components,
        com_position=np.array([0.0, 0.0, 0.0]),
        com_velocity=np.array([0.1, -0.05, 0.0]),
        orientation=np.array([1.0, 0.0, 0.0, 0.0]),
        omega_world=np.array([0.0, 0.0, 1.2]),
    )
    initial_distances = body.pairwise_distances_world()
    for _ in range(50):
        body.step_kinematic(0.02)
    final_distances = body.pairwise_distances_world()
    np.testing.assert_allclose(initial_distances, final_distances, atol=1e-6)
