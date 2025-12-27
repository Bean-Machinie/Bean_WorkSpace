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


def test_rigid_body_rotation_invariants() -> None:
    components = [
        RigidBodyComponent(component_id="A", mass=1.2, position_body=np.array([1.4, -0.2, 0.3])),
        RigidBodyComponent(component_id="B", mass=0.7, position_body=np.array([-0.6, 0.9, -0.1])),
        RigidBodyComponent(component_id="C", mass=1.8, position_body=np.array([0.2, -0.7, 0.6])),
        RigidBodyComponent(component_id="D", mass=0.9, position_body=np.array([-0.3, 0.1, -0.8])),
    ]
    body = RigidBody(
        entity_id="RB-ROT",
        components=components,
        com_position=np.array([0.0, 0.0, 0.0]),
        com_velocity=np.array([0.0, 0.0, 0.0]),
        orientation=np.array([1.0, 0.0, 0.0, 0.0]),
        omega_world=np.array([0.0, 0.0, 1.5]),
    )

    dt = 0.01
    steps = 100
    positions_initial = body.component_positions_world()
    r_norm0 = [np.linalg.norm(pos - body.com_position) for pos in positions_initial]

    for _ in range(steps):
        body.step_kinematic(dt)
        positions = body.component_positions_world()
        velocities = body.component_velocities_world()

        for idx, (pos, vel) in enumerate(zip(positions, velocities)):
            r_rel = pos - body.com_position
            v_rel = vel - body.com_velocity
            r_norm = float(np.linalg.norm(r_rel))
            v_norm = float(np.linalg.norm(v_rel))

            assert abs(r_norm - r_norm0[idx]) <= 1e-6 * (r_norm0[idx] + 1.0)
            dot = float(np.dot(v_rel, r_rel))
            assert abs(dot) <= 1e-6 * (v_norm * r_norm + 1.0)
            expected = float(np.linalg.norm(np.cross(body.omega_world, r_rel)))
            assert abs(v_norm - expected) <= 1e-6 * (expected + 1.0)
