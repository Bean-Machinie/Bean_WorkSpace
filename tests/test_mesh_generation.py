import numpy as np

from space_dynamics_workbench.app.mesh_generation import generate_mass_points_from_vertices


def test_generate_mass_points_bbox() -> None:
    vertices = np.array(
        [
            [-2.0, -1.0, -0.5],
            [-2.0, -1.0, 0.5],
            [-2.0, 3.0, -0.5],
            [-2.0, 3.0, 0.5],
            [4.0, -1.0, -0.5],
            [4.0, -1.0, 0.5],
            [4.0, 3.0, -0.5],
            [4.0, 3.0, 0.5],
        ],
        dtype=float,
    )
    points = generate_mass_points_from_vertices(vertices, max_points=5)
    assert len(points) <= 5
    assert len(points) >= 1

    v_min = vertices.min(axis=0)
    v_max = vertices.max(axis=0)
    for point in points:
        assert point.shape == (3,)
        assert np.all(point >= v_min - 1e-9)
        assert np.all(point <= v_max + 1e-9)


def test_generate_mass_points_deterministic() -> None:
    vertices = np.array(
        [
            [-1.0, -1.0, -1.0],
            [-1.0, 1.0, 1.0],
            [1.0, -1.0, 1.0],
            [1.0, 1.0, -1.0],
        ],
        dtype=float,
    )
    points_a = generate_mass_points_from_vertices(vertices, max_points=5)
    points_b = generate_mass_points_from_vertices(vertices, max_points=5)
    assert len(points_a) == len(points_b)
    for a, b in zip(points_a, points_b):
        np.testing.assert_allclose(a, b)
