import numpy
import pytest

from test_interface import empty_grid
from test_results import escape_routes_
from test_results import VOID, NODE, STATION, SINGULARITY, BIG_RANDOM_GRID


def test_all_stations():
    height, width = 8, 5
    grid = empty_grid(height, width)
    grid[:] = STATION
    r = escape_routes_(grid)

    for row in range(height):
        for column in range(width):
            route = list(r.route(row, column))
            assert route == []


def test_no_station():
    height, width = 8, 5
    grid = empty_grid(height, width)
    grid[0, 0] = NODE
    grid[-1, -1] = NODE
    grid[-1, 0] = NODE
    grid[0, -1] = NODE
    r = escape_routes_(grid)

    for row in range(height):
        for column in range(width):
            with pytest.raises(ValueError):
                r.route(row, column)


def test_trivial_route_readme_example():
    grid = empty_grid(1, 2)
    grid[0, 1] = STATION
    r = escape_routes_(grid)

    assert list(r.route(0, 0)) == [(0, 1)]


def test_trivial_route_long_grid_readme_example():
    grid = empty_grid(1, 1024)
    grid[0, 1023] = STATION
    r = escape_routes_(grid)

    for column in range(1023):
        assert list(r.route(0, column)) == [(0, 1023)]


def test_too_many_nodes_readme_example():
    grid = empty_grid(1, 1024)
    grid[:] = NODE
    grid[0, 1023] = STATION
    r = escape_routes_(grid)

    for column in range(1023):
        route = list(r.route(0, column))
        assert len(route) == 1023-column
        assert route == [(0, c) for c in range(column+1, 1024)]


def test_unreachble_corner_readme_example():
    grid = empty_grid(2, 2)
    grid[1, 1] = STATION
    r = escape_routes_(grid)

    with pytest.raises(ValueError):
        r.route(0, 0)


def test_corner_node_readme_example():
    grid = empty_grid(2, 2)
    grid[1, 1] = STATION
    grid[0, 1] = NODE
    r = escape_routes_(grid)

    assert list(r.route(0, 0)) == [(0, 1), (1, 1)]


def test_far_station_might_be_better_readme_example():
    grid = empty_grid(1024, 2)
    grid[0, 1] = NODE
    grid[1, 1] = STATION
    grid[1023, 0] = STATION
    r = escape_routes_(grid)

    assert list(r.route(0, 0)) == [(1023, 0)]


def test_no_beam_through_singularity_simple():
    grid = empty_grid(1, 5)
    grid[0, 0] = STATION
    grid[0, 3] = SINGULARITY
    r = escape_routes_(grid)

    assert list(r.route(0, 2)) == [(0, 0)]
    with pytest.raises(ValueError):
        r.route(0, 3)
    with pytest.raises(ValueError):
        r.route(0, 4)


def test_no_beam_through_singularity_readme_example():
    grid = empty_grid(1, 13)
    grid[0, 0] = STATION
    grid[0, 11] = SINGULARITY
    grid[0, 12] = STATION
    r = escape_routes_(grid)

    assert list(r.route(0, 10)) == [(0, 0)]


def test_beam_around_singularity():
    grid = empty_grid(3, 3)
    grid[2, 0] = NODE
    grid[2, 2] = NODE
    grid[0, 2] = STATION
    grid[0, 1] = SINGULARITY
    r = escape_routes_(grid)

    assert list(r.route(0, 0)) == [(2, 0), (2, 2), (0, 2)]


def test_only_useless_nodes():
    grid = empty_grid(6, 6)
    grid[numpy.diag_indices(6)] = NODE
    grid[3, 3] = STATION
    grid[2, 3] = SINGULARITY
    grid[4, 3] = SINGULARITY
    grid[3, 2] = SINGULARITY
    grid[3, 4] = SINGULARITY
    r = escape_routes_(grid)

    for row in range(6):
        for column in range(6):
            if not (row == column == 3):
                with pytest.raises(ValueError):
                    r.route(row, column)
    assert list(r.route(3, 3)) == []


def test_node_accessible_from_multiple_stations():
    grid = empty_grid(4, 2)
    grid[0, 1] = STATION
    grid[1, 1] = NODE
    grid[3, 0] = STATION

    r1 = escape_routes_(grid)
    assert list(r1.route(3, 1)) == [(3, 0)]

    # no matter the orientation:
    r2 = escape_routes_(grid.T)
    assert list(r2.route(1, 3)) == [(0, 3)]


def test_readme_image():
    grid = empty_grid(7, 11)
    grid[:] = [
        [0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1],
        [0, 0, 0, 0, 1, 1, 3, 1, 2, 0, 0],
        [1, 2, 3, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 1, 0, 1, 3, 2, 1],
        [0, 0, 0, 3, 0, 3, 2, 0, 0, 0, 0],
        [1, 0, 0, 0, 1, 0, 0, 0, 0, 3, 0],
        [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
    ]
    r = escape_routes_(grid)

    spaceship_route = list(r.route(2, 4))
    assert len(spaceship_route) == 4
    assert spaceship_route == [(5, 4), (5, 0), (2, 0), (2, 1)]

    ambiguous_route = list(r.route(1, 4))
    first_direction = r.directions[1, 4]
    assert first_direction in (b'>', b'v')
    if first_direction == b'>':
        assert ambiguous_route == [(1, 5), (0, 5), (0, 1), (2, 1)]
    elif first_direction == b'v':
        assert ambiguous_route == [(5, 4), (5, 0), (2, 0), (2, 1)]


@pytest.mark.timeout(20)
def test_large_grid_slow():
    grid = BIG_RANDOM_GRID
    for i in range(10):
        print(i)
        r = escape_routes_(grid)

        station = numpy.argwhere(grid == 2)[0]
        assert list(r.route(*station)) == []

        singularity = numpy.argwhere(grid == 3)[0]
        with pytest.raises(ValueError):
            r.route(*singularity)

        row, col = numpy.unravel_index(numpy.argmax(r.distances), grid.shape)
        for i in range(50):
            assert len(list(r.route(row, col))) == r.distances[row, col]
