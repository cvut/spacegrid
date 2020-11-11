import os

import numpy
import pytest

from test_interface import empty_grid

from spacegrid import escape_routes

VOID = 0
NODE = 1
STATION = 2
SINGULARITY = 3

BIG_ENOUGH_NUMBER = int(os.getenv('BIG_ENOUGH_NUMBER', 2048))


def big_random_grid():
    line = empty_grid(BIG_ENOUGH_NUMBER**2, 1)[:, 0]
    line[:BIG_ENOUGH_NUMBER**2//4] = STATION
    numpy.random.shuffle(line)
    line[:BIG_ENOUGH_NUMBER**2//4] = SINGULARITY
    numpy.random.shuffle(line)
    line[:BIG_ENOUGH_NUMBER**2//4] = NODE
    line[-1] = STATION
    line[-2] = SINGULARITY
    numpy.random.shuffle(line)
    return line.reshape(BIG_ENOUGH_NUMBER, BIG_ENOUGH_NUMBER)


# we do this globally to avoid interfering with the times
BIG_RANDOM_GRID = big_random_grid()


def escape_routes_(grid):
    print('GRID:')
    print(grid)
    print()
    result = escape_routes(grid)
    print('DISTANCES:')
    print(result.distances)
    print()
    print('DIRECTIONS:')
    print(result.directions)
    print()
    print('SAFE FACTOR:', result.safe_factor)
    return result


def numpyfunc(*args, **kwargs):
    """A decorator to use instead of numpy.frompyfunc"""
    def _decorator(func):
        return numpy.frompyfunc(func, *args, **kwargs)
    return _decorator


@numpyfunc(2, 1)
def _ascii_within(direction_byte, reference_string):
    return direction_byte in reference_string.encode('ascii')


def ascii_within(directions_array, references_array):
    """
    For each element of directions_array, check whether it is contained
    in the co-located element in references_array.
    Elements of directions_array are expected to be 1-bytes ASCII chars,
    elements of references_array are arbitrarily long ASCII strings.
    The references_array consists of "unicode" strings (with ASCII chars only)
    for easier write-up.
    """
    return numpy.all(_ascii_within(directions_array, references_array))


def test_empty_grid():
    grid = empty_grid(4, 4)
    r = escape_routes_(grid)

    assert numpy.all(r.distances == -1)
    assert numpy.all(r.directions == b' ')
    assert r.safe_factor == pytest.approx(0.0)


def test_all_stations():
    height, width = 8, 5
    grid = empty_grid(height, width)
    grid[:] = STATION
    r = escape_routes_(grid)

    assert numpy.all(r.distances == 0)
    assert numpy.all(r.directions == b'+')
    assert r.safe_factor == pytest.approx(1.0)


def test_no_station():
    height, width = 8, 5
    grid = empty_grid(height, width)
    grid[0, 0] = NODE
    grid[-1, -1] = NODE
    grid[-1, 0] = NODE
    grid[0, -1] = NODE
    r = escape_routes_(grid)

    assert numpy.all(r.distances == -1)
    assert numpy.all(r.directions == b' ')
    assert r.safe_factor == pytest.approx(0.0)


def test_trivial_route_readme_example():
    grid = empty_grid(1, 2)
    grid[0, 1] = STATION
    r = escape_routes_(grid)

    assert numpy.array_equal(r.distances, [[1, 0]])
    assert numpy.array_equal(r.directions, [[b'>', b'+']])
    assert r.safe_factor == pytest.approx(1.0)


def test_trivial_route_long_grid_readme_example():
    grid = empty_grid(1, 1024)
    grid[0, 1023] = STATION
    r = escape_routes_(grid)

    assert numpy.array_equal(r.distances, [[1] * 1023 + [0]])
    assert numpy.array_equal(r.directions, [[b'>'] * 1023 + [b'+']])
    assert r.safe_factor == pytest.approx(1.0)


def test_too_many_nodes_readme_example():
    grid = empty_grid(1, 1024)
    grid[:] = NODE
    grid[0, 1023] = STATION
    r = escape_routes_(grid)

    assert numpy.array_equal(r.distances, [numpy.arange(0, 1024)[::-1]])
    assert numpy.array_equal(r.directions, [[b'>'] * 1023 + [b'+']])
    assert r.safe_factor == pytest.approx(1.0)


def test_unreachble_corner_readme_example():
    grid = empty_grid(2, 2)
    grid[1, 1] = STATION
    r = escape_routes_(grid)
    assert numpy.array_equal(r.distances, [[-1, 1], [1, 0]])
    assert numpy.array_equal(r.directions, [[b' ', b'v'], [b'>', b'+']])
    assert r.safe_factor == pytest.approx(0.75)


def test_corner_node_readme_example():
    grid = empty_grid(2, 2)
    grid[1, 1] = STATION
    grid[0, 1] = NODE
    r = escape_routes_(grid)

    assert numpy.array_equal(r.distances, [[2, 1], [1, 0]])
    assert ascii_within(r.directions, [['v>', 'v'], ['>', '+']])
    assert r.safe_factor == pytest.approx(1.0)


def test_far_station_might_be_better_readme_example():
    grid = empty_grid(1024, 2)
    grid[0, 1] = NODE
    grid[1, 1] = STATION
    grid[1023, 0] = STATION
    r = escape_routes_(grid)

    assert numpy.all(r.distances[:-1, 0] == 1)
    assert r.distances[-1, 0] == 0
    assert numpy.all(r.distances[0, 1] == 1)
    assert numpy.all(r.distances[1, 1] == 0)
    assert numpy.all(r.distances[2:, 1] == 1)

    assert r.directions[0, 0] == b'v'
    assert r.directions[1, 0] in (b'>', b'v')
    assert numpy.all(r.directions[2:-1, 0] == b'v')
    assert r.directions[1023, 0] == b'+'

    assert r.directions[0, 1] == b'v'
    assert r.directions[1, 1] == b'+'
    assert numpy.all(r.directions[2:-1, 1] == b'^')
    assert r.directions[1023, 1] in (b'<', b'^')

    assert r.safe_factor == pytest.approx(1.0)


def test_no_beam_through_singularity_simple():
    grid = empty_grid(1, 5)
    grid[0, 0] = STATION
    grid[0, 3] = SINGULARITY
    r = escape_routes_(grid)

    assert numpy.array_equal(r.distances, [[0, 1, 1, -1, -1]])
    assert numpy.array_equal(r.directions, [[b'+', b'<', b'<', b' ', b' ']])
    assert r.safe_factor == pytest.approx(0.6)


def test_no_beam_through_singularity_readme_example():
    grid = empty_grid(1, 13)
    grid[0, 0] = STATION
    grid[0, 11] = SINGULARITY
    grid[0, 12] = STATION
    r = escape_routes_(grid)

    assert r.distances[0, 0] == 0
    assert numpy.all(r.distances[0, 1:11] == 1)
    assert r.distances[0, 11] == -1
    assert r.distances[0, 12] == 0

    assert r.directions[0, 0] == b'+'
    assert numpy.all(r.directions[0, 1:11] == b'<')
    assert r.directions[0, 11] == b' '
    assert r.directions[0, 12] == b'+'

    assert r.safe_factor == pytest.approx(12/13)


def test_beam_around_singularity():
    grid = empty_grid(3, 3)
    grid[2, 0] = NODE
    grid[2, 2] = NODE
    grid[0, 2] = STATION
    grid[0, 1] = SINGULARITY
    r = escape_routes_(grid)

    assert numpy.array_equal(r.distances, [[3, -1, 0], [3, -1, 1], [2, 2, 1]])
    assert numpy.array_equal(r.directions, [[b'v', b' ', b'+'],
                                            [b'v', b' ', b'^'],
                                            [b'>', b'>', b'^']])
    assert r.safe_factor == pytest.approx(7/9)


def test_only_useless_nodes():
    grid = empty_grid(6, 6)
    grid[numpy.diag_indices(6)] = NODE
    grid[3, 3] = STATION
    grid[2, 3] = SINGULARITY
    grid[4, 3] = SINGULARITY
    grid[3, 2] = SINGULARITY
    grid[3, 4] = SINGULARITY
    r = escape_routes_(grid)

    assert r.distances[3, 3] == 0
    assert numpy.sum(r.distances) == -6*6+1
    assert r.directions[3, 3] == b'+'
    assert numpy.count_nonzero(r.directions == b' ') == 6*6-1
    assert r.safe_factor == pytest.approx(1/6/6)


def test_node_accessible_from_multiple_stations():
    grid = empty_grid(4, 2)
    grid[0, 1] = STATION
    grid[1, 1] = NODE
    grid[3, 0] = STATION

    r1 = escape_routes_(grid)
    assert r1.distances[3, 1] == 1

    # no matter the orientation:
    r2 = escape_routes_(grid.T)
    assert r2.distances[1, 3] == 1


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

    expected_distances = [
        [2, 1,  2,  2, 2,  2,  3, 2,  1,  1, 2],
        [2, 1,  5,  5, 4,  3, -1, 1,  0,  1, 1],
        [1, 0, -1, -1, 4,  4,  1, 2,  1,  1, 2],
        [2, 1,  4,  4, 4,  3,  1, 2, -1,  0, 1],
        [2, 1, -1, -1, 4, -1,  0, 1,  1,  1, 1],
        [2, 1,  3,  3, 3,  4,  1, 3,  4, -1, 2],
        [2, 1,  2,  2, 2,  2,  1, 2,  2,  2, 2],
    ]
    assert numpy.array_equal(r.distances, expected_distances)

    # WARNING: This array might not be 100% complete
    # If your direction is not listed, verify manually and open a PR
    possible_directions = [
        ['>v', 'v', '<', '<', '<',  '<', '<>', 'v', 'v', 'v',  'v' ],
        ['v',  'v', '>', '>', '>v', '^', ' ',  '>', '+', '<v', '<' ],
        ['>',  '+', ' ', ' ', 'v',  '^v', 'v', '^', '^', 'v',  'v' ],
        ['^',  '^', '>', '>', 'v>', '>', 'v',  '^', ' ', '+',  '<' ],
        ['^',  '^', ' ', ' ', 'v',  ' ', '+',  '<', '<', '^<', '<' ],
        ['^',  '^', '<', '<', '<',  '<', '^',  '^', '<', ' ',  '^' ],
        ['>',  '^', '>', '>', '>',  '>', '^',  '<', '<', '<',  '^<'],
    ]
    assert ascii_within(r.directions, possible_directions)

    reachable = numpy.count_nonzero(numpy.array(expected_distances) >= 0)
    assert r.safe_factor == reachable / grid.size


@pytest.mark.timeout(20)
def test_large_grid_slow():
    grid = BIG_RANDOM_GRID
    count = numpy.count_nonzero

    for i in range(10):
        print(i)
        r = escape_routes_(grid)

        assert count(r.distances == 0) == count(r.directions == b'+') == count(grid == 2)
        assert count(r.distances == -1) >= count(grid == 3)
        assert 0.0 < r.safe_factor < 1.0
