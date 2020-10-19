import numpy
import pytest

from spacegrid import escape_routes


def empty_grid(*shape):
    return numpy.zeros(shape, dtype=numpy.uint8)


def test_disatnces_shape_type():
    grid = empty_grid(3, 7)
    result = escape_routes(grid)
    assert result.distances.shape == grid.shape
    assert isinstance(result.distances[0, 0], numpy.signedinteger)


def test_directions_shape_type():
    grid = empty_grid(4, 15)
    result = escape_routes(grid)
    assert result.directions.shape == grid.shape
    assert isinstance(result.directions[0, 0], bytes)
    assert len(result.directions[0, 0]) == 1


def test_safe_factor_type_size():
    grid = empty_grid(4, 15)
    result = escape_routes(grid)
    assert isinstance(result.safe_factor, float)
    assert 0.0 <= result.safe_factor <= 1.0


def test_safe_factor_type_size_empty():
    grid = empty_grid(0, 0)
    result = escape_routes(grid)
    assert numpy.isnan(result.safe_factor)  # `nan == nan` is False


def test_route_empty_grid():
    grid = empty_grid(2, 2)
    result = escape_routes(grid)
    for row in 0, 1:
        for column in 0, 1:
            with pytest.raises(ValueError):
                result.route(row, column)


def test_route_outside():
    grid = empty_grid(2, 2)
    result = escape_routes(grid)
    with pytest.raises(IndexError):
        result.route(100, 100)


def test_route_nonindex():
    grid = empty_grid(2, 2)
    result = escape_routes(grid)
    with pytest.raises(IndexError):
        result.route('spam', 'eggs')


def test_route_no_argument():
    grid = empty_grid(2, 2)
    result = escape_routes(grid)
    with pytest.raises(TypeError):
        result.route()


def test_route_more_arguments():
    grid = empty_grid(2, 2)
    result = escape_routes(grid)
    with pytest.raises(TypeError):
        result.route(1, 1, 1)


def test_route_nonempty_grid():
    grid = empty_grid(2, 2)
    grid[0, 0] = 2
    result = escape_routes(grid)
    route = result.route(1, 0)  # this route exists
    route = list(route)  # asserts this can be converted
    assert isinstance(route[0], tuple)
    assert len(route[0]) == 2
    assert isinstance(route[0][0], int)
    assert isinstance(route[0][1], int)


def test_no_args_not_possible():
    with pytest.raises(TypeError):
        escape_routes()


def test_multiple_args_not_possible():
    with pytest.raises(TypeError):
        escape_routes(numpy.zeros((1, 1), dtype=int), None)


def test_1D_array_not_possible():
    with pytest.raises(TypeError):
        escape_routes(numpy.zeros((1,), dtype=int))


def test_3D_array_not_possible():
    with pytest.raises(TypeError):
        escape_routes(numpy.zeros((1, 1, 1), dtype=int))


def test_float_array_not_possible():
    with pytest.raises(TypeError):
        escape_routes(numpy.zeros((1, 1), dtype=float))


def test_nonarray_not_possible():
    with pytest.raises(TypeError):
        escape_routes([0, 0, 0])
