import queue
import numpy

from . import speedup

__all__ = ['escape_routes']


class _EscapeRoutes:
    """
    Internal class to calculate and represent escape_routes() results.
    A reference to the given grid is kept and used in route().
    """
    def __init__(self, grid):
        self.grid = grid
        self.distances, self.directions = self.distance_directions(grid)

    @staticmethod
    def smallest_signed_dtype(value):
        """
        Helper function. Gives a smallest possible signed numpy dtype to handle
        distances of given value.
        """
        for dtype in numpy.int8, numpy.int16, numpy.int32, numpy.int64:
            if dtype(value) == value:
                return dtype
        raise OverflowError(f'grid of size {value} is too big')

    @classmethod
    def distance_directions(cls, grid):
        try:
            shape = grid.shape
            dtype = grid.dtype
            size = grid.size
            ndim = grid.ndim
        except AttributeError:
            raise TypeError('grid does not quack like a numpy array')
        if ndim != 2:
            raise TypeError('grid does not quack like a 2D-numpy array')
        if not numpy.issubdtype(dtype, numpy.integer):
            raise TypeError('grid does not quack like an integer array')

        return speedup.flood(grid.astype(numpy.uint8, copy=False))

    @property
    def safe_factor(self):
        reachable = numpy.count_nonzero(self.distances >= 0)
        try:
            return reachable / self.distances.size
        except ZeroDivisionError:
            return float('nan')

    def _route_generator(self, row, column):
        """
        A generator implementation of .route() without the exceptions.
        """
        direction = self.directions[row, column]
        while not direction == b'+':
            if direction == b'v':
                row += 1
            elif direction == b'>':
                column += 1
            elif direction == b'^':
                row -= 1
            elif direction == b'<':
                column -= 1
            if self.grid[row, column] > 0:
                direction = self.directions[row, column]
                yield row, column

    def route(self, row, column):
        """
        For given coordinates of a space ship,
        returns a sequence of coordinates of the best route to a safe station.
        Dos not include coordinates of the space ship.
        Includes coordinates of the safe station,
        unless the space ship is on a safe station,
        in that case it returns an empty sequence.

        Raises ValueError when not possible.
        Raises IndexError when given coordinates are out of bounds.

        Assumes self.directions are valid. If self.directions is tempered with,
        this may give incorrect results or even enter an endless loop.

        Not a generator so we can raise as soon as called.
        """
        if self.directions[row, column] == b' ':
            raise ValueError('No route to host...ehm...safe station')
        return self._route_generator(row, column)


def escape_routes(grid):
    """
    This function calculates the escape routes from the 2D space grid.

    For details, see the assignment at github.com/cvut/spacegrid
    """
    return _EscapeRoutes(grid)
