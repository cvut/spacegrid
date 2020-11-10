import queue
import numpy

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

        distances_dtype = cls.smallest_signed_dtype(size)
        directions_dtype = numpy.dtype(('a', 1))  # one ASCII character

        distances = numpy.full(shape, -1, dtype=distances_dtype)
        directions = numpy.full(shape, b' ', dtype=directions_dtype)

        jobs = queue.Queue()

        stations = numpy.argwhere(grid == 2)
        for row, column in stations:
            directions[row, column] = b'+'
            distances[row, column] = 0
            jobs.put(((row, column), 1))  # next distance

        while not jobs.empty():
            loc_, dist_ = jobs.get()
            for direction in 'down', 'right', 'up', 'left':  # any order
                # reset the values for each direction
                loc, dist = loc_, dist_
                while True:
                    # walk that direction, but stop at boundary
                    # this ugly if could be delegated to functions in dict :/
                    if direction == 'down':
                        loc = loc[0]+1, loc[1]
                        reverse = b'^'
                        if loc[0] == shape[0]:
                            break
                    elif direction == 'right':
                        loc = loc[0], loc[1]+1
                        reverse = b'<'
                        if loc[1] == shape[1]:
                            break
                    elif direction == 'up':
                        loc = loc[0]-1, loc[1]
                        reverse = b'v'
                        if loc[0] < 0:
                            break
                    else:  # left
                        loc = loc[0], loc[1]-1
                        reverse = b'>'
                        if loc[1] < 0:
                            break
                    # singularities cannot be beamed trough
                    # no point of checking past a safe station
                    if grid[loc] > 1:
                        break
                    # been there better? skip but check further if not a node
                    if 0 <= distances[loc] <= dist:
                        if grid[loc] == 1:
                            break
                        continue
                    distances[loc] = dist
                    directions[loc] = reverse
                    # if it's a transport node,
                    # we need to increase our distance and
                    # schedule a job
                    if grid[loc] == 1:
                        dist += 1
                        jobs.put((loc, dist))

        return distances, directions

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
