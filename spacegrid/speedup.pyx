import numpy

cimport cython
cimport numpy
from cpython.mem cimport PyMem_Malloc, PyMem_Free

cdef struct job:
    int r
    int c
    numpy.int64_t dist


# or use a deque from C++
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.initializedcheck(False)
@cython.cdivision(True)
cdef class JobQueue:
    cdef job * jobs
    cdef int top, bottom, size

    def __cinit__(self, int size):
        self.jobs = <job *>PyMem_Malloc(size*sizeof(job))
        if self.jobs == NULL:
            raise MemoryError()
        self.top = 0
        self.bottom = 0
        self.size = size

    def __dealloc__(self):
        if self.jobs != NULL:
            PyMem_Free(self.jobs)

    cdef void put(self, job j):
        self.jobs[self.top % self.size] = j
        self.top += 1

    cdef job get(self):
        self.bottom += 1
        return self.jobs[(self.bottom-1) % self.size]

    cdef bint empty(self):
        return self.bottom == self.top


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.initializedcheck(False)
def flood(numpy.ndarray[numpy.uint8_t, ndim=2] grid):
    cdef numpy.ndarray[numpy.int64_t, ndim=2] distances
    cdef numpy.ndarray[char, ndim=2] directions
    cdef job j, j_
    cdef int r, c
    cdef int direction
    cdef char reverse

    # Initialize everything as unreachable
    distances = numpy.full((grid.shape[0], grid.shape[1]), -1, dtype=numpy.int64)
    directions = numpy.full((grid.shape[0], grid.shape[1]), b' ', dtype=('a', 1))

    cdef JobQueue jobs = JobQueue(grid.shape[0] * grid.shape[1])
    for r, c in numpy.argwhere(grid == 2):
        directions[r, c] = b'+'
        distances[r, c] = 0
        j = job(r, c, 1)  # next distance
        jobs.put(j)

    while not jobs.empty():
        j_ = jobs.get()

        for direction in range(4):
            j = job(j_.r, j_.c, j_.dist)
            while True:
                # walk that direction, but stop at boundary
                # this ugly if was easy to convert to Cython ;)
                if direction == 0:  # down
                    j.r += 1
                    reverse = b'^'
                    if j.r == grid.shape[0]:
                        break
                elif direction == 1:  # right
                    j.c += 1
                    reverse = b'<'
                    if j.c == grid.shape[1]:
                        break
                elif direction == 2:  # up
                    j.r -= 1
                    reverse = b'v'
                    if j.r < 0:
                        break
                else:  # left
                    j.c -= 1
                    reverse = b'>'
                    if j.c < 0:
                        break
                # singularities cannot be beamed trough
                # no point of checking past a safe station
                if grid[j.r, j.c] > 1:
                    break
                # been there better? skip but check further if not a node
                if 0 <= distances[j.r, j.c] <= j.dist:
                    if grid[j.r, j.c] == 1:
                        break
                    continue
                distances[j.r, j.c] = j.dist
                directions[j.r, j.c] = reverse
                # if it's a transport node,
                # we need to increase our distance and
                # schedule a job
                if grid[j.r, j.c] == 1:
                    j.dist += 1
                    jobs.put(j)

    return distances, directions
