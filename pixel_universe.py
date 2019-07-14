import numpy as np
from tqdm import tqdm


class Entity(object):
    """An entity that exists in the pixel universe.

    Attributes
    ----------
    m : float
        The mass of the entity.
    r : float
        The radius of the entity.
    x : float
        The x coordinate of the entity.
    y : float
        The y coordinate of the entity.
    dx : float
        The velocity in the x-direction of the entity.
    dy : float
        The velocity in the y-direction of the entity.

    """
    def __init__(self, m, r, x, y, dx, dy):
        """Create an Entity.

        Parameters
        ----------
        m : float
            The initial mass of the entity.
        r : float
            The initial radius of the entity.
        x : float
            The initial x coordinate of the entity.
        y : float
            The initial y coordinate of the entity.
        dx : float
            The initial velocity in the x-direction of the entity.
        dy : float
            The initial velocity in the y-direction of the entity.

        """
        self.m = m
        self.r = r
        self.x = x
        self.y = y
        self.dx = dx
        self.dy = dy

    def __repr__(self):
        x = str(self.__dict__).lstrip('{').rstrip('}')
        return 'Entity({})'.format(x)

    def get_vec(self):
        """Get the vector representation of the entity's position

        Returns
        -------
        vector : ndarray of float
            `vector[0]` contains the x coordinate of the entity.
            `vector[1]` contains the y coordinate of the entity.
        """
        return np.array((self.x, self.y))

    def update(self, acc, t_step):
        """Update the position and velocity of the entity in place.

        This uses the kinematic equations of motion, and assumes a constant
        velocity for the duration of `t_step`.

        Parameters
        ----------
        acc : tuple of float
            Acceleration of the entity. `acc[0]` is the acceleration in the
            x-direction, `acc[1]` is the acceleration in the y-direction.
        t_step : float
            The size of the timestep for which the acceleration is applied.

        """
        acc_x = acc[0]
        acc_y = acc[1]

        dx_i = self.dx
        dy_i = self.dy

        self.dx = dx_i + acc_x*t_step
        self.dy = dy_i + acc_y*t_step
        self.x = self.x + (dx_i + self.dx)/2*t_step
        self.y = self.y + (dy_i + self.dy)/2*t_step


def get_positions(entities: list):
    """Return an array of position vectors for a list of Entity objects."""
    pos = [e.get_vec() for e in entities]
    return np.concatenate(pos, axis=0).reshape(len(entities), -1)


def get_masses(elements: list):
    """Return an array of masses for a list of Entity objects."""
    masses = [e.m for e in elements]
    return np.array(masses)


# The "borrowed" functions below use clever broadcasting to vectorize
# calculating the acceleration as much as possible. Alas, these are not my own,
# and are gratefully borrowed from this answer on Stack Overflow:
# https://stackoverflow.com/questions/52562117/efficiently-compute-n-body-gravitation-in-python

def get_displacements(positions):
    """"Get a matrix representation of displacements.

    Get a matrix representation of the displacements between all position
    vectors in array `positions`.

    Parameters
    ----------
    positions : ndarray of float
        Ndarray of shape (n, 2) containing position vectors.
        See the `get_positions` function.

    Returns
    -------
    displacements : ndarray of float
        Ndarray of shape (n, n, 2) containing the displacements between
        each element in `positions` and every other element in `positions`.

    """
    return positions.reshape(1, -1, 2) - positions.reshape(-1, 1, 2)


def get_distances(displacements):
    """Compute the norm of each element in `displacements`

    Parameters
    ----------
    displacements: ndarray of float
        Ndarray of shape (n, n, 2) containing the displacements. between
        n vectors and every other vector. Recommended to use the direct
        output of the `get_displacements` function.

    Returns
    -------
    distances : ndarray of float
        Ndarray of shape (n, n) containing the norm of each element in
        `displacements`.

    """
    distances = np.linalg.norm(displacements, axis=2)
    return distances


def get_mass_matrix(masses):
    return masses.reshape(1, -1, 1) * masses.reshape(-1, 1, 1)


def get_accelerations(masses, displacements, distances, G):
    mass_matrix = get_mass_matrix(masses)
    forces = G * mass_matrix * displacements / np.expand_dims(distances, 2)**3
    return forces.sum(axis=1)/masses.reshape(-1, 1)


def accelerations(positions, masses, G=1):
    # Numpy array broadcasting allows us to multiply all the weights together
    mass_matrix = masses.reshape(1, -1, 1)*masses.reshape(-1, 1, 1)
    # Numpy array broadcasting allows to us to subtract the position from each
    # entity from every other entity
    displacements = positions.reshape(1, -1, 2) - positions.reshape(-1, 1, 2)
    distances = np.linalg.norm(displacements, axis=2)  # Our final axis contain our "coordinates"
    distances[distances == 0] = 1  # Avoid div by zero errors
    # We need the extra **3 (instead of **2) is due to the above line
    forces = G * mass_matrix * displacements / ((np.expand_dims(distances, 2))**3)
    return forces.sum(axis=1)/masses.reshape(-1, 1)
