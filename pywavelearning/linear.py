"""

Linear wave calculations 

Original conde obtained from here: https://github.com/ChrisBarker-NOAA/wave_utils

"""

import numpy as np
from scipy.constants import g

def wave_number(omega, h, g=g):
    """
    Computes the wave number for given frequency and water depth

    ----------
    Args:
        omega [Mandatory (float)]: Wave frequency

        h [Mandatory (float)]: Water depth

        g [Optinal (float)]: Gravitational acceleration (defaults to 9.80665 m/s^2)
    ----------
    Returns:
        k [Mandatory (float)]: Wave number
    """

    p = omega**2 * h / g
    q = dispersion(p)
    k = q * omega**2 / g

    return k


def frequency(k, h, g=g):
    """
    Computes the frequency for a given wave number and water depth
    assumes linear dispersion relationship

    ----------
    Args:
        k [Mandatory (float)]: Wave number

        h [Mandatory (float)]: Water depth

        g [Mandatory (float)]: Gravitational acceleration (defaults to 9.80665 m/s^2)
    ----------
    Returns:
        k [Mandatory (float)]: wave frequency
    """
    return np.sqrt(g * k * np.tanh(k * h))


def dispersion(p, tol=1e-14, max_iter=1000):
    """
    The linear dispersion relation in non-dimensional form:
    
    finds q, given p
    q = gk/omega^2     non-d wave number
    p = omega^2 h / g   non-d water depth
    
    Starts with the Fenton and McKee approximation, then iterates with Newton's
    method until accurate to within tol.
    
    ----------
    Args:
        p [Mandatory (float)]: Non-dimensional water depth

        tol [Optional (float)]: acceptable tolerance

        max_iter [Optional (int)]: maximum number of iterations to accept
    ----------
    Returns:
        q [Mandatory (float)]: wave dispersion relationship
    """

    if p <= 0.0:
        raise ValueError("Non dimensional water depth d must be >= 0.0")
    
    # First guess (from Fenton and McKee):
    q = np.tanh(p ** 0.75) ** (-2.0 / 3.0)

    iter = 0
    f = q * np.tanh(q * p) - 1
    while abs(f) > tol:
        qp = q * p
        fp = qp / (np.cosh(qp) ** 2) + np.tanh(qp)
        q = q - f / fp
        f = q * np.tanh(q * p) - 1
        iter += 1
        if iter > max_iter:
            raise RuntimeError("Maximum number of iterations reached in dispersion()")
    return q


def max_u(a, omega, h, z=None, g=g):
    """
    Compute the maximum Eulerian horizontal velocity at a given depth
        
    ----------
    Args:
        a [Mandatory (float)]: Wave amplitude (1/2 the height)

        omega [Mandatory (float)]: Wave frequency

        h [Mandatory (float)]: Water depth

        Z [Optional (float)]: Depth at which to compute the velocity. Equals to
                              h if None

        
    ----------
    Returns:
        u [Mandatory (float)]: horizontal velocity
    """

    z = h if z is None else z
    k = wave_number(g, omega, h)
    u = a * omega * (np.cosh(k * (h + z)) / np.sinh(k * h))

    return u


def amp_scale_at_depth(omega, h, z, g=g):
    """
    Compute the scale factor of the orbital amplitude at the given depth
        
    ----------
    Args:
        a [Mandatory (float)]: Wave amplitude (1/2 the height)

        omega [Mandatory (float)]: Wave frequency

        h [Mandatory (float)]: Water depth

        z [Optional (float)]:  depth at which to compute the scale factor

        g [Optional (float)]: Gravitational acceleration (defaults to 9.80665 m/s^2)
    ----------
    Returns:
        a [Mandatory (float)]: depth at which to compute the scale factor
    """

    k = wave_number(g, omega, h)

    return np.cosh(k * (h + z)) / np.cosh(k * (h))


def celerity(k, h, g=g):
    """
    Compute the celerity (wave speed, phase speed) for a given wave number and depth
        
    ----------
    Args:
        k [Mandatory (float)]: Wave number

        h [Mandatory (float)]: Water depth

        g [Optional (int)]: Gravitational acceleration (defaults to 9.80665 m/s^2)
    ----------
    Returns:
        c [Mandatory (float)]: wave celerity
    """

    C = np.sqrt(g / k * np.tanh(k * h))

    return C


def group_speed(k, h, g=g):
    """
    Compute the group speed for a given wave number and depth
        
    ----------
    Args:
        k [Mandatory (float)]: Wave number

        h [Mandatory (float)]: Water depth

        g [Optional (int)]: Gravitational acceleration (defaults to 9.80665 m/s^2)
    ----------
    Returns:
        c [Mandatory (float)]: wave celerity
    """
    
    n = 1.0 / 2 * (1 + (2 * k * h / np.sinh(2 * k * h)))
    Cg = n * celerity(k, h, g)

    return Cg


def shoaling_coeff(omega, h0, h2, g=g):
    """
    Compute the shoaling coeff for two depths: h0 and h2.
    The shoaling coeff is the ratio of wave height (H2) at a particular
    point of interest to the original or deep water wave height (H0).
    Pass in h0 = None for deep water
        
    ----------
    Args:
        omega [Mandatory (float)]: Wave frequency

        h0 [Mandatory (float)]: Initial water dept

        h0 [Mandatory (float)]: Depth at which to compute the shoaling coeff

        g [Optional (int)]: Gravitational acceleration (defaults to 9.80665 m/s^2)
    ----------
    Returns:
        s [Mandatory (float)]: shoaling coeff    
    """
    k2 = wave_number(g, omega, h2)
    Cg2 = group_speed(k2, h2, g)
    if h0 is not None:
        k0 = wave_number(g, omega, h0)
        Cg0 = group_speed(k0, h0, g)
        Ks = np.sqrt(Cg0 / Cg2)
        return Ks
    else:  # Deep water
        return np.sqrt((g / (2 * omega)) / Cg2)