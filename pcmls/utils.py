"""
Utilities for point cloud moving least squares
"""
import jax.numpy as jnp
from jax import grad, hessian, jit, vmap


def franke(x, y):
    """
    Compute Franke's bivariate test function, a weighted sum of exponentials,
    at the coordinates x and y.

    Args:
    x (array_like):  n-element 1D or 2D vector
        x-coordinate(s) at which to evaluate the function.
    y (array_like):  n-element 1D or 2D vector
        y-coordinate(s) at which to evaluate the function.

    Returns:
    ndarray: n-element 1D or 2D vector the same size as x and y
        The values of Franke's function at the given coordinates.
    """

    term1 = 0.75 * jnp.exp(-(0.25 * (9 * x - 2) ** 2) - 0.25 * ((9 * y - 2) ** 2))
    term2 = 0.75 * jnp.exp(-((9 * x + 1) ** 2) / 49.0 - 0.1 * (9 * y + 1))
    term3 = 0.5 * jnp.exp(-(9 * x - 7) ** 2 / 4.0 - 0.25 * ((9 * y - 3) ** 2))
    term4 = -0.2 * jnp.exp(-(9 * x - 4) ** 2 - (9 * y - 7) ** 2)

    return term1 + term2 + term3 + term4


_grad_franke = grad(franke, argnums=(0, 1))
_vgrad_franke = jit(vmap(_grad_franke, in_axes=(0, 0)))


def grad_franke(x, y):
    """
    Compute the gradient of Franke's bivariate test function at the coordinates
    x and y

    Args:
    x (array_like):  n-element 1D or 2D vector
        x-coordinate(s) at which to evaluate the function gradient.
    y (array_like):  n-element 1D or 2D vector
        y-coordinate(s) at which to evaluate the function gradient.

    Returns:
    ndarray: n x 2 matrix
        The gradient of Franke's function at the given coordinates.
    """

    g = _vgrad_franke(x, y)
    g = jnp.array([g[0], g[1]]).T

    return g


_hess_franke = hessian(franke, argnums=(0, 1))
_vhess_franke = jit(vmap(_hess_franke, in_axes=(0, 0)))


def hess_franke(x, y):
    """
    Compute the Hessian of Franke's bivariate test function at the coordinates
    x and y

    Args:
    x (array_like):  n-element 1D or 2D vector
        x-coordinate(s) at which to evaluate the function Hessian.
    y (array_like):  n-element 1D or 2D vector
        y-coordinate(s) at which to evaluate the function Hessian.

    Returns:
    ndarray: n x 2 x 2 matrix
        The Hessian of Franke's function at the given coordinates.
    """

    h = _vhess_franke(x, y)
    h = jnp.array([[
        [h[0][0][i], h[0][1][i]],
        [h[1][0][i], h[1][1][i]]
    ] for i in range(len(h[0][0]))])

    return h
