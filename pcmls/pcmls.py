"""
Core functionality for point cloud moving least squares.
"""
import jax.numpy as jnp
import jax.scipy as jsp
from jax import grad, hessian, jit, vmap
from jax import lax
from jax import custom_jvp

import numpy as np
from scipy.special import comb
from itertools import combinations

from sklearn.neighbors import NearestNeighbors

import progressbar


def _pcmls_gaussian(v, h):
    """
    Compute a Gaussian weight function

    Args:
    v (array_like): n x dim matrix.
        Separation vectors between source points and the current query point.
    h (array_like): n-element 1D or 2D vector or a numeric scalar.
        Weight function scale parameter.

    Returns:
    ndarray: n-element 1D or 2D array.
        Weight function evaluated for each separation vector.
    """
    
    d2 = jnp.sum(v**2, axis=1)
    return jnp.exp(-d2 / h**2)


def _pcmls_wendland(v, h):
    """
    Compute Wendland's weight function.

    Args:
    v (array_like): n x dim matrix.
        Separation vectors between source points and the current query point.
    h (array_like): n-element 1D or 2D vector or a numeric scalar.
        Weight function scale parameter.

    Returns:
    ndarray: n-element 1D or 2D array.
        Weight function evaluated for each separation vector.
    """
    
    d2 = jnp.sum(v**2, axis=1)
    in_bounds = jnp.logical_and(0 < d2, d2 < h**2)
    d2 = jnp.where(in_bounds, d2, 1)
    d = jnp.sqrt(d2)

    w = jnp.where(in_bounds, (1-(d/h))**4 * (4*(d/h)+1), 0)
    return w


# List of weight functions
_all_pcmls_weight_functions = [_pcmls_gaussian, _pcmls_wendland]
_all_pcmls_weight_index_maps = {'gaussian': 0, 'wendland': 1}


# Custom rule for computing monomials with integer powers
@custom_jvp
def _pcmls_int_pow(x, n):
    return jnp.where(n <= 0, jnp.ones_like(x), jnp.power(x, n))


# Custom derivative rule for computing monomials with integer powers
@_pcmls_int_pow.defjvp
def _pcmls_int_pow_jvp(primals, tangents):
    x, n = primals
    x_dot, _ = tangents
    jac = jnp.where(n <= 0, jnp.zeros_like(x_dot), n * _pcmls_int_pow(x, n - 1))
    return _pcmls_int_pow(x, n), jac * x_dot


@jit
def _eval_pcmls(q, p, fp_array, power_array, weight_function_index=0, weight_param=1):
    """
    Workhorse function for evaluating the MLS interpolant and its derivatives

    Args:
    q (array-like): dim-element 1D array.
        The current query point.
    p (array-like): (num_sources) x dim matrix.
        The complete set of source points.
    fp_array (array_like): (num_monomials) x (num_sources) matrix.
        Scalar function values at source points tiled for each monomial.
    power_array (array_like): (num_monomials) x dim x (num_sources).
        Integer powers defining each monomial term in the interpolant.
    weight_function_index (int): Scalar index into the dict of possible
        weight functions.
    weight_param (float): Scalar weight parameter (see weight functions for
        implementation details).

    Returns:
        float: The interpolated value of the scalar function at 'q'.
    """

    q = q[None, :]
    num_monomials = fp_array.shape[0]
    
    v = p - q  # P x dim
    w = lax.switch(weight_function_index, _all_pcmls_weight_functions,
                   v, weight_param)
    
    # Compute monomial basis functions
    v = jnp.tile(jnp.transpose(v[:, :, None], (2, 1, 0)),
                 (num_monomials, 1, 1))  # k x dim x P
    b = _pcmls_int_pow(v, power_array)
    b = jnp.prod(b, axis=1)  # k x P

    w = jnp.tile(w[None, :], (num_monomials, 1))
    wb = w * b  # k x P

    # Construct and solve the linear problem
    rhs = jnp.sum(wb * fp_array, axis=1)  # k x 1
    lhs = jnp.sum(wb[None, :, :] * b[:, None, :], axis=2).T  # k x k
    lhs = (lhs + lhs.T) / 2
    c = jsp.linalg.solve(lhs, rhs[:, None], assume_a='sym')

    return jnp.squeeze(c[0])


# Scalar derivatives of the workhorse function
_grad_pcmls = jit(grad(_eval_pcmls, argnums=0))
_hess_pcmls = jit(hessian(_eval_pcmls, argnums=0))

# Vectorized derivatives of the workhorse function
_veval_pcmls = jit(vmap(_eval_pcmls, in_axes=(0, None, None, None, None, 0)))
_vgrad_pcmls = jit(vmap(_grad_pcmls, in_axes=(0, None, None, None, None, 0)))
_vhess_pcmls = jit(vmap(_hess_pcmls, in_axes=(0, None, None, None, None, 0)))


def enumerate_monomials(m, dim):
    """
    Enumerate integer powers of all monomials of degree <= m in dim-dimensions.

    Args:
    m (int): The maximal degree of the monomial.
    dim (int): The dimension of the variables in each monomial.

    Returns:
    ndarray: (m+dim)!/(d! m!) x dim matrix.
        Complete list of integer powers for all possible monomials.
    """

    divider_positions = list(combinations(range(m+dim), dim))
    num_monomials = len(divider_positions)
    power_matrix = np.ones((num_monomials, m+dim), dtype=int)

    for row, cols in enumerate(divider_positions):
        power_matrix[row, list(cols)] = 0

    power_matrix_idx = np.cumsum(power_matrix == 0, axis=1)
    power_matrix_idx = np.ravel_multi_index(
        (np.indices(power_matrix_idx.shape)[0].ravel(), power_matrix_idx.ravel()), 
        dims=(num_monomials, dim+1))

    power_matrix = np.bincount(power_matrix_idx, weights=power_matrix.ravel(),
                               minlength=(num_monomials * (dim+1)))
    power_matrix = np.reshape(power_matrix, (num_monomials, dim+1))
    power_matrix = power_matrix[:, :dim]
    
    return power_matrix


def pcmls(q, p, fp, m=2, weight_function='gaussian', weight_method='knn',
          weight_param=10, vectorized=False, compute_gradients=True,
          compute_hessians=True, verbose=False):
    """
    Evaluate a moving least squares (MLS) interpolant/derivatives of a scalar
    function defined at scattered data points

    Args:
    q (array-like): (num_queries) x dim matrix.
        List of query points at which to evalute the interpolant/derivatives.
    p (array-like): (num_sources) x dim matrix.
        List of source points at which the scalar function is defiend.
    fp (array_like): (num_sources)-element 1D or 2D vector
        Values of the scalar function at the source points.
    m (int, optional): Maximal degree of the polynomial interpolant.
        Defaults to 2.
    weight_function (str, optional): The type of weight function to use.
        Defaults to 'gaussian'.
    weight_method (str, optional): Defines how the weight parameter for each
        source point is computed. Defaults to 'knn'.
    weight_param (float or int, optional): The scalar weight parameters.
        If weight_method == 'knn', the weight parameter is the nearest-
        neighbor index the defines a unique weight parameter for each query
        point. If weight_method == 'scalar' then this is the uniform parameter
        used for all query points. Defaults to 'knn'.
    vectorized (bool, optional): Whether to perform a vectorized
        computation or to compute the MLS interpolant and its gradients
        serially for all query points. Vectorized computations are faster,
        but very memory hungry. Defaults to False.
    compute_gradients (bool, optional): Whether to compute MLS
        gradients. Defaults to True.
    compute_hessians (bool, optional): Whether to compute MLS Hessians.
        Defaults to True.
    verbose (bool, optional): Whether to produce verbose progress output.
        Defaults to False.

    Returns:
    ndarray: (num_queries)-element 1D or 2D vector.
        The interpolated function values at the query points.
    ndarray: (num_queries) x dim matrix.
        The interpolated function gradients at the query points.
    ndarray: (num_queries) x dim x dim array.
        The interpolated function Hessian matrices at the query points.
    """

    if q.ndim == 1:
        q = q[None, :]
    assert q.ndim == 2, 'Please supply query points as a #q x dim matrix'
    num_queries = q.shape[0]
    dim = q.shape[1]

    assert (p.ndim == 2 and p.shape[1] == dim), \
        'Please supply source points as a #p x dim matrix'
    num_sources = p.shape[0]

    assert fp.size == num_sources, 'Scalar function input must have #p elements'
    if fp.ndim == 1:
        fp = fp[None, :]
    if fp.ndim == 2:
        if fp.shape[0] != 1:
            fp = fp.T
            if fp.shape[0] != 1:
                raise ValueError('Scalar function must be supplied as a vector')
    else:
        raise ValueError('Scalar function must be supplied as a vector')

    assert (isinstance(m, int) and m > 0), \
        'Polynomial order must be a positive integer'
    
    assert (isinstance(weight_param, (int, float)) and weight_param > 0), \
        'Weight parameter must be a positive scalar'

    weight_function_index = _all_pcmls_weight_index_maps[weight_function]

    # Handle weight parameters
    if weight_method == 'scalar':      
        weight_param = weight_param * np.ones((num_queries,1))
        
    elif weight_method == 'knn':      
        nn = NearestNeighbors(n_neighbors=weight_param)
        nn.fit(p)
        weight_param, _ = nn.kneighbors(q)
        weight_param = weight_param[:, -1]
        
    elif weight_method == 'knnmontecarlo':      
        raise ValueError('Monte Carlo parameter estimation not yet implemented!')
        
    else:
        raise ValueError('Invalid weight function parameter estimation method')

    # Generate polynomial powers
    power_array = enumerate_monomials(m, dim)
    num_monomials = power_array.shape[0]
    power_array = np.tile(power_array[:, :, None], (1, 1, num_sources))

    # Re-size the scalar function array (see eval_pcmls)
    fp_array = np.tile(fp, (num_monomials, 1))

    # Convert everything to JAX style arrays
    q = jnp.array(q)
    p = jnp.array(p)
    fp_array = jnp.array(fp_array)
    weight_param = jnp.array(weight_param)
    weight_param = weight_param[:, None]
    power_array = jnp.array(power_array)

    if vectorized:

        if verbose:
            print('Computing function values... ', end='')
        fq = _veval_pcmls(q, p, fp_array, power_array,
                         weight_function_index, weight_param)
        if verbose:
            print('Done')
        
        if compute_gradients:
            if verbose:
                print('Computing gradients... ', end='')
            grad_fq = _vgrad_pcmls(q, p, fp_array, power_array,
                                   weight_function_index, weight_param)
            if verbose:
                print('Done')
        else:
            grad_fq = None
            
        if compute_hessians:
            if verbose:
                print('Computing Hessians... ', end='')
            hess_fq = _vhess_pcmls(q, p, fp_array, power_array,
                                   weight_function_index, weight_param)
            if verbose:
                print('Done')
        else:
            hess_fq = None

    else:

        fq = np.zeros((num_queries, 1))
        grad_fq = np.zeros((num_queries, dim)) \
            if compute_gradients else None
        hess_fq = np.zeros((num_queries, dim, dim)) \
            if compute_hessians else None

        iterable = (progressbar.progressbar(range(num_queries))
                    if verbose else
                    range(num_queries))
        for i in iterable:
            fq[i] = _eval_pcmls(q[i, :], p, fp_array, power_array,
                                weight_function_index, weight_param[i])
            
            if compute_gradients:
                grad_fq[i, :] = _grad_pcmls(q[i, :], p, fp_array, power_array,
                                            weight_function_index, weight_param[i])

            if compute_hessians:
                hess_fq[i, :, :] = _hess_pcmls(q[i, :], p, fp_array, power_array,
                                               weight_function_index, weight_param[i])

    # Restore output to Numpy style arrays
    fq = np.array(fq.ravel())
    grad_fq = np.array(grad_fq)
    hess_fq = np.array(hess_fq)
    
    return fq, grad_fq, hess_fq

