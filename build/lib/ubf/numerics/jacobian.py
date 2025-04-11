"""
Numerical methods for computing Jacobians.

This module provides finite-difference methods for computing Jacobians,
which are essential for the UBF framework.
"""

import numpy as np
from typing import Callable


def numerical_jacobian(f_handle: Callable, var: np.ndarray, num_vars: int, epsilon_fd: float = 1e-6) -> np.ndarray:
    """
    Compute the Jacobian of a function using the central difference method.
    
    Args:
        f_handle: The function to differentiate.
        var: The point at which to compute the Jacobian.
        num_vars: Number of variables in the input.
        epsilon_fd: Step size for finite difference.
        
    Returns:
        The Jacobian matrix.
    """
    f0 = f_handle(var)
    f0 = np.array(f0).flatten()
    len_f0 = f0.size
    J = np.zeros((len_f0, num_vars))
    
    for i in range(num_vars):
        var_perturb = var.copy()
        var_perturb[i] += epsilon_fd
        f1 = np.array(f_handle(var_perturb)).flatten()
        var_perturb[i] -= 2 * epsilon_fd
        f2 = np.array(f_handle(var_perturb)).flatten()
        J[:, i] = (f1 - f2) / (2 * epsilon_fd)
        
    return J


def numerical_jacobian_c_u(c_handle: Callable, x: np.ndarray, u: np.ndarray, n: int, m: int, 
                            epsilon_fd: float = 1e-6) -> np.ndarray:
    """
    Compute the Jacobian of a function with respect to the control input.
    
    This function specifically computes the Jacobian of the composition function
    with respect to the control input u.
    
    Args:
        c_handle: The composition function.
        x: Current state.
        u: Current control input.
        n: Number of states.
        m: Number of control inputs.
        epsilon_fd: Step size for finite difference.
        
    Returns:
        The Jacobian matrix.
    """
    J = np.zeros((n, m))
    
    for i in range(m):
        u_perturb = u.copy()
        u_perturb[i] += epsilon_fd
        c1 = c_handle(x, u_perturb)
        u_perturb[i] -= 2 * epsilon_fd
        c2 = c_handle(x, u_perturb)
        J[:, i] = (c1 - c2) / (2 * epsilon_fd)
        
    return J 