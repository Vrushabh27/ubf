"""
Core functionality for Universal Barrier Function (UBF) methods.

This module provides the core implementation of the Universal Barrier Function (UBF)
framework, including higher-order barrier function computations and the log-sum-exp
technique for combining multiple barrier functions.
"""

import numpy as np
import cvxpy as cp
from typing import List, Callable, Tuple, Optional


def log_sum_exp(beta: float, h_funcs: List[Callable], x: np.ndarray, u: np.ndarray) -> float:
    """
    Compute the log-sum-exp of a list of barrier functions.
    
    This allows smooth composition of multiple barrier functions into a single barrier.
    
    Args:
        beta: UBF smoothness parameter.
        h_funcs: List of barrier functions to combine.
        x: Current state vector.
        u: Current control input vector.
        
    Returns:
        The log-sum-exp value of the barrier functions.
    """
    num_ubfs = len(h_funcs)
    h_vals = np.array([beta * h(x, u) for h in h_funcs])
    h_min = np.min(h_vals)
    sum_exp = np.sum(np.exp(-(h_vals - h_min)))
    return h_min + np.log(sum_exp) + np.log(1/num_ubfs)


def compute_individual_higher_order_ubfs(
    h_funcs: List[Callable], 
    m_order_indiv: List[int], 
    f: Callable, 
    alpha_ubf: float,
    frac: float,
    n: int
) -> List[List[Callable]]:
    """
    Compute higher-order barrier functions for each individual barrier function.
    
    Args:
        h_funcs: List of original barrier functions.
        m_order_indiv: List containing the order for each individual UBF.
        f: System dynamics function.
        alpha_ubf: UBF gain parameter.
        frac: Exponent for UBF.
        n: Number of states.
        
    Returns:
        List of lists, where each inner list contains the higher-order functions for one barrier.
    """
    # Import here to avoid circular imports
    from ubf.numerics.jacobian import numerical_jacobian
    
    num_ubfs = len(h_funcs)
    h_orders_indiv = []
    
    for i in range(num_ubfs):
        orders = []
        orders.append(h_funcs[i])
        
        for order in range(1, m_order_indiv[i]):
            def h_prev(x, u, h_prev_func=orders[-1]):
                return h_prev_func(x, u)
                
            def dh_prev_dx(x, u, h_prev_func=orders[-1]):
                return numerical_jacobian(lambda x_val: np.array(h_prev_func(x_val, u)).reshape(1,), x, n)
                
            def h_dot(x, u, h_prev_func=orders[-1]):
                return np.dot(dh_prev_dx(x, u), f(x, u))[0]
                
            new_h = lambda x, u, h_prev=h_prev, h_dot=h_dot: h_dot(x, u) + alpha_ubf * (h_prev(x, u) ** frac)
            orders.append(new_h)
            
        h_orders_indiv.append(orders)
        
    return h_orders_indiv


def construct_ubf(
    h_final_indiv: List[Callable], 
    beta: float
) -> Callable:
    """
    Construct the Universal Barrier Function from individual barrier functions.
    
    Args:
        h_final_indiv: List of final higher-order barrier functions.
        beta: UBF smoothness parameter.
        
    Returns:
        The unified UBF function.
    """
    def h_ubf(x, u):
        return log_sum_exp(beta, h_final_indiv, x, u) / beta
    
    return h_ubf


def compute_global_higher_order_ubfs(
    h_ubf: Callable,
    m_order: int,
    f: Callable,
    alpha: float,
    alpha_ubf: float,
    frac: float,
    n: int,
    m: int,
    x_goal: np.ndarray,
    c_func: Callable,
    dc_du: Callable
) -> List[Callable]:
    """
    Compute global higher-order UBFs.
    
    Args:
        h_ubf: Base UBF function.
        m_order: Global higher-order UBF order.
        f: System dynamics function.
        alpha: Integral controller gain.
        alpha_ubf: UBF gain parameter.
        frac: Exponent for UBF.
        n: Number of states.
        m: Number of control inputs.
        x_goal: Goal state.
        c_func: Composition function.
        dc_du: Jacobian of composition function with respect to u.
        
    Returns:
        List of higher-order UBF functions.
    """
    # Import here to avoid circular imports
    from ubf.numerics.jacobian import numerical_jacobian
    
    h_orders_global = [h_ubf]
    
    for order in range(1, m_order):
        def dh_prev_dx(x, u, h_prev=h_orders_global[-1]):
            return numerical_jacobian(lambda x_val: np.array(h_prev(x_val, u)).reshape(1,), x, n)
            
        def dh_prev_du(x, u, h_prev=h_orders_global[-1]):
            return numerical_jacobian(lambda u_val: np.array(h_prev(x, u_val)).reshape(1,), u, m)
            
        def phi(x, u):
            A = dc_du(x, u)
            b = x_goal - c_func(x, u)
            phi_sol, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
            return alpha * phi_sol
            
        def h_dot(x, u, h_prev=h_orders_global[-1]):
            return np.dot(dh_prev_dx(x, u), f(x, u)) + np.dot(dh_prev_du(x, u), phi(x, u))
            
        h_new = lambda x, u, h_prev=h_orders_global[-1], h_dot=h_dot: h_dot(x, u) + alpha_ubf * (h_prev(x, u) ** frac)
        h_orders_global.append(h_new)
        
    return h_orders_global


def solve_ubf_qp(
    x: np.ndarray,
    u: np.ndarray,
    f: Callable,
    h_final: Callable,
    alpha: float,
    alpha_ubf: float,
    frac: float,
    x_goal: np.ndarray,
    c_func: Callable,
    dc_du: Callable,
    n: int,
    m: int,
    solver=cp.OSQP
) -> np.ndarray:
    """
    Solve the quadratic program for UBF control.
    
    Args:
        x: Current state.
        u: Current control input.
        f: System dynamics function.
        h_final: Final higher-order UBF function.
        alpha: Integral controller gain.
        alpha_ubf: UBF gain parameter.
        frac: Exponent for UBF.
        x_goal: Goal state.
        c_func: Composition function.
        dc_du: Jacobian of composition function with respect to u.
        n: Number of states.
        m: Number of control inputs.
        solver: CVXPY solver to use.
        
    Returns:
        The optimal control adjustment.
    """
    # Import here to avoid circular imports
    from ubf.numerics.jacobian import numerical_jacobian
    
    def dhm_dx(x, u):
        return numerical_jacobian(lambda x_val: np.array(h_final(x_val, u)).reshape(1,), x, n)

    def dhm_du(x, u):
        return numerical_jacobian(lambda u_val: np.array(h_final(x, u_val)).reshape(1,), u, m)
    
    A = dc_du(x, u)
    b = x_goal - c_func(x, u)
    phi_sol, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
    phi_val = alpha * phi_sol

    # Use the Jacobian from h_final to set up the QP.
    p = dhm_du(x, u)  # Should have shape (1, m)
    q = np.dot(dhm_dx(x, u), f(x, u)) + np.dot(dhm_du(x, u), phi_val) + alpha_ubf * (h_final(x, u) ** frac)
    
    v = cp.Variable(m)
    objective = cp.Minimize(0.5 * cp.sum_squares(v))
    constraints = [p @ v + q >= 0]
    prob = cp.Problem(objective, constraints)
    prob.solve(solver=solver, verbose=False)
    
    if v.value is None:
        print("Warning: QP did not converge")
        v_star = np.zeros(m)
    else:
        v_star = v.value
    
    du_dt = phi_val + v_star
    return du_dt 