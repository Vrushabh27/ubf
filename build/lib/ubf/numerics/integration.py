"""
Numerical integration methods for the UBF framework.

This module provides forward-Euler integration routines for numerical simulation
of dynamical systems in the UBF framework.
"""

import numpy as np
from typing import Callable, Tuple, List, Optional


def forward_euler_integration(x_initial: np.ndarray, u: np.ndarray, f_handle: Callable, 
                              N_steps: int, Delta_tau: float) -> np.ndarray:
    """
    Perform forward Euler integration of a dynamical system.
    
    Args:
        x_initial: Initial state.
        u: Control input (assumed constant during integration).
        f_handle: System dynamics function.
        N_steps: Number of integration steps.
        Delta_tau: Integration step size.
        
    Returns:
        The final state after integration.
    """
    x_temp = x_initial.copy()
    for _ in range(N_steps):
        x_temp = x_temp + f_handle(x_temp, u) * Delta_tau
    return x_temp


def simulate_system(x_initial: np.ndarray, u_initial: np.ndarray, f: Callable,
                    dt: float, tf: float, 
                    controller: Callable,
                    m_order: int = 1,
                    h_orders_global: Optional[List[Callable]] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Simulate a dynamical system with UBF control.
    
    Args:
        x_initial: Initial state.
        u_initial: Initial control input.
        f: System dynamics function.
        dt: Time step for simulation.
        tf: Final simulation time.
        controller: Controller function that returns control adjustment.
        m_order: Order of the global UBF.
        h_orders_global: List of global higher-order UBF functions.
        
    Returns:
        Tuple containing state trajectory, control trajectory, and UBF values.
    """
    time = np.arange(0, tf+dt, dt)
    num_steps = len(time)
    
    n = x_initial.size  # Number of states
    m = u_initial.size  # Number of control inputs
    
    x_traj = np.zeros((n, num_steps))
    u_traj = np.zeros((m, num_steps))
    
    if h_orders_global is not None:
        h_traj = np.zeros((m_order, num_steps))
    else:
        h_traj = None
    
    x_current = x_initial.copy()
    u_current = u_initial.copy()
    
    x_traj[:, 0] = x_current
    u_traj[:, 0] = u_current
    
    if h_orders_global is not None:
        for order in range(m_order):
            h_traj[order, 0] = h_orders_global[order](x_current, u_current)
    
    for k in range(1, num_steps):
        # Compute control adjustment
        du_dt = controller(x_current, u_current)
        
        # Update control
        u_new = u_current + du_dt * dt
        
        # Update state using dynamics
        x_new = x_current + f(x_current, u_new) * dt
        
        # Store trajectory
        x_traj[:, k] = x_new
        u_traj[:, k] = u_new
        
        if h_orders_global is not None:
            for order in range(m_order):
                h_traj[order, k] = h_orders_global[order](x_new, u_new)
        
        # Update current state and control
        x_current = x_new.copy()
        u_current = u_new.copy()
    
    return x_traj, u_traj, h_traj 