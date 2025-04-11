# Universal Barrier Functions (UBF) 

A Python package implementing the Universal Barrier Function (UBF) methodology for safety-critical control systems.

## Overview

The Universal Barrier Function (UBF) framework provides a general, modular approach to implementing barrier function-based safety controllers for dynamic systems. The package is designed to be highly configurable, allowing users to easily replace all tuning parameters, system dynamics, and safety constraints to adapt the framework to any system.


## Installation

### From PyPI (recommended)

```bash
pip install ubf
```

### From source

Clone the repository and install in development mode:

```bash
git clone https://github.com/Vrushabbh27/ubf.git
cd ubf
pip install -e .
```

## Dependencies

The package requires the following dependencies:
- numpy (>= 1.19.0)
- matplotlib (>= 3.3.0)
- cvxpy (>= 1.1.0)
- scipy (>= 1.5.0)

These will be automatically installed when using pip.

## Usage

### Basic Usage

```python
import numpy as np
from ubf.core import ubf_core
from ubf.numerics import jacobian, integration
from ubf.systems import quadrotor

# Run the quadrotor example
quadrotor.main()
```

### Core Components

The framework is organized into the following main components:

1. **Core UBF Logic (`ubf.core.ubf_core`)**: Contains the implementation of the universal barrier function methodology, including higher-order barrier functions and the log-sum-exp technique for combining multiple barrier functions.

2. **Numerical Methods (`ubf.numerics`)**: 
   - `jacobian.py`: Provides finite-difference methods for computing Jacobians.
   - `integration.py`: Implements forward-Euler integration routines.

3. **System Examples (`ubf.systems`)**: 
   - `quadrotor.py`: A complete example of a 3D quadrotor simulation using the UBF framework.

### Customizing the Framework

#### Replacing System Dynamics

To use the UBF framework with your own system, you need to define the system dynamics function `f(x, u)` that computes the state derivative:

```python
def my_system_dynamics(x, u):
    """
    Define the state derivative for your system.
    
    Args:
        x: State vector.
        u: Control input vector.
        
    Returns:
        State derivative vector.
    """
    dxdt = np.zeros_like(x)
    # Define your system dynamics here
    # ...
    return dxdt
```

#### Defining Safety Constraints

Safety constraints are defined as barrier functions `h(x, u)` that should remain positive for safety:

```python
def my_safety_constraint(x, u):
    """
    Define a safety constraint for your system.
    
    Args:
        x: State vector.
        u: Control input vector.
        
    Returns:
        Scalar value that should remain positive for safety.
    """
    # Define your safety constraint
    # Example: Keep the system within a circle of radius 10
    return 10**2 - x[0]**2 - x[1]**2
```

#### Setting UBF Parameters

All key parameters are user-tunable:

```python
# Simulation parameters
dt = 0.01                   # Time step
tf = 100                    # Final simulation time

# Controller parameters
alpha = 20.0                # Integral controller gain

# UBF parameters
beta = 15.0                 # UBF smoothness parameter
m_order = 2                 # Global higher-order UBF order
alpha_ubf = 5               # UBF K_infty function gain
frac = 1                    # Exponent for UBF

# Individual UBF orders
m_order_indiv = [2, 1, 3]   # Order for each individual UBF
```

#### Complete Example

Here's a complete example of implementing a UBF-QP controller for a 2D system with obstacles:

```python
"""
Universal Barrier Function based Quadratic Program (UBF-QP) based
controller for synthesizing safe control inputs under limited actuation
and for higher order systems.

Author: Vrushabh Zinage
"""

import numpy as np
import matplotlib.pyplot as plt
import cvxpy as cp
from scipy.optimize import lsq_linear

# ============================ Parameters ============================
# Simulation Parameters
dt = 0.01                   # Time step size (seconds)
tf = 3                      # Final simulation time (seconds)
time = np.arange(0, tf+dt, dt)  # Time vector
num_steps = len(time)       # Number of simulation steps

# Controller Parameters
alpha = 25.0                # Integral controller gain
max_iter = 10               # Maximum iterations per step (not used in this example)
tol = 1e-6                  # Tolerance for convergence

# UBF Parameters
beta = 20.0                 # UBF smoothness parameter (higher = closer to max)
m_order = 2                 # Order of the higher-order UBF (m_order >=1)
alpha_ubf = 1               # UBF K_infty function
frac = 1                    # Parameter for UBF

# Regularization Parameter
epsilon = 1e-6              # Small value to avoid singularities in Jacobian

# Forward-Euler Integration Parameters for computing integral control law
N = 55                      # Number of Forward-Euler steps
Delta_tau = 0.01            # Step size for tau (seconds)
T = N * Delta_tau           # Total integration time for composition function c(x,u)

# ============================ System Definition ============================
# Define the system dynamics
def f(x, u):
    return np.array([u[0], u[1]])

# Determine system dimensions
n = 2                       # Number of states
m = 2                       # Number of control inputs

# Define the goal state
x_goal = np.array([5, 4])   # Example goal state for a 2D system

# ========================= Safety Constraints (UBFs) =========================
# Define UBF functions
def h1(x, u):
    # Obstacle 1 (circular obstacle)
    return (x[0]-3)**2 + (x[1]-3)**2 - 0.4

def h2(x, u):
    # Obstacle 2 (circular obstacle)
    return (x[0]-1.5)**2 + (x[1]-1.5)**2 - 0.25

def h3(x, u):
    # Control input constraint (||u|| <= sqrt(200))
    return 200 - np.dot(u, u)

h_funcs = [h1, h2, h3]
num_ubfs = len(h_funcs)

# ===== Helper Functions =====
def log_sum_exp(beta, h_funcs, x, u):
    """Compute the Log-Sum-Exp for combining UBFs."""
    h_vals = np.array([beta * h(x, u) for h in h_funcs])
    h_min = np.min(h_vals)
    sum_exp = np.sum(np.exp(-(h_vals - h_min)))
    return h_min - np.log(sum_exp) + np.log(1/num_ubfs)

def numerical_jacobian(f_handle, var, num_vars):
    """Compute numerical Jacobian using central difference."""
    epsilon_fd = 1e-6
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

def numerical_jacobian_c_u(c_handle, x, u, n, m):
    """Compute Jacobian of composition function with respect to u."""
    epsilon_fd = 1e-6
    J = np.zeros((n, m))
    
    for i in range(m):
        u_perturb = u.copy()
        u_perturb[i] += epsilon_fd
        c1 = c_handle(x, u_perturb)
        u_perturb[i] -= 2 * epsilon_fd
        c2 = c_handle(x, u_perturb)
        J[:, i] = (c1 - c2) / (2 * epsilon_fd)
    
    return J

def forward_euler_integration(x_initial, u, f_handle, N_steps, Delta_tau):
    """Perform Forward-Euler Integration."""
    x = x_initial.copy()
    for _ in range(N_steps):
        x = x + f_handle(x, u) * Delta_tau
    return x

# ===== Constructing the Universal Barrier Function (UBF) =====
# Define the UBF using the Log-Sum-Exp trick
def h_ubf(x, u):
    return log_sum_exp(beta, h_funcs, x, u) / beta + (np.log(num_ubfs) / beta)

# Define the composition function c(x,u) using Forward-Euler Integration
def c(x, u):
    return forward_euler_integration(x, u, f, N, Delta_tau)

# Compute the Jacobian dc/du numerically
def dc_du_func(x, u):
    return numerical_jacobian_c_u(c, x, u, n, m)

# ===== Higher-Order UBFs =====
# Helper functions for computing higher-order UBFs
def compute_dh_dx(h_func, x, u):
    return numerical_jacobian(lambda x_val: np.array([h_func(x_val, u)]), x, n)

def compute_dh_du(h_func, x, u):
    return numerical_jacobian(lambda u_val: np.array([h_func(x, u_val)]), u, m)

def compute_phi(x, u):
    A = dc_du_func(x, u)
    b = x_goal - c(x, u)
    phi_sol, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
    return alpha * phi_sol

# Initialize storage for higher-order UBFs
h_orders = [h_ubf]  # h^1 = h_ubf

# Define the second-order UBF
def h_ubf_2(x, u):
    h_prev = h_ubf
    dh_dx = compute_dh_dx(h_prev, x, u)
    dh_du = compute_dh_du(h_prev, x, u)
    phi_val = compute_phi(x, u)
    h_dot = np.dot(dh_dx, f(x, u)) + np.dot(dh_du, phi_val)
    return float(h_dot + alpha_ubf * (h_prev(x, u) ** frac))

if m_order >= 2:
    h_orders.append(h_ubf_2)

# Use h_final based on the order
h_final = h_orders[m_order - 1]

# ===== Simulation Setup =====
# Initialize state and control vectors
x_current = np.array([1, 1])   # Initial state
u_current = np.zeros(m)        # Initial control inputs

# Preallocate storage for trajectories
x_traj = np.zeros((n, num_steps))  # State trajectory
u_traj = np.zeros((m, num_steps))  # Control input trajectory
h_traj = np.zeros((m_order, num_steps))  # Higher-order UBF trajectories

# Set initial conditions
x_traj[:, 0] = x_current
u_traj[:, 0] = u_current

# Calculate initial UBF values
for order in range(m_order):
    if order == 0:
        h_traj[order, 0] = float(h_orders[order](x_current, u_current))
    elif order == 1:
        h_traj[order, 0] = float(h_ubf_2(x_current, u_current))

# ===== Simulation Loop =====
for k in range(1, num_steps):
    # Current state and control
    x = x_current.copy()
    u = u_current.copy()
    
    # Compute phi using the integral controller
    phi = compute_phi(x, u)
    
    # Compute p(x,u) and q(x,u) for the QP
    dh_dx = compute_dh_dx(h_final, x, u)
    dh_du = compute_dh_du(h_final, x, u)
    
    p = dh_du  # Should have shape (1, m)
    q = float(np.dot(dh_dx, f(x, u)) + np.dot(dh_du, phi) + alpha_ubf * (h_final(x, u) ** frac))
    
    # Formulate and solve the QP
    # Minimize: 0.5 * ||v||^2
    # Subject to: p @ v + q >= 0
    v = cp.Variable(m)
    objective = cp.Minimize(0.5 * cp.sum_squares(v))
    constraints = [p @ v + q >= 0]
    prob = cp.Problem(objective, constraints)
    
    try:
        prob.solve(solver=cp.OSQP, verbose=False)
        if v.value is None:
            print(f"Warning: QP did not converge at step {k}")
            v_star = np.zeros(m)
        else:
            v_star = v.value
    except Exception as e:
        print(f"QP solver error at step {k}: {e}")
        v_star = np.zeros(m)
    
    # Update control input using the integral control law
    # du/dt = phi + v_star
    du_dt = phi + v_star
    u_new = u + du_dt * dt
    
    # Update state using Forward Euler Integration
    x_new = x + f(x, u_new) * dt
    
    # Update higher-order UBFs
    for order in range(m_order):
        if order == 0:
            h_traj[order, k] = float(h_orders[order](x_new, u_new))
        elif order == 1:
            h_traj[order, k] = float(h_ubf_2(x_new, u_new))
    
    # Store trajectories
    x_traj[:, k] = x_new
    u_traj[:, k] = u_new
    
    # Update current state and control
    x_current = x_new
    u_current = u_new

# ===== Visualization =====
# Plot Trajectory (for 2D systems)
plt.figure(figsize=(10, 8))
plt.plot(x_traj[0, :], x_traj[1, :], 'b-', linewidth=2, label='Trajectory')
plt.plot(x_goal[0], x_goal[1], 'ro', markersize=10, label='Goal Position')

# Plot Obstacles
theta = np.linspace(0, 2*np.pi, 100)

# Obstacle 1
r1 = np.sqrt(0.4)
x_obs1 = 3 + r1 * np.cos(theta)
y_obs1 = 3 + r1 * np.sin(theta)
plt.fill(x_obs1, y_obs1, 'r', alpha=0.3, edgecolor='k', label='Obstacle 1')

# Obstacle 2
r2 = np.sqrt(0.25)
x_obs2 = 1.5 + r2 * np.cos(theta)
y_obs2 = 1.5 + r2 * np.sin(theta)
plt.fill(x_obs2, y_obs2, 'g', alpha=0.3, edgecolor='k', label='Obstacle 2')

plt.xlabel('x_1')
plt.ylabel('x_2')
plt.title(f'Trajectory with Integral Controller and {m_order}-Order UBF')
plt.legend()
plt.grid(True)
plt.axis('equal')

# Plot Higher-Order UBFs
plt.figure(figsize=(10, 6))
for order in range(m_order):
    plt.subplot(m_order, 1, order + 1)
    plt.plot(time, h_traj[order, :], linewidth=2)
    plt.xlabel('Time (s)')
    plt.ylabel(f'h^({order+1})')
    plt.title(f'Universal Barrier Function of Order {order+1}')
    plt.grid(True)
plt.tight_layout()

# Plot Control Inputs Over Time
plt.figure(figsize=(10, 8))
for i in range(m):
    plt.subplot(m, 1, i + 1)
    plt.plot(time, u_traj[i, :], linewidth=2)
    plt.xlabel('Time (s)')
    plt.ylabel(f'u_{i+1}')
    plt.title(f'Control Input u_{i+1} Over Time')
    plt.grid(True)
plt.tight_layout()

# Plot Control Input Norm Over Time
plt.figure(figsize=(10, 6))
u_norm = np.sqrt(u_traj[0, :]**2 + u_traj[1, :]**2)
plt.plot(time, u_norm, linewidth=2)
plt.xlabel('Time (s)')
plt.ylabel('||u||')
plt.title('Norm of Control Input Over Time')
plt.grid(True)
plt.show()
```

## Quadrotor Example

The package includes a complete example of a 3D quadrotor system navigating through an environment with obstacles. The quadrotor is controlled using the UBF framework to ensure safety constraints are maintained.

To run the quadrotor example:

```python
from ubf.systems import quadrotor
quadrotor.main()
```

This will simulate the quadrotor and generate visualizations showing:
1. The 3D trajectory of the quadrotor
2. The values of the barrier functions over time
3. The control inputs over time
4. The norm of the control input over time

## License

This project is licensed under the MIT License - see the LICENSE file for details.


## Citation

If you use this package in your research, please cite:

```
Zinage, V., & Bakolas, E. (2023). Universal Barrier Functions for Safety and Stability. [To be submitted].
``` 
