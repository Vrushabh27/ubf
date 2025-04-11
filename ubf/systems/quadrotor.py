"""
3D Quadrotor simulation using the UBF framework.

This module provides a specific example of using the UBF framework for a 3D quadrotor system,
but is designed to be general enough that users can modify parameters and replace the dynamics
and constraint functions to apply it to other systems.
"""

import numpy as np
import matplotlib.pyplot as plt
import cvxpy as cp

# =============================================================================
# General Simulation Parameters (User-Tunable)
# =============================================================================
dt = 0.005                # Time step in seconds.
tf = 80                   # Final simulation time in seconds.
time = np.arange(0, tf+dt, dt)
num_steps = len(time)

# =============================================================================
# General Controller Parameters (User-Tunable)
# =============================================================================
alpha = 25.0              # Integral controller gain.
max_iter = 10             # Maximum NR iterations (not explicitly used below).
tol = 1e-6                # Tolerance for convergence.

# =============================================================================
# General UBF Parameters (User-Tunable)
# =============================================================================
beta = 20.0               # UBF smoothness parameter.
m_order = 1               # Global higher-order UBF order.
alpha_ubf = 3             # UBF K_infty function gain.
frac = 1                  # Exponent for UBF.

# =============================================================================
# General Individual UBF Orders (User-Tunable)
# =============================================================================
num_ubfs = 3              # Number of safety constraints.
m_order_indiv = [2, 2, 1] # Order for each individual UBF.
epsilon = 1e-6            # Regularization parameter.

# =============================================================================
# General Forward-Euler Integration Parameters (User-Tunable)
# =============================================================================
N = 2915                  # Number of integration steps.
Delta_tau = 0.01          # Integration step size.
T_int = N * Delta_tau     # Total integration time.

# =============================================================================
# Specific Quadrotor Parameters and Dynamics (Example of a Specific System)
# These dynamics and parameters can be replaced by the user.
# =============================================================================
m_q = 1.0                 # Mass (kg)
Ixx = 0.01                # Moment of inertia about x-axis (kg m^2)
Iyy = 0.01                # Moment of inertia about y-axis (kg m^2)
Izz = 0.02                # Moment of inertia about z-axis (kg m^2)
g = 9.81                  # Gravity (m/s^2)

def f(x, u):
    """
    Computes the state derivative for the 3D quadrotor.
    This dynamics function is provided as an example and can be replaced by the user.
    x: 12-dimensional state vector.
    u: 4-dimensional control vector.
    Returns: 12-dimensional state derivative vector.
    """
    dxdt = np.zeros_like(x)
    dxdt[0:3] = x[6:9]         # Velocities (position derivatives).
    dxdt[3:6] = x[9:12]        # Angular rates (p, q, r).
    
    # Linear accelerations.
    dxdt[6] = (u[0] / m_q) * np.sin(x[4])                  # x acceleration (depends on theta).
    dxdt[7] = (u[0] / m_q) * (-np.sin(x[3]))                 # y acceleration (depends on phi).
    dxdt[8] = (u[0] / m_q) * (np.cos(x[3]) * np.cos(x[4])) - g  # z acceleration.
    
    # Angular accelerations.
    dxdt[9] = (1 / Ixx) * (u[1] - (Iyy - Izz) * x[10] * x[11])
    dxdt[10] = (1 / Iyy) * (u[2] - (Izz - Ixx) * x[9] * x[11])
    dxdt[11] = (1 / Izz) * (u[3] - (Ixx - Iyy) * x[9] * x[10])
    
    return dxdt

n = 12    # Number of states.
m = 4     # Number of control inputs.

# Define the goal state (example for quadrotor; can be changed by the user).
x_goal = np.array([5, 5, 5, 0, 0, 0, 0, 0, 0, 0, 0, 0])

# =============================================================================
# Safety Constraints (UBFs) (General - User Can Replace These Functions)
# =============================================================================
def h1(x, u):
    return (x[0] - 3)**2 + (x[1] - 3)**2 + (x[2] - 3)**2 - 0.4

def h2(x, u):
    return (x[0] - 1.5)**2 + (x[1] - 1.5)**2 + (x[2] - 2.0)**2 - 0.25

def h3(x, u):
    return 200 - np.dot(u, u)

h_funcs = [h1, h2, h3]

# =============================================================================
# General Helper Functions (Applicable to Any Dynamics and Parameters)
# =============================================================================
def numerical_jacobian(f_handle, var, num_vars):
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
    x_temp = x_initial.copy()
    for _ in range(N_steps):
        x_temp = x_temp + f_handle(x_temp, u) * Delta_tau
    return x_temp

def log_sum_exp(beta, h_funcs_list, x, u):
    num_ubfs = len(h_funcs_list)
    h_vals = np.array([beta * h(x, u) for h in h_funcs_list])
    h_min = np.min(h_vals)
    sum_exp = np.sum(np.exp(-(h_vals - h_min)))
    return h_min + np.log(sum_exp) + np.log(1/num_ubfs)

# =============================================================================
# Compute Higher-Order Barrier Functions for Each h_funcs[i] (General)
# =============================================================================
def compute_individual_ubfs():
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

h_orders_indiv = compute_individual_ubfs()
h_final_indiv = [orders[-1] for orders in h_orders_indiv]

# =============================================================================
# Construct the Universal Barrier Function (UBF) (General)
# =============================================================================
def h_ubf(x, u):
    return log_sum_exp(beta, h_final_indiv, x, u) / beta

def dh_dx(x, u):
    return numerical_jacobian(lambda x_val: np.array(h_ubf(x_val, u)).reshape(1,), x, n)

def dh_du(x, u):
    return numerical_jacobian(lambda u_val: np.array(h_ubf(x, u_val)).reshape(1,), u, m)

# =============================================================================
# Integral Controller via Composition Function (General)
# =============================================================================
def c_func(x, u):
    return forward_euler_integration(x, u, f, N, Delta_tau)

def dc_du(x, u):
    return numerical_jacobian_c_u(c_func, x, u, n, m)

# =============================================================================
# Global Higher-Order UBFs (General, for m_order > 1)
# =============================================================================
def compute_global_ubfs():
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

h_orders_global = compute_global_ubfs()
h_final = h_orders_global[-1]

def dhm_dx(x, u):
    return numerical_jacobian(lambda x_val: np.array(h_final(x_val, u)).reshape(1,), x, n)

def dhm_du(x, u):
    return numerical_jacobian(lambda u_val: np.array(h_final(x, u_val)).reshape(1,), u, m)

# =============================================================================
# Simulation Setup and Main Loop (Example: Quadrotor)
# =============================================================================
def run_simulation():
    """
    Run the quadrotor simulation with UBF control.
    
    The following simulation code demonstrates the use of the UBF framework.
    Users can replace the dynamics function f(x,u) and safety constraints h(x,u) along
    with all other parameters to simulate any system.
    
    Returns:
        Tuple containing time, state trajectory, control trajectory, and UBF values.
    """
    x_current = np.array([0, 0, 0.5, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    u_current = np.array([m_q * g, 0, 0, 0])
    
    x_traj = np.zeros((n, num_steps))
    u_traj = np.zeros((m, num_steps))
    h_traj = np.zeros((m_order, num_steps))
    
    x_traj[:, 0] = x_current
    u_traj[:, 0] = u_current
    for order in range(m_order):
        h_traj[order, 0] = h_orders_global[order](x_current, u_current)
    
    for k in range(1, num_steps):
        x = x_current.copy()
        u = u_current.copy()
        
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
        prob.solve(solver=cp.OSQP, verbose=False)
        
        if v.value is None:
            print(f"Warning: QP did not converge at step {k}")
            v_star = np.zeros(m)
        else:
            v_star = v.value
    
        du_dt = phi_val + v_star
        u_new = u + du_dt * dt
    
        x_new = x + f(x, u_new) * dt
    
        for order in range(m_order):
            h_traj[order, k] = h_orders_global[order](x_new, u_new)
        
        x_traj[:, k] = x_new
        u_traj[:, k] = u_new
        
        x_current = x_new.copy()
        u_current = u_new.copy()
        
    return time, x_traj, u_traj, h_traj

# =============================================================================
# Visualization
# =============================================================================
def visualize_results(time, x_traj, u_traj, h_traj):
    """
    Visualize the simulation results.
    
    Args:
        time: Time array.
        x_traj: State trajectory.
        u_traj: Control trajectory.
        h_traj: Barrier function values.
    """
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(x_traj[0, :], x_traj[1, :], x_traj[2, :], 'b-', linewidth=2, label='Trajectory')
    ax.scatter(x_goal[0], x_goal[1], x_goal[2], c='r', s=100, label='Goal Position')
    
    theta = np.linspace(0, 2*np.pi, 100)
    phi_plot = np.linspace(0, np.pi, 50)
    Theta, Phi = np.meshgrid(theta, phi_plot)
    
    r1 = np.sqrt(0.4)
    x_obs1 = 3 + r1 * np.sin(Phi) * np.cos(Theta)
    y_obs1 = 3 + r1 * np.sin(Phi) * np.sin(Theta)
    z_obs1 = 3 + r1 * np.cos(Phi)
    ax.plot_surface(x_obs1, y_obs1, z_obs1, color='r', alpha=0.3, edgecolor='none')
    
    r2 = np.sqrt(0.25)
    x_obs2 = 1.5 + r2 * np.sin(Phi) * np.cos(Theta)
    y_obs2 = 1.5 + r2 * np.sin(Phi) * np.sin(Theta)
    z_obs2 = 2.0 + r2 * np.cos(Phi)
    ax.plot_surface(x_obs2, y_obs2, z_obs2, color='g', alpha=0.3, edgecolor='none')
    
    z_ground = 0.5
    ground_x = np.array([-10, 10, 10, -10])
    ground_y = np.array([-10, -10, 10, 10])
    ground_z = np.array([z_ground]*4)
    ax.plot_trisurf(ground_x, ground_y, ground_z, color='c', alpha=0.1, edgecolor='none')
    
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    ax.set_title('3D Trajectory with Integral Controller and UBF')
    ax.legend()
    plt.show()
    
    plt.figure()
    for order in range(m_order):
        plt.subplot(m_order, 1, order+1)
        plt.plot(time, h_traj[order, :], linewidth=2)
        plt.xlabel('Time (s)')
        plt.ylabel(f'$h^({order+1})$')
        plt.title(f'Universal Barrier Function of Order {order+1}')
        plt.grid(True)
    plt.tight_layout()
    plt.show()
    
    plt.figure()
    for i in range(m):
        plt.subplot(m, 1, i+1)
        plt.plot(time, u_traj[i, :], linewidth=2)
        plt.xlabel('Time (s)')
        plt.ylabel(f'$u_{i+1}$')
        plt.title(f'Control Input u_{i+1} Over Time')
        plt.grid(True)
    plt.tight_layout()
    plt.show()
    
    plt.figure()
    u_norm = np.linalg.norm(u_traj, axis=0)
    plt.plot(time, u_norm, linewidth=2)
    plt.xlabel('Time (s)')
    plt.ylabel('Norm of Control Input')
    plt.title('Norm of Control Input Over Time')
    plt.grid(True)
    plt.show()

# Entry point for running the simulation
def main():
    time, x_traj, u_traj, h_traj = run_simulation()
    visualize_results(time, x_traj, u_traj, h_traj)

if __name__ == "__main__":
    main() 