"""
Run an optimized version of the quadrotor simulation and save all plots as image files.
With debug information to identify performance bottlenecks.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend to prevent display windows
import matplotlib.pyplot as plt
from ubf.systems import quadrotor
import time as timing
import sys

# Increase the control gains to make the quadrotor reach the goal faster
print("Setting quadrotor control gains...")
quadrotor.alpha = 50.0     # Increase from 25.0 to 50.0
quadrotor.alpha_ubf = 5.0  # Increase from 3.0 to 5.0

# Override the visualize_results function to save plots to files
original_visualize = quadrotor.visualize_results

def save_visualize_results(time, x_traj, u_traj, h_traj):
    """
    Modified visualization function that saves plots to files instead of displaying them.
    """
    print("Starting to generate plots...")
    
    # 1. 3D Trajectory plot
    print("Generating 3D trajectory plot...")
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(x_traj[0, :], x_traj[1, :], x_traj[2, :], 'b-', linewidth=2, label='Trajectory')
    ax.scatter(quadrotor.x_goal[0], quadrotor.x_goal[1], quadrotor.x_goal[2], c='r', s=100, label='Goal Position')
    
    # Use fewer points in the mesh grid for faster plotting
    theta = np.linspace(0, 2*np.pi, 20)  # Reduced from 100
    phi_plot = np.linspace(0, np.pi, 10)  # Reduced from 50
    Theta, Phi = np.meshgrid(theta, phi_plot)
    
    r1 = np.sqrt(0.4)
    x_obs1 = 3 + r1 * np.sin(Phi) * np.cos(Theta)
    y_obs1 = 3 + r1 * np.sin(Phi) * np.sin(Theta)
    z_obs1 = 3 + r1 * np.cos(Phi)
    print("  Adding obstacle 1...")
    ax.plot_surface(x_obs1, y_obs1, z_obs1, color='r', alpha=0.3, edgecolor='none')
    
    r2 = np.sqrt(0.25)
    x_obs2 = 1.5 + r2 * np.sin(Phi) * np.cos(Theta)
    y_obs2 = 1.5 + r2 * np.sin(Phi) * np.sin(Theta)
    z_obs2 = 2.0 + r2 * np.cos(Phi)
    print("  Adding obstacle 2...")
    ax.plot_surface(x_obs2, y_obs2, z_obs2, color='g', alpha=0.3, edgecolor='none')
    
    z_ground = 0.5
    ground_x = np.array([-10, 10, 10, -10])
    ground_y = np.array([-10, -10, 10, 10])
    ground_z = np.array([z_ground]*4)
    print("  Adding ground surface...")
    ax.plot_trisurf(ground_x, ground_y, ground_z, color='c', alpha=0.1, edgecolor='none')
    
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    ax.set_title('3D Trajectory with Integral Controller and UBF')
    ax.legend()
    plt.tight_layout()
    print("  Saving 3D trajectory plot...")
    plt.savefig('quadrotor_3d_trajectory.png')
    plt.close()
    
    # 2. Barrier Function Values
    print("Generating barrier function plots...")
    plt.figure(figsize=(12, 8))
    for order in range(quadrotor.m_order):
        plt.subplot(quadrotor.m_order, 1, order+1)
        plt.plot(time, h_traj[order, :], linewidth=2)
        plt.xlabel('Time (s)')
        plt.ylabel(f'$h^({order+1})$')
        plt.title(f'Universal Barrier Function of Order {order+1}')
        plt.grid(True)
    plt.tight_layout()
    print("  Saving barrier function plots...")
    plt.savefig('quadrotor_barrier_functions.png')
    plt.close()
    
    # 3. Control Inputs
    print("Generating control input plots...")
    plt.figure(figsize=(12, 10))
    for i in range(quadrotor.m):
        plt.subplot(quadrotor.m, 1, i+1)
        plt.plot(time, u_traj[i, :], linewidth=2)
        plt.xlabel('Time (s)')
        plt.ylabel(f'$u_{i+1}$')
        plt.title(f'Control Input u_{i+1} Over Time')
        plt.grid(True)
    plt.tight_layout()
    print("  Saving control input plots...")
    plt.savefig('quadrotor_control_inputs.png')
    plt.close()
    
    # 4. Control Input Norm
    print("Generating control norm plot...")
    plt.figure(figsize=(10, 6))
    u_norm = np.linalg.norm(u_traj, axis=0)
    plt.plot(time, u_norm, linewidth=2)
    plt.xlabel('Time (s)')
    plt.ylabel('Norm of Control Input')
    plt.title('Norm of Control Input Over Time')
    plt.grid(True)
    plt.tight_layout()
    print("  Saving control norm plot...")
    plt.savefig('quadrotor_control_norm.png')
    plt.close()
    
    print("All plots have been saved to files:")
    print("- quadrotor_3d_trajectory.png")
    print("- quadrotor_barrier_functions.png")
    print("- quadrotor_control_inputs.png")
    print("- quadrotor_control_norm.png")

# Fix the quadrotor simulation function to have a timeout
original_run_simulation = quadrotor.run_simulation

def run_simulation_with_timeout():
    """Wrapper around the original run_simulation function with progress updates"""
    print("Starting simulation with progress updates...")
    
    x_current = np.array([0, 0, 0.5, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    u_current = np.array([quadrotor.m_q * quadrotor.g, 0, 0, 0])
    
    x_traj = np.zeros((quadrotor.n, quadrotor.num_steps))
    u_traj = np.zeros((quadrotor.m, quadrotor.num_steps))
    h_traj = np.zeros((quadrotor.m_order, quadrotor.num_steps))
    
    x_traj[:, 0] = x_current
    u_traj[:, 0] = u_current
    for order in range(quadrotor.m_order):
        h_traj[order, 0] = quadrotor.h_orders_global[order](x_current, u_current)
    
    # For longer simulation, we'll use all steps
    # Report progress every 5% of total steps
    max_steps = quadrotor.num_steps
    progress_interval = max(1, int(max_steps * 0.05))
    print(f"Will simulate {max_steps} steps (approximately {quadrotor.tf} seconds)")
    print(f"Progress will be reported every {progress_interval} steps")
    
    # Report distance to goal at intervals
    goal_distance_initial = np.linalg.norm(x_current[:3] - quadrotor.x_goal[:3])
    print(f"Initial distance to goal: {goal_distance_initial:.2f} units")
    print(f"Goal position: {quadrotor.x_goal[:3]}")
    
    # Keep track of closest approach to goal
    closest_distance = goal_distance_initial
    closest_step = 0
    goal_reached = False
    
    simulation_start = timing.time()
    for k in range(1, max_steps):
        if k % progress_interval == 0 or k % 500 == 0:
            # Calculate distance to goal for position components only
            current_distance = np.linalg.norm(x_current[:3] - quadrotor.x_goal[:3])
            
            if current_distance < closest_distance:
                closest_distance = current_distance
                closest_step = k
            
            elapsed = timing.time() - simulation_start
            est_total = (elapsed/k) * max_steps
            print(f"Simulation step {k}/{max_steps} ({k/max_steps*100:.1f}%) - Distance to goal: {current_distance:.2f} units")
            print(f"  Current position: {x_current[:3]}")
            print(f"  Elapsed: {elapsed:.1f}s, Est. total: {est_total:.1f}s, Remaining: {est_total-elapsed:.1f}s")
            
        x = x_current.copy()
        u = u_current.copy()
        
        A = quadrotor.dc_du(x, u)
        b = quadrotor.x_goal - quadrotor.c_func(x, u)
        phi_sol, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
        phi_val = quadrotor.alpha * phi_sol
    
        # Use the Jacobian from h_final to set up the QP.
        p = quadrotor.dhm_du(x, u)  # Should have shape (1, m)
        q = np.dot(quadrotor.dhm_dx(x, u), quadrotor.f(x, u)) + np.dot(quadrotor.dhm_du(x, u), phi_val) + quadrotor.alpha_ubf * (quadrotor.h_final(x, u) ** quadrotor.frac)
        
        import cvxpy as cp
        v = cp.Variable(quadrotor.m)
        objective = cp.Minimize(0.5 * cp.sum_squares(v))
        constraints = [p @ v + q >= 0]
        prob = cp.Problem(objective, constraints)
        
        try:
            prob.solve(solver=cp.OSQP, verbose=False)
            if v.value is None:
                print(f"Warning: QP did not converge at step {k}")
                v_star = np.zeros(quadrotor.m)
            else:
                v_star = v.value
        except Exception as e:
            print(f"QP solver error at step {k}: {e}")
            v_star = np.zeros(quadrotor.m)
    
        du_dt = phi_val + v_star
        u_new = u + du_dt * quadrotor.dt
    
        x_new = x + quadrotor.f(x, u_new) * quadrotor.dt
    
        for order in range(quadrotor.m_order):
            h_traj[order, k] = quadrotor.h_orders_global[order](x_new, u_new)
        
        x_traj[:, k] = x_new
        u_traj[:, k] = u_new
        
        x_current = x_new.copy()
        u_current = u_new.copy()
        
        # Check if we've reached close to the goal
        current_distance = np.linalg.norm(x_current[:3] - quadrotor.x_goal[:3])
        if current_distance < 0.5 and not goal_reached:
            goal_reached = True
            print(f"\n*** GOAL REACHED at step {k} (distance: {current_distance:.2f}) ***\n")
            # Continue simulation to show stabilization
            
        # Emergency stop if moving too far from goal
        if current_distance > 3 * goal_distance_initial:
            print(f"WARNING: Quadrotor diverging - distance {current_distance:.2f} exceeds 3x initial distance")
            print(f"Current position: {x_current[:3]}")
            break
    
    sim_time = timing.time() - simulation_start
    print(f"Simulation completed in {sim_time:.2f} seconds!")
    final_distance = np.linalg.norm(x_current[:3] - quadrotor.x_goal[:3])
    print(f"Final distance to goal: {final_distance:.2f} units")
    print(f"Closest approach to goal: {closest_distance:.2f} units (at step {closest_step})")
    
    return quadrotor.time, x_traj, u_traj, h_traj

# Modify parameters to make the simulation run for longer while still being efficient
# 1. Increase simulation time to allow reaching the goal
quadrotor.tf = 15  # Reduced from 25 seconds to 15 seconds for faster completion
quadrotor.time = np.arange(0, quadrotor.tf + quadrotor.dt, quadrotor.dt)
quadrotor.num_steps = len(quadrotor.time)

# 2. Keep integration steps manageable (original value is 615)
quadrotor.N = 30  # Maintain 30 steps
quadrotor.T_int = quadrotor.N * quadrotor.Delta_tau

# 3. Override the visualization and simulation functions
quadrotor.visualize_results = save_visualize_results

# Run the simulation
print("Running optimized quadrotor simulation with higher control gains...")
start_time = timing.time()
print(f"Starting simulation at: {start_time}")

try:
    # Override the main function to use our patched run_simulation
    original_main = quadrotor.main
    
    def patched_main():
        time_vec, x_traj, u_traj, h_traj = run_simulation_with_timeout()
        save_visualize_results(time_vec, x_traj, u_traj, h_traj)
    
    quadrotor.main = patched_main
    quadrotor.main()
    
    end_time = timing.time()
    print(f"Simulation complete. Total time: {end_time - start_time:.2f} seconds")
except Exception as e:
    print(f"ERROR: Simulation failed with exception: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1) 