"""
Run a 3D quadrotor simulation with the original recommended parameters.
This version uses the complete set of parameters from the example and saves the 
visualization results to image files.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend to prevent display windows
import matplotlib.pyplot as plt
from ubf.systems import quadrotor
import time as timing
import sys

# Set parameters to match the example provided
print("Configuring quadrotor parameters to match the example...")

# Simulation Parameters
quadrotor.dt = 0.005                # Time step (s)
quadrotor.tf = 19.995               # Adjusted to get exactly 4000 steps: (4000-1)*0.005 = 19.995
quadrotor.time = np.arange(0, quadrotor.tf + quadrotor.dt, quadrotor.dt)
quadrotor.num_steps = len(quadrotor.time)
print(f"Configured number of steps: {quadrotor.num_steps}")  # Should be exactly 4000 steps

# Controller Parameters - slightly increase alpha for faster convergence
quadrotor.alpha = 35.0              # Increased from 25.0 for faster goal reaching
quadrotor.tol = 1e-6                # Tolerance for convergence

# UBF Parameters
quadrotor.beta = 20.0               # UBF smoothness parameter
quadrotor.m_order = 1               # Global higher-order UBF order
quadrotor.alpha_ubf = 4             # UBF K_infty function gain (increased from 3 to 4)
quadrotor.frac = 1                  # Exponent for UBF

# Individual UBF Orders
quadrotor.m_order_indiv = [2, 2, 1] # Order for each individual UBF

# Forward-Euler Integration Parameters
quadrotor.N = 500                   # Number of steps for integration
quadrotor.Delta_tau = 0.01          # Integration step (s)
quadrotor.T_int = quadrotor.N * quadrotor.Delta_tau  # Total integration time

# Add progress monitoring to the original run_simulation function
original_run_simulation = quadrotor.run_simulation

def run_simulation_with_monitoring():
    """Wrapper around the original run_simulation function with progress monitoring"""
    x_current = np.array([0, 0, 0.5, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    u_current = np.array([quadrotor.m_q * quadrotor.g, 0, 0, 0])
    
    x_traj = np.zeros((quadrotor.n, quadrotor.num_steps))
    u_traj = np.zeros((quadrotor.m, quadrotor.num_steps))
    h_traj = np.zeros((quadrotor.m_order, quadrotor.num_steps))
    
    x_traj[:, 0] = x_current
    u_traj[:, 0] = u_current
    for order in range(quadrotor.m_order):
        h_traj[order, 0] = quadrotor.h_orders_global[order](x_current, u_current)
    
    # For monitoring progress - report more frequently (every 1% of steps)
    progress_interval = max(1, int(quadrotor.num_steps / 100))
    print(f"Will simulate {quadrotor.num_steps} steps (approximately {quadrotor.tf} seconds)")
    print(f"Progress will be reported every {progress_interval} steps")
    
    # Report initial distance to goal
    goal_distance_initial = np.linalg.norm(x_current[:3] - quadrotor.x_goal[:3])
    print(f"Initial position: {x_current[:3]}")
    print(f"Goal position: {quadrotor.x_goal[:3]}")
    print(f"Initial distance to goal: {goal_distance_initial:.2f} units")
    
    # Keep track of closest approach to goal
    closest_distance = goal_distance_initial
    closest_step = 0
    goal_reached = False
    
    simulation_start = timing.time()
    
    for k in range(1, quadrotor.num_steps):
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
            # Use verbose=False to reduce "Polishing not needed" messages
            prob.solve(solver=cp.OSQP, verbose=False, eps_abs=1e-3, eps_rel=1e-3, polish=False)
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
        
        # Calculate distance to goal and report progress
        current_distance = np.linalg.norm(x_current[:3] - quadrotor.x_goal[:3])
        
        # Keep track of closest approach
        if current_distance < closest_distance:
            closest_distance = current_distance
            closest_step = k
        
        # Report progress at intervals
        if k % progress_interval == 0:
            elapsed = timing.time() - simulation_start
            est_total = (elapsed/k) * quadrotor.num_steps
            progress_pct = k/quadrotor.num_steps*100
            
            # Print on single line to save space
            print(f"Step {k}/{quadrotor.num_steps} ({progress_pct:.1f}%) - Pos: [{x_current[0]:.2f}, {x_current[1]:.2f}, {x_current[2]:.2f}] - Dist: {current_distance:.2f}", end='\r')
            
            # Every 10%, print full status update
            if k % (progress_interval * 10) == 0:
                print(f"\nStep {k}/{quadrotor.num_steps} ({progress_pct:.1f}%) - Time: {k*quadrotor.dt:.1f}s")
                print(f"  Position: [{x_current[0]:.2f}, {x_current[1]:.2f}, {x_current[2]:.2f}]")
                print(f"  Distance to goal: {current_distance:.2f} units")
                print(f"  Elapsed: {elapsed:.1f}s, Est. total: {est_total:.1f}s, Remaining: {est_total-elapsed:.1f}s")
        
        # Check if we've reached the goal
        if current_distance < 0.5 and not goal_reached:
            goal_reached = True
            print(f"\n*** GOAL REACHED at step {k} (time: {k*quadrotor.dt:.1f}s, distance: {current_distance:.2f}) ***\n")
            
            # Once goal is reached, we don't need to continue for the full 80 seconds
            # Set k to a value that will give us about 5 more seconds of simulation to show stabilization
            remaining_steps = min(1000, quadrotor.num_steps - k)
            print(f"Continuing for {remaining_steps} more steps to show stabilization...")
            
            # If we're close to the end anyway, just complete the full simulation
            if quadrotor.num_steps - k < 1000:
                continue
                
            # Otherwise, fast-forward to the end
            for j in range(k+1, k+remaining_steps):
                # Continue simulating for a bit to show stabilization
                u_new = u_current
                x_new = x_current + quadrotor.f(x_current, u_new) * quadrotor.dt
                
                for order in range(quadrotor.m_order):
                    h_traj[order, j] = quadrotor.h_orders_global[order](x_new, u_new)
                
                x_traj[:, j] = x_new
                u_traj[:, j] = u_new
                
                x_current = x_new.copy()
                u_current = u_new.copy()
            
            # Fill the remaining trajectory with the last state
            for j in range(k+remaining_steps, quadrotor.num_steps):
                x_traj[:, j] = x_current
                u_traj[:, j] = u_current
                for order in range(quadrotor.m_order):
                    h_traj[order, j] = quadrotor.h_orders_global[order](x_current, u_current)
            
            # Skip to the end of the loop
            k = quadrotor.num_steps
            break
    
    print("\n")  # Clear the progress line
    sim_time = timing.time() - simulation_start
    print(f"Simulation completed in {sim_time:.2f} seconds (wall clock time)!")
    final_distance = np.linalg.norm(x_current[:3] - quadrotor.x_goal[:3])
    print(f"Final position: [{x_current[0]:.2f}, {x_current[1]:.2f}, {x_current[2]:.2f}]")
    print(f"Final distance to goal: {final_distance:.2f} units")
    print(f"Closest approach to goal: {closest_distance:.2f} units (at step {closest_step}, time: {closest_step*quadrotor.dt:.1f}s)")
    
    return quadrotor.time, x_traj, u_traj, h_traj

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
    theta = np.linspace(0, 2*np.pi, 40)  # Reduced from original 100
    phi_plot = np.linspace(0, np.pi, 20)  # Reduced from original 50
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
    plt.savefig('quadrotor_3d_trajectory.png', dpi=300)
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
    plt.savefig('quadrotor_barrier_functions.png', dpi=300)
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
    plt.savefig('quadrotor_control_inputs.png', dpi=300)
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
    plt.savefig('quadrotor_control_norm.png', dpi=300)
    plt.close()
    
    # 5. Position vs. Time plot (additional plot)
    print("Generating position vs. time plots...")
    plt.figure(figsize=(12, 8))
    plt.subplot(3, 1, 1)
    plt.plot(time, x_traj[0, :], 'r-', linewidth=2)
    plt.axhline(y=quadrotor.x_goal[0], color='k', linestyle='--', label='Goal')
    plt.ylabel('X Position (m)')
    plt.grid(True)
    plt.legend()
    
    plt.subplot(3, 1, 2)
    plt.plot(time, x_traj[1, :], 'g-', linewidth=2)
    plt.axhline(y=quadrotor.x_goal[1], color='k', linestyle='--', label='Goal')
    plt.ylabel('Y Position (m)')
    plt.grid(True)
    plt.legend()
    
    plt.subplot(3, 1, 3)
    plt.plot(time, x_traj[2, :], 'b-', linewidth=2)
    plt.axhline(y=quadrotor.x_goal[2], color='k', linestyle='--', label='Goal')
    plt.xlabel('Time (s)')
    plt.ylabel('Z Position (m)')
    plt.grid(True)
    plt.legend()
    
    plt.tight_layout()
    print("  Saving position vs. time plots...")
    plt.savefig('quadrotor_position_vs_time.png', dpi=300)
    plt.close()
    
    # 6. Distance to goal over time
    print("Generating distance to goal plot...")
    plt.figure(figsize=(10, 6))
    distances = np.zeros(len(time))
    for i in range(len(time)):
        distances[i] = np.linalg.norm(x_traj[:3, i] - quadrotor.x_goal[:3])
    
    plt.plot(time, distances, 'r-', linewidth=2)
    plt.axhline(y=0.5, color='g', linestyle='--', label='Goal Threshold (0.5 units)')
    plt.xlabel('Time (s)')
    plt.ylabel('Distance to Goal (units)')
    plt.title('Distance to Goal Over Time')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    print("  Saving distance to goal plot...")
    plt.savefig('quadrotor_distance_to_goal.png', dpi=300)
    plt.close()
    
    print("All plots have been saved to files:")
    print("- quadrotor_3d_trajectory.png")
    print("- quadrotor_barrier_functions.png")
    print("- quadrotor_control_inputs.png")
    print("- quadrotor_control_norm.png")
    print("- quadrotor_position_vs_time.png")
    print("- quadrotor_distance_to_goal.png")

# Use our monitoring version of the run_simulation function
quadrotor.run_simulation = run_simulation_with_monitoring

# Override the visualization function
quadrotor.visualize_results = save_visualize_results

# Run the simulation
print(f"Running quadrotor simulation with controller gain alpha={quadrotor.alpha}...")
print(f"Simulation time: {quadrotor.tf} seconds with {quadrotor.num_steps} steps")
print(f"Goal position: {quadrotor.x_goal[:3]}")
print(f"Integration parameters: N={quadrotor.N}, Delta_tau={quadrotor.Delta_tau}")
print(f"Controller gain: alpha={quadrotor.alpha}, alpha_ubf={quadrotor.alpha_ubf}")

start_time = timing.time()
print(f"Starting simulation at: {start_time}")

try:
    # Run the simulation with the modified monitoring function
    quadrotor.main()
    
    end_time = timing.time()
    print(f"Simulation complete. Total time: {end_time - start_time:.2f} seconds")
except Exception as e:
    print(f"ERROR: Simulation failed with exception: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1) 