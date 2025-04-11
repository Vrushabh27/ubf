"""
Run an optimized version of the quadrotor simulation and save all plots as image files.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend to prevent display windows
import matplotlib.pyplot as plt
from ubf.systems import quadrotor

# Override the visualize_results function to save plots to files
original_visualize = quadrotor.visualize_results

def save_visualize_results(time, x_traj, u_traj, h_traj):
    """
    Modified visualization function that saves plots to files instead of displaying them.
    """
    # 1. 3D Trajectory plot
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(x_traj[0, :], x_traj[1, :], x_traj[2, :], 'b-', linewidth=2, label='Trajectory')
    ax.scatter(quadrotor.x_goal[0], quadrotor.x_goal[1], quadrotor.x_goal[2], c='r', s=100, label='Goal Position')
    
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
    plt.tight_layout()
    plt.savefig('quadrotor_3d_trajectory.png')
    plt.close()
    
    # 2. Barrier Function Values
    plt.figure(figsize=(12, 8))
    for order in range(quadrotor.m_order):
        plt.subplot(quadrotor.m_order, 1, order+1)
        plt.plot(time, h_traj[order, :], linewidth=2)
        plt.xlabel('Time (s)')
        plt.ylabel(f'$h^({order+1})$')
        plt.title(f'Universal Barrier Function of Order {order+1}')
        plt.grid(True)
    plt.tight_layout()
    plt.savefig('quadrotor_barrier_functions.png')
    plt.close()
    
    # 3. Control Inputs
    plt.figure(figsize=(12, 10))
    for i in range(quadrotor.m):
        plt.subplot(quadrotor.m, 1, i+1)
        plt.plot(time, u_traj[i, :], linewidth=2)
        plt.xlabel('Time (s)')
        plt.ylabel(f'$u_{i+1}$')
        plt.title(f'Control Input u_{i+1} Over Time')
        plt.grid(True)
    plt.tight_layout()
    plt.savefig('quadrotor_control_inputs.png')
    plt.close()
    
    # 4. Control Input Norm
    plt.figure(figsize=(10, 6))
    u_norm = np.linalg.norm(u_traj, axis=0)
    plt.plot(time, u_norm, linewidth=2)
    plt.xlabel('Time (s)')
    plt.ylabel('Norm of Control Input')
    plt.title('Norm of Control Input Over Time')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('quadrotor_control_norm.png')
    plt.close()
    
    print("All plots have been saved to files:")
    print("- quadrotor_3d_trajectory.png")
    print("- quadrotor_barrier_functions.png")
    print("- quadrotor_control_inputs.png")
    print("- quadrotor_control_norm.png")

# Modify parameters to make the simulation run faster
# 1. Reduce simulation time
quadrotor.tf = 10  
quadrotor.time = np.arange(0, quadrotor.tf + quadrotor.dt, quadrotor.dt)
quadrotor.num_steps = len(quadrotor.time)

# 2. Reduce integration steps (original value is 615)
quadrotor.N = 100
quadrotor.T_int = quadrotor.N * quadrotor.Delta_tau

# Override the visualization function
quadrotor.visualize_results = save_visualize_results

# Run the simulation
print("Running optimized quadrotor simulation with file output...")
quadrotor.main()
print("Simulation complete.") 