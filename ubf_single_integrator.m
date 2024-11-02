% Universal Barrier Function based Quadratic Program (UBF-QP) based
% controller for syntheszing safe control inputs under limited actuation
% and for higher order systems


% Author: Vrushabh Zinage

% Clear workspace, command window, and close all figures
clear; clc; close all;

%% =============================== Parameters ===============================

% Simulation Parameters
dt = 0.01;                % Time step size (seconds)
tf = 3;                  % Final simulation time (seconds)
time = 0:dt:tf;           % Time vector
num_steps = length(time); % Number of simulation steps

% Controller Parameters
alpha = 25.0;             % Integral controller gain
max_iter = 10;            % Maximum Newton-Raphson iterations per step
tol = 1e-6;               % Tolerance for Newton-Raphson convergence

% UBF Parameters
beta = 20.0;              % UBF smoothness parameter (higher = closer to max)
m_order = 2;              % Order of the higher-order UBF (m_order >=1)
alpha_ubf = 1;            % UBF K_infty function
frac = 1;                 %Parameter for UBF

% Regularization Parameter
epsilon = 1e-6;           % Small value to avoid singularities in Jacobian

% Forward-Euler Integration Parameters for computing integral control law
N = 55;                   % Number of Forward-Euler steps
Delta_tau = 0.01;         % Step size for tau (seconds)
T = N * Delta_tau;        % Total integration time for composition function c(x,u)

%% ============================ System Definition ============================

% Define the system dynamics


% System dynamics function handle
f = @(x, u) [u(1);u(2)];

% Determine system dimensions
n = length(f([0;0], [0;0]));    % Number of states (e.g., 2)
m = length([0;0]);              % Number of control inputs (e.g., 2)

% Define the goal state
x_goal = [5; 4];                % Example goal state for a 2D system


%% ========================= Safety Constraints (UBFs) ========================

% Define the number of UBFs (safety constraints)
num_ubfs = 3;

% Define UBF functions as function handles
h_funcs = cell(num_ubfs,1);

% Example UBFs:
% h1: Obstacle 1 (circular obstacle)
h_funcs{1} = @(x, u)  (x(1)-3)^2 + (x(2)-3)^2 - 0.4;

% h2: Obstacle 2 (circular obstacle)
h_funcs{2} = @(x, u) (x(1)-1.5)^2 + (x(2)-1.5)^2 - 0.25;

% h3: Control input constraint (e.g., ||u|| <= 2)
h_funcs{3} = @(x, u) (200-u'*u);   


%% ===== Constructing the Universal Barrier Function (UBF) =====

% Define the UBF using the Log-Sum-Exp (LSE) Trick
h_ubf = @(x, u) (log_sum_exp(beta, h_funcs, x, u, num_ubfs) / beta) + (log(num_ubfs)/beta);

% Compute the Jacobians of the UBF with respect to x and u numerically
dh_dx = @(x, u) numerical_jacobian(@(x_val) h_ubf(x_val, u), x, n);
dh_du = @(x, u) numerical_jacobian(@(u_val) h_ubf(x, u_val), u, m);

%% ===================== Integral Controller =====================

% Define the composition function c(x,u) iteratively using Forward-Euler Integration
c = @(x, u) forward_euler_integration(x, u, f, N, Delta_tau);

% Compute the Jacobian dc/du numerically
dc_du = @(x, u) numerical_jacobian_c_u(c, x, u, n, m);


%% ===================== Higher-Order UBFs =====================

% Initialize storage for higher-order UBFs
h_orders = cell(m_order, 1);
h_orders{1} = h_ubf;  % h^1 = h_ubf

for order = 2:m_order
    % Compute dh/dx and dh/du of the previous UBF (h_orders{order-1})
    dh_prev_dx_val = @(x, u) numerical_jacobian(@(x_val) h_orders{order-1}(x_val, u), x, n);
    dh_prev_du_val = @(x, u) numerical_jacobian(@(u_val) h_orders{order-1}(x, u_val), u, m);

    % Compute phi using the integral controller
    phi = @(x, u) alpha * (dc_du(x, u) \ (x_goal - c(x, u)));

    % Compute h_dot = dh/dx * f(x, u) + dh/du * phi(x, u)
    h_dot = @(x, u) dh_prev_dx_val(x, u) * f(x, u) + dh_prev_du_val(x, u) * phi(x, u);

    % Define the new higher-order UBF
    h_orders{order} = @(x, u) h_dot(x, u) + alpha_ubf*(h_orders{order-1}(x,u))^frac;
end


% Define the final higher-order UBF
h_final = h_orders{m_order};

% Compute gradients of h_final w.r.t x and u
dhm_dx = @(x, u) numerical_jacobian(@(x_val) h_final(x_val, u), x, n);
dhm_du = @(x, u) numerical_jacobian(@(u_val) h_final(x, u_val), u, m);

%% ===================== Simulation Setup =====================

% Initialize state and control vectors
 x_current =  [1;1];%zeros(n,1);             % Initial state
% x_current =  zeros(n,1);             % Initial state
u_current = zeros(m,1);             % Initial control inputs

% Preallocate storage for trajectories
x_traj = zeros(n, num_steps);       % State trajectory
u_traj = zeros(m, num_steps);       % Control input trajectory
h_traj = zeros(m_order, num_steps); % Higher-order UBF trajectories

% Set initial conditions
x_traj(:,1) = x_current;
u_traj(:,1) = u_current;
for order = 1:m_order
    h_traj(order,1) = h_orders{order}(x_current, u_current);
end

%% ===================== Simulation Loop =====================

for k = 2:num_steps
    k;
    % Current state and control
    x = x_current;
    u = u_current;
    norm(u_current);

    % Compute phi using the integral controller
    phi = alpha * (dc_du(x, u) \ (x_goal - c(x, u)));
    
    % Compute p(x,u) and q(x,u) for the QP
    p = dhm_du(x, u)';
    q = dhm_dx(x, u) * f(x, u) + dhm_du(x, u) * phi + alpha_ubf*(h_final(x,u))^frac;
    u;
    % h_ubf(x,u)
    % Formulate the QP
    % Minimize: 0.5 * ||v||^2
    % Subject to: p^T * v + q >= 0

    H = eye(m);      % Quadratic term (identity matrix)
    f_qp = zeros(m,1);  % Linear term

    % Inequality constraint: A*v <= b
    A_qp = -p';    % Negative sign because we need p^T v + q >= 0
    b_qp = q;
    % q
    % Ensure that p and q are finite
 % if any(~isfinite(p)) || any(~isfinite(q))
 %      warning('Non-finite values detected in QP constraints at time step %d.', k);
 %      v_star = zeros(m,1);
 %      continue;
 %  end
    
    % Solve the QP for v_star
    options = optimoptions('quadprog', 'Display', 'off');
    [v_star, ~, exitflag] = quadprog(H, f_qp, A_qp, b_qp, [], [], [], [], [], options);
    % 
     if exitflag ~= 1
          warning('QP did not converge at time step %d.', k);
          v_star = zeros(m,1);  % Use zero adjustment if QP fails
      end

    % Update control input using the integral control law
    % du/dt = phi + v_star
    du_dt = phi + v_star;

    u_new = u + du_dt * dt;

    % Update state using Forward Euler Integration
    x_new = x + f(x, u_new) * dt;

    % Update higher-order UBFs
    for order = 1:m_order
        h_traj(order,k) = h_orders{order}(x_new, u_new);
    end

    % Store trajectories
    x_traj(:,k) = x_new;
    u_traj(:,k) = u_new;

    % Update current state and control
    x_current = x_new;
    u_current = u_new;
end

%% ===================== Visualization =====================

% Plot Trajectory (for 2D systems)

    figure;
    plot(x_traj(1,:), x_traj(2,:), 'b-', 'LineWidth', 2); hold on;
    plot(x_goal(1), x_goal(2), 'ro', 'MarkerSize', 10, 'LineWidth', 2);

    % Plot Obstacles
    theta = linspace(0, 2*pi, 100);

    % Obstacle 1
    r1 = sqrt(0.4);
    x_obs1 = 3 + r1 * cos(theta);
    y_obs1 = 3 + r1 * sin(theta);
    fill(x_obs1, y_obs1, 'r', 'FaceAlpha', 0.3, 'EdgeColor', 'k');

    % Obstacle 2
    r2 = sqrt(0.25);
    x_obs2 = 1.5 + r2 * cos(theta);
    y_obs2 = 1.5 + r2 * sin(theta);
    fill(x_obs2, y_obs2, 'g', 'FaceAlpha', 0.3, 'EdgeColor', 'k');

    xlabel('$x_1$', 'Interpreter', 'latex');
    ylabel('$x_2$', 'Interpreter', 'latex');
    title(['Trajectory with Integral Controller and ', num2str(m_order), '-Order UBF'], 'Interpreter', 'latex');
    legend('Trajectory', 'Goal Position', 'Obstacle 1', 'Obstacle 2', 'Interpreter', 'latex');
    grid on;
    axis equal;

% Plot Higher-Order UBFs
figure;
for order = 1:m_order
    subplot(m_order,1,order);
    plot(time, h_traj(order,:), 'LineWidth', 2);
    xlabel('Time (s)', 'Interpreter', 'latex');
    ylabel(['$h^{(', num2str(order), ')}$'], 'Interpreter', 'latex');
    title(['Universal Barrier Function of Order ', num2str(order)], 'Interpreter', 'latex');
    grid on;
end

% Plot Control Inputs Over Time
figure;
for i = 1:m
    subplot(m,1,i);
    plot(time, u_traj(i,:), 'LineWidth', 2);
    xlabel('Time (s)', 'Interpreter', 'latex');
    ylabel(['$u_', num2str(i), '$'], 'Interpreter', 'latex');
    title(['Control Input $u_', num2str(i), '$ Over Time'], 'Interpreter', 'latex');
    grid on;
end

% Plot Control Inputs Over Time
figure;
    plot(time, u_traj(1,:).*u_traj(1,:)+u_traj(2,:).*u_traj(2,:), 'LineWidth', 2);
    xlabel('Time (s)', 'Interpreter', 'latex');
    ylabel('Norm', 'Interpreter', 'latex');
    title(['Norm of Control Input Over Time'], 'Interpreter', 'latex');
    grid on;

%% ===================== Helper Functions =====================

% Function to compute Log-Sum-Exp for UBF
function lse = log_sum_exp(beta, h_funcs, x, u, num_ubfs)
    h_vals = zeros(num_ubfs,1);
    for i = 1:num_ubfs
        h_vals(i) = beta * h_funcs{i}(x, u);
    end
    h_max = max(h_vals);
     sum_exp = sum(exp(-h_vals));
     lse = - log(sum_exp);
    %sum_exp = sum(exp(-h_vals + h_max));
    %lse = h_max - log(sum_exp);
end

% Function to compute numerical Jacobian w.r.t. x or u
function J = numerical_jacobian(f_handle, var, num_vars)
    epsilon_fd = 1e-6;  % Perturbation
    f0 = f_handle(var);
    len_f0 = length(f0);
    J = zeros(len_f0, num_vars);
    for i = 1:num_vars
        var_perturb = var;
        var_perturb(i) = var_perturb(i) + epsilon_fd;
        f1 = f_handle(var_perturb);
        var_perturb(i) = var_perturb(i) - 2 * epsilon_fd;
        f2 = f_handle(var_perturb);
        J(:,i) = (f1 - f2) / (2 * epsilon_fd);
    end
end


% Specialized function for computing dc/du
function J = numerical_jacobian_c_u(c_handle, x, u, n, m)
    epsilon_fd = 1e-6;  % Perturbation
    J = zeros(n, m);
    for i = 1:m
        u_perturb = u;
        u_perturb(i) = u_perturb(i) + epsilon_fd;
        c1 = c_handle(x, u_perturb);
        u_perturb(i) = u_perturb(i) - 2 * epsilon_fd;
        c2 = c_handle(x, u_perturb);
        J(:,i) = (c1 - c2) / (2 * epsilon_fd);
    end
end


% Function to perform Forward-Euler Integration iteratively
function x_final = forward_euler_integration(x_initial, u, f_handle, N_steps, Delta_tau)
    x = x_initial;
    for step = 1:N_steps
        x = x + f_handle(x, u) * Delta_tau;
    end
    x_final = x;
end

