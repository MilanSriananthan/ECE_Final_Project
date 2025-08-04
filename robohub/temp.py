from mobile_manipulator_unicycle import MobileManipulatorUnicycle
import time
import numpy as np
from scipy.interpolate import CubicHermiteSpline
import matplotlib.pyplot as plt
import csv

# Connect to the robot
robot = MobileManipulatorUnicycle(robot_id=1, backend_server_ip="192.168.0.2")

# Initial movements (from your original code)
start_time = time.time()
while time.time() - start_time < 2.:
    robot.set_mobile_base_speed_and_gripper_power(v=0.5, omega=0.0, gripper_power=0.0)
    robot.set_leds(255, 0, 0)
    time.sleep(0.05)

start_time = time.time()
while time.time() - start_time < 2.:
    robot.set_mobile_base_speed_and_gripper_power(v=-0.5, omega=0.0, gripper_power=0.0)
    robot.set_leds(255, 0, 0)
    time.sleep(0.05)

starting_pose = robot.get_poses()[0]

print("Rotate CCW for 2 seconds and set the LEDs green")
start_time = time.time()
while starting_pose[2] < -np.pi / 16 or starting_pose[2] > np.pi / 16:
    robot.set_mobile_base_speed_and_gripper_power(v=0.0, omega=10.0, gripper_power=0.0)
    robot.set_leds(0, 255, 0)
    time.sleep(0.01)
    starting_pose = robot.get_poses()[0]

# Stop and get starting pose
robot.set_mobile_base_speed_and_gripper_power(0.0, 0.0, 0.0)
time.sleep(2)
starting_pose = robot.get_poses()[0]

# TASK 1: Kinematically feasible path planning
print("=== TASK 1: Path Planning ===")
T = 20.0
N = 40
t_vals = np.linspace(0, T, N)

# Start and goal positions (z_s and z_g)
x_s, y_s, theta_s = starting_pose
print(f"Starting pose: x={x_s:.3f}, y={y_s:.3f}, theta={theta_s:.3f}")

# Set goal position and orientations according to project specs
z_s = np.array([x_s, y_s])
z_g = np.array([x_s + 1.0, y_s + 1.0])  # Arbitrary choice
theta_s_desired = 0.0  # Initial orientation as specified
theta_g = -np.pi / 6   # Final orientation as specified

print(f"Goal: x={z_g[0]:.3f}, y={z_g[1]:.3f}, theta_final={theta_g:.3f}")

# Cubic Hermite spline for path planning
# Initial and final velocities based on orientations
v_mag = 0.1  # Magnitude of velocity
dz_s = v_mag * np.array([np.cos(theta_s_desired), np.sin(theta_s_desired)])
dz_g = v_mag * np.array([np.cos(theta_g), np.sin(theta_g)])

# Create splines for z_d(t) = [x_d(t), y_d(t)]^T
spline_x = CubicHermiteSpline([0, T], [z_s[0], z_g[0]], [dz_s[0], dz_g[0]])
spline_y = CubicHermiteSpline([0, T], [z_s[1], z_g[1]], [dz_s[1], dz_g[1]])

# Get desired flat output and its derivatives
z_d = np.array([spline_x(t_vals), spline_y(t_vals)])  # 2xN array
dz_d = np.array([spline_x(t_vals, 1), spline_y(t_vals, 1)])  # First derivatives
ddz_d = np.array([spline_x(t_vals, 2), spline_y(t_vals, 2)])  # Second derivatives

print("Path planning completed.")

# TASK 2: Feedforward controller (for reference)
print("=== TASK 2: Feedforward Controller ===")
# Compute feedforward inputs using differential flatness (Equation 2)
v_ff = np.sqrt(dz_d[0]**2 + dz_d[1]**2)
omega_ff = np.zeros_like(v_ff)

for i in range(len(omega_ff)):
    denominator = dz_d[0][i]**2 + dz_d[1][i]**2
    if denominator > 1e-6:
        omega_ff[i] = (ddz_d[1][i] * dz_d[0][i] - ddz_d[0][i] * dz_d[1][i]) / denominator
    else:
        omega_ff[i] = 0.0

print("Feedforward controller computed.")

# TASK 3: State Feedback Controller
print("=== TASK 3: State Feedback Controller ===")

# Controller gains (tune these as needed)
K_P = 0.1 * np.eye(2)  # Proportional gain matrix (2x2)
K_D = 0.1 * np.eye(2)  # Derivative gain matrix (2x2)
# 0.5, 1.5
print(f"Controller gains: K_P = {K_P[0,0]}, K_D = {K_D[0,0]}")

# Time step
dt = T / N
start_time = time.time()

# Storage for results
pose_list = []
v_fb_values = []
omega_fb_values = []
time_values = []
z_actual_list = []
dz_actual_list = []
u_w_list = []

# Initialize velocity estimation
prev_x, prev_y = None, None
prev_time = time.time()

print("Starting state feedback control execution...")

# Main control loop implementing Task 3
for i in range(N):
    current_time = time.time()
    
    # Get current pose
    current_pose = robot.get_poses()[0]
    x, y, theta = current_pose
    pose_list.append(np.array(current_pose))
    
    # Current flat output z(t) = [x, y]^T
    z_current = np.array([x, y])
    z_actual_list.append(z_current.copy())
    
    # Estimate current flat output velocity dz(t) using finite differences
    if i > 0 and prev_x is not None:
        actual_dt = current_time - prev_time
        if actual_dt > 0:
            dz_current = np.array([(x - prev_x) / actual_dt, (y - prev_y) / actual_dt])
        else:
            dz_current = np.array([0.0, 0.0])
    else:
        dz_current = np.array([0.0, 0.0])
    
    dz_actual_list.append(dz_current.copy())
    prev_x, prev_y = x, y
    prev_time = current_time
    
    # Current desired values
    z_d_current = z_d[:, i]      # Desired position
    dz_d_current = dz_d[:, i]    # Desired velocity
    ddz_d_current = ddz_d[:, i]  # Desired acceleration
    
    # State feedback controller (Equation 4)
    # u_w(t) = K_P * (z_d(t) - z(t)) + K_D * (dz_d(t) - dz(t))
    position_error = z_d_current - z_current
    velocity_error = dz_d_current - dz_current
    
    u_w = K_P @ position_error + K_D @ velocity_error
    u_w_list.append(u_w.copy())
    
    # The control input u_w represents the desired acceleration ddz
    # So the total desired acceleration is: ddz_d + u_w
    ddz_total = ddz_d_current + u_w
    
    # Total desired velocity (for computing v and omega)
    dz_total = dz_d_current  # We use desired velocity for the transformation
    
    # Convert to unicycle inputs using differential flatness (Equation 2)
    # v(t) = ±√(dz1²+ dz2²)
    v = np.sqrt(dz_total[0]**2 + dz_total[1]**2)
    
    # ω(t) = (ddz2*dz1 - ddz1*dz2)/(dz1² + dz2²)
    denominator = dz_total[0]**2 + dz_total[1]**2
    if denominator > 1e-6:
        omega = (ddz_total[1] * dz_total[0] - ddz_total[0] * dz_total[1]) / denominator
    else:
        omega = 0.0
    
    # Apply reasonable limits
    v = np.clip(v, 0.0, 1.0)
    omega = np.clip(omega, -2.0, 2.0)
    
    # Store values for plotting
    v_fb_values.append(v)
    omega_fb_values.append(omega)
    time_values.append(i * dt)
    
    # Send control commands to robot
    robot.set_mobile_base_speed_and_gripper_power(v=v, omega=omega, gripper_power=0.0)
    robot.set_leds(255, 165, 0)  # Orange for movement
    
    # Maintain timing
    target_time = start_time + (i + 1) * dt
    while time.time() < target_time:
        time.sleep(0.001)
    
    if i % 10 == 0:  # Print progress
        pos_error_norm = np.linalg.norm(position_error)
        vel_error_norm = np.linalg.norm(velocity_error)
        print(f"Step {i}/{N}: pos_err={pos_error_norm:.3f}, vel_err={vel_error_norm:.3f}, v={v:.3f}, ω={omega:.3f}")

# Stop at the end
robot.set_mobile_base_speed_and_gripper_power(0.0, 0.0, 0.0)
robot.set_leds(0, 255, 0)  # Green when done

# Print final results
final_pose = robot.get_poses()[0]
final_error = np.linalg.norm([final_pose[0] - z_g[0], final_pose[1] - z_g[1]])
print(f"\nFinal robot pose: ({final_pose[0]:.3f}, {final_pose[1]:.3f}, {final_pose[2]:.3f})")
print(f"Goal was: ({z_g[0]:.3f}, {z_g[1]:.3f}, {theta_g:.3f})")
print(f"Final position error: {final_error:.3f}m")

# Process collected data
x_actual = [pose[0] for pose in pose_list]
y_actual = [pose[1] for pose in pose_list]
z_actual_array = np.array(z_actual_list).T  # 2xN array
dz_actual_array = np.array(dz_actual_list).T  # 2xN array
u_w_array = np.array(u_w_list).T  # 2xN array

# Create comprehensive plots
plt.figure(figsize=(15, 12))

# Plot 1: Trajectory
plt.subplot(3, 3, 1)
plt.plot(x_actual, y_actual, 'b-', linewidth=2, label='Actual Trajectory')
plt.plot(z_d[0], z_d[1], 'r--', linewidth=2, label='Desired Trajectory')
plt.plot(z_s[0], z_s[1], 'go', markersize=8, label='Start')
plt.plot(z_g[0], z_g[1], 'ro', markersize=8, label='Goal')
plt.xlabel('X position (m)')
plt.ylabel('Y position (m)')
plt.title('Task 3: Actual vs Desired Trajectory')
plt.legend()
plt.grid(True)
plt.axis('equal')

# Plot 2: Position tracking
plt.subplot(3, 3, 2)
plt.plot(time_values, z_d[0][:len(time_values)], 'r--', label='x_desired', linewidth=2)
plt.plot(time_values, [pose[0] for pose in pose_list[:len(time_values)]], 'b-', label='x_actual', linewidth=2)
plt.ylabel('X Position (m)')
plt.title('X Position Tracking')
plt.legend()
plt.grid(True)

plt.subplot(3, 3, 3)
plt.plot(time_values, z_d[1][:len(time_values)], 'r--', label='y_desired', linewidth=2)
plt.plot(time_values, [pose[1] for pose in pose_list[:len(time_values)]], 'b-', label='y_actual', linewidth=2)
plt.ylabel('Y Position (m)')
plt.title('Y Position Tracking')
plt.legend()
plt.grid(True)

# Plot 3: Velocity profiles
plt.subplot(3, 3, 4)
plt.plot(time_values, v_ff[:len(time_values)], 'g--', label='Feedforward v', linewidth=2)
plt.plot(time_values, v_fb_values, 'b-', label='Feedback v', linewidth=2)
plt.ylabel('Linear Velocity (m/s)')
plt.title('Linear Velocity Profile')
plt.legend()
plt.grid(True)

plt.subplot(3, 3, 5)
plt.plot(time_values, omega_ff[:len(time_values)], 'g--', label='Feedforward ω', linewidth=2)
plt.plot(time_values, omega_fb_values, 'b-', label='Feedback ω', linewidth=2)
plt.ylabel('Angular Velocity (rad/s)')
plt.title('Angular Velocity Profile')
plt.legend()
plt.grid(True)

# Plot 4: Control input u_w
plt.subplot(3, 3, 6)
if len(u_w_array) > 0:
    plt.plot(time_values, u_w_array[0][:len(time_values)], 'purple', label='u_w1', linewidth=2)
    plt.plot(time_values, u_w_array[1][:len(time_values)], 'orange', label='u_w2', linewidth=2)
plt.ylabel('Control Input u_w')
plt.title('State Feedback Control Input')
plt.legend()
plt.grid(True)

# Plot 5: Position errors
plt.subplot(3, 3, 7)
pos_errors = []
for i in range(len(time_values)):
    if i < len(z_actual_list):
        error = np.linalg.norm(z_d[:, i] - z_actual_list[i])
        pos_errors.append(error)
plt.plot(time_values[:len(pos_errors)], pos_errors, 'g-', linewidth=2)
plt.xlabel('Time (s)')
plt.ylabel('Position Error (m)')
plt.title('Position Error Over Time')
plt.grid(True)

# Plot 6: Velocity errors  
plt.subplot(3, 3, 8)
vel_errors = []
for i in range(len(time_values)):
    if i < len(dz_actual_list):
        error = np.linalg.norm(dz_d[:, i] - dz_actual_list[i])
        vel_errors.append(error)
plt.plot(time_values[:len(vel_errors)], vel_errors, 'm-', linewidth=2)
plt.xlabel('Time (s)')
plt.ylabel('Velocity Error (m/s)')
plt.title('Velocity Error Over Time')
plt.grid(True)

# Plot 7: Phase portrait (position error vs velocity error)
plt.subplot(3, 3, 9)
if len(pos_errors) > 0 and len(vel_errors) > 0:
    min_len = min(len(pos_errors), len(vel_errors))
    plt.plot(pos_errors[:min_len], vel_errors[:min_len], 'b-', linewidth=2)
    plt.plot(pos_errors[0], vel_errors[0], 'go', markersize=8, label='Start')
    if min_len > 1:
        plt.plot(pos_errors[min_len-1], vel_errors[min_len-1], 'ro', markersize=8, label='End')
plt.xlabel('Position Error (m)')
plt.ylabel('Velocity Error (m/s)')
plt.title('Error Phase Portrait')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()

# Save comprehensive data
with open('task3_feedback_results.csv', mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['Time', 'X_desired', 'Y_desired', 'X_actual', 'Y_actual', 
                    'dX_desired', 'dY_desired', 'dX_actual', 'dY_actual',
                    'V_command', 'Omega_command', 'u_w1', 'u_w2'])
    
    for i in range(len(time_values)):
        row = [time_values[i]]
        
        # Desired position
        row.extend([z_d[0][i], z_d[1][i]])
        
        # Actual position
        if i < len(z_actual_list):
            row.extend([z_actual_list[i][0], z_actual_list[i][1]])
        else:
            row.extend([0, 0])
            
        # Desired velocity
        row.extend([dz_d[0][i], dz_d[1][i]])
        
        # Actual velocity
        if i < len(dz_actual_list):
            row.extend([dz_actual_list[i][0], dz_actual_list[i][1]])
        else:
            row.extend([0, 0])
            
        # Control commands
        row.extend([v_fb_values[i], omega_fb_values[i]])
        
        # Control input u_w
        if i < len(u_w_list):
            row.extend([u_w_list[i][0], u_w_list[i][1]])
        else:
            row.extend([0, 0])
            
        writer.writerow(row)

print("\n=== TASK 3 COMPLETED ===")
print("State feedback controller implemented according to project specifications:")
print("- Used flat output z = [x, y]^T")
print("- Implemented u_w(t) = K_P*(z_d - z) + K_D*(dz_d - dz)")
print("- Converted u_w to v and ω using differential flatness")
print("- Results saved to 'task3_feedback_results.csv'")
print(f"- Final position error: {final_error:.4f}m")