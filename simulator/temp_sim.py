from mobile_manipulator_unicycle_sim import MobileManipulatorUnicycleSim
import time
import numpy as np
from scipy.interpolate import CubicHermiteSpline
import matplotlib.pyplot as plt
import csv

# Connect to the robot
robot = MobileManipulatorUnicycleSim(robot_id=4, backend_server_ip="192.168.0.2")

start_time = time.time()
while time.time() - start_time < 2.:
    robot.set_mobile_base_speed_and_gripper_power(v=0.5, omega=0.0, gripper_power=0.0)
    robot.set_leds(255, 0, 0)
    time.sleep(0.05)

# Move backward
start_time = time.time()
while time.time() - start_time < 2.:
    robot.set_mobile_base_speed_and_gripper_power(v=-0.5, omega=0.0, gripper_power=0.0)
    robot.set_leds(255, 0, 0)
    time.sleep(0.05)

# Stop and get starting pose
robot.set_mobile_base_speed_and_gripper_power(0.0, 0.0, 0.0)
time.sleep(2)
starting_pose = robot.get_poses()[0]

# Define Hermite spline parameters
T = 20.0
N = 100
t_vals = np.linspace(0, T, N)

# Start and goal
print("Starting pose:", starting_pose)
x_s, y_s, theta_s = starting_pose
print(f"Start: x={x_s:.3f}, y={y_s:.3f}, theta={theta_s:.3f}")

# Goal position and orientation
x_g, y_g, theta_g = x_s + 1.0, y_s + 1.0, theta_s - np.pi / 6
v_mag = 0.1

# Initial and final velocities in global frame
dx_s = v_mag * np.cos(theta_s)
dy_s = v_mag * np.sin(theta_s)
dx_g = v_mag * np.cos(theta_g)
dy_g = v_mag * np.sin(theta_g)

print(f"Goal: x={x_g:.3f}, y={y_g:.3f}, theta={theta_g:.3f}")

# Create Hermite splines
spline_x = CubicHermiteSpline([0, T], [x_s, x_g], [dx_s, dx_g])
spline_y = CubicHermiteSpline([0, T], [y_s, y_g], [dy_s, dy_g])

# Get desired values from spline
x_d = spline_x(t_vals)
y_d = spline_y(t_vals)
dx_d = spline_x(t_vals, 1)
dy_d = spline_y(t_vals, 1)
ddx_d = spline_x(t_vals, 2)
ddy_d = spline_y(t_vals, 2)

# Compute desired orientation from velocity direction
theta_d = np.arctan2(dy_d, dx_d)

# Feedforward control inputs
v_ff = np.sqrt(dx_d**2 + dy_d**2)

# Compute angular velocity using proper differential flatness relationship
# For a unicycle: omega = (ddx_d * sin(theta) - ddy_d * cos(theta)) / v
# But we'll use the cross product formula which is more numerically stable
omega_ff = np.zeros_like(v_ff)
for i in range(len(omega_ff)):
    if v_ff[i] > 1e-4:  # Avoid division by small numbers
        omega_ff[i] = (dx_d[i] * ddy_d[i] - dy_d[i] * ddx_d[i]) / (dx_d[i]**2 + dy_d[i]**2)
    else:
        omega_ff[i] = 0.0

# Time per step
dt = T / N
start_time = time.time()

# Feedback controller gains (reduced for stability)
Kp_pos = 1.5  # Position gain
Kd_pos = 0.8  # Velocity gain
Kp_theta = 2.0  # Orientation gain

# Storage for results
pose_list = []
v_fb_values = []
omega_fb_values = []
time_values = []

# Initialize variables for velocity estimation
prev_x, prev_y, prev_theta = None, None, None
prev_time = time.time()

print("Starting trajectory execution...")

# Main control loop
for i in range(N):
    current_time = time.time()
    
    # Get current pose
    current_pose = robot.get_poses()[0]
    x, y, theta = current_pose
    pose_list.append(np.array(current_pose))
    
    # Estimate current velocities using finite differences
    if i > 0 and prev_x is not None:
        actual_dt = current_time - prev_time
        if actual_dt > 0:
            dx_actual = (x - prev_x) / actual_dt
            dy_actual = (y - prev_y) / actual_dt
            dtheta_actual = (theta - prev_theta) / actual_dt
            # Wrap angle difference
            while dtheta_actual > np.pi:
                dtheta_actual -= 2*np.pi
            while dtheta_actual < -np.pi:
                dtheta_actual += 2*np.pi
        else:
            dx_actual, dy_actual, dtheta_actual = 0.0, 0.0, 0.0
    else:
        dx_actual, dy_actual, dtheta_actual = 0.0, 0.0, 0.0
    
    prev_x, prev_y, prev_theta = x, y, theta
    prev_time = current_time
    
    # Position errors
    ex = x_d[i] - x
    ey = y_d[i] - y
    
    # Velocity errors
    edx = dx_d[i] - dx_actual
    edy = dy_d[i] - dy_actual
    
    # Orientation error
    etheta = theta_d[i] - theta
    # Wrap angle error
    while etheta > np.pi:
        etheta -= 2*np.pi
    while etheta < -np.pi:
        etheta += 2*np.pi
    
    # Feedback control in Cartesian space
    ux = Kp_pos * ex + Kd_pos * edx
    uy = Kp_pos * ey + Kd_pos * edy
    
    # Transform feedback to robot frame
    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)
    
    # Project Cartesian feedback onto robot's local frame
    u_forward = ux * cos_theta + uy * sin_theta
    u_lateral = -ux * sin_theta + uy * cos_theta
    
    # Compute control inputs
    v = v_ff[i] + u_forward
    omega = omega_ff[i] + Kp_theta * etheta + u_lateral / max(0.1, abs(v))  # Lateral error becomes angular correction
    
    # Apply reasonable limits
    v = np.clip(v, -1.0, 1.0)
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
        time.sleep(0.001)  # Shorter sleep for better timing
    
    if i % 10 == 0:  # Print progress
        print(f"Step {i}/{N}: pos=({x:.3f}, {y:.3f}), theta={theta:.3f}, v={v:.3f}, ω={omega:.3f}")

# Stop at the end
robot.set_mobile_base_speed_and_gripper_power(0.0, 0.0, 0.0)
robot.set_leds(0, 255, 0)  # Green when done

# Print final results
final_pose = robot.get_poses()[0]
print("Final robot pose:", final_pose)
print(f"Goal was: ({x_g:.3f}, {y_g:.3f}, {theta_g:.3f})")
print(f"Final error: pos={np.sqrt((final_pose[0]-x_g)**2 + (final_pose[1]-y_g)**2):.3f}m, theta={abs(final_pose[2]-theta_g):.3f}rad")

# Process collected data
x_actual = [pose[0] for pose in pose_list]
y_actual = [pose[1] for pose in pose_list]

# Plot actual vs desired trajectory
plt.figure(figsize=(12, 10))

# Trajectory plot
plt.subplot(2, 2, 1)
plt.plot(x_actual, y_actual, 'b-', linewidth=2, label='Actual Trajectory')
plt.plot(x_d, y_d, 'r--', linewidth=2, label='Desired Trajectory')
plt.plot(x_s, y_s, 'go', markersize=8, label='Start')
plt.plot(x_g, y_g, 'ro', markersize=8, label='Goal')
plt.xlabel('X position (m)')
plt.ylabel('Y position (m)')
plt.title('Actual vs Desired Trajectory')
plt.legend()
plt.grid(True)
plt.axis('equal')

# Velocity profiles
plt.subplot(2, 2, 2)
plt.plot(time_values, v_ff[:len(time_values)], 'r--', label='Feedforward v', linewidth=2)
plt.plot(time_values, v_fb_values, 'b-', label='Actual v', linewidth=2)
plt.ylabel('Linear Velocity (m/s)')
plt.title('Linear Velocity Profile')
plt.legend()
plt.grid(True)

plt.subplot(2, 2, 3)
plt.plot(time_values, omega_ff[:len(time_values)], 'r--', label='Feedforward ω', linewidth=2)
plt.plot(time_values, omega_fb_values, 'b-', label='Actual ω', linewidth=2)
plt.xlabel('Time (s)')
plt.ylabel('Angular Velocity (rad/s)')
plt.title('Angular Velocity Profile')
plt.legend()
plt.grid(True)

# Position error
plt.subplot(2, 2, 4)
pos_errors = [np.sqrt((x_d[i]-x_actual[i])**2 + (y_d[i]-y_actual[i])**2) 
             for i in range(len(x_actual))]
plt.plot(time_values, pos_errors, 'g-', linewidth=2)
plt.xlabel('Time (s)')
plt.ylabel('Position Error (m)')
plt.title('Position Error Over Time')
plt.grid(True)

plt.tight_layout()
plt.show(block=True)

# Save trajectory data
with open('feedback_trajectory.csv', mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['Time', 'X_desired', 'Y_desired', 'Theta_desired', 'X_actual', 'Y_actual', 'Theta_actual', 'V_command', 'Omega_command'])
    for i in range(len(x_d)):
        if i < len(x_actual):
            writer.writerow([time_values[i] if i < len(time_values) else i*dt, 
                           x_d[i], y_d[i], theta_d[i], 
                           x_actual[i], y_actual[i], pose_list[i][2],
                           v_fb_values[i] if i < len(v_fb_values) else 0,
                           omega_fb_values[i] if i < len(omega_fb_values) else 0])

print("Trajectory data saved to feedback_trajectory.csv")
print("Execution complete!")