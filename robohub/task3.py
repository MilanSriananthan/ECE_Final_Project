from mobile_manipulator_unicycle import MobileManipulatorUnicycle
import time
import numpy as np
from scipy.interpolate import CubicHermiteSpline
import matplotlib.pyplot as plt
import csv

# Connect to the robot
robot = MobileManipulatorUnicycle(robot_id=7, backend_server_ip="192.168.0.2")

# Define Hermite spline parameters
T = 20.0
N = 100  # Increased for better velocity estimation
t_vals = np.linspace(0, T, N)

# Start and goal
x_s, y_s, theta_s = 0.0, 0.0, 0.0
x_g, y_g, theta_g = 1.0, 1.0, -np.pi / 6
v_mag = 0.1

dx_s = v_mag * np.cos(theta_s)
dy_s = v_mag * np.sin(theta_s)
dx_g = v_mag * np.cos(theta_g)
dy_g = v_mag * np.sin(theta_g)

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

# Feedforward control inputs
v_ff = np.sqrt(dx_d**2 + dy_d**2)
omega_ff = (ddy_d * dx_d - ddx_d * dy_d) / (dx_d**2 + dy_d**2)

# Time per step
dt = T / N
start_time = time.time()

# Feedback controller gains (tune these as needed)
Kp = 2.0  # Proportional gain
Kd = 1.0  # Derivative gain

# LED colors: orange for movement
r, g, b = 255, 165, 0

# Storage for results
all_poses = []
pose_list = []
v_fb_values = []
omega_fb_values = []
time_values = []

# Initialize variables for velocity estimation
prev_x, prev_y = None, None

# Main loop
for i in range(N):
    # Get current pose
    current_pose = robot.get_poses()[0]
    x, y, theta = current_pose
    pose_list.append(np.array(current_pose))
    
    # Estimate current velocity using finite differences
    if i > 0 and prev_x is not None:
        dx = (x - prev_x) / dt
        dy = (y - prev_y) / dt
    else:
        dx, dy = 0.0, 0.0
    
    prev_x, prev_y = x, y
    
    # Calculate position error
    pos_error_x = x_d[i] - x
    pos_error_y = y_d[i] - y
    
    # Calculate velocity error
    vel_error_x = dx_d[i] - dx
    vel_error_y = dy_d[i] - dy
    
    # Compute feedback control in flat output space
    ux = Kp * pos_error_x + Kd * vel_error_x
    uy = Kp * pos_error_y + Kd * vel_error_y
    
    # Convert to v and omega using differential flatness
    # Note: We use the desired velocities in the transformation
    # to avoid singularities when current velocity is near zero
    v = v_ff[i] + np.sqrt(ux**2 + uy**2) * np.sign(v_ff[i])
    omega = omega_ff[i] + (ux * dy_d[i] - uy * dx_d[i]) / (dx_d[i]**2 + dy_d[i]**2 + 1e-5)
    
    # Store values for plotting
    v_fb_values.append(v)
    omega_fb_values.append(omega)
    time_values.append(i * dt)
    
    # Send control commands to robot
    robot.set_mobile_base_speed_and_gripper_power(v=v, omega=omega, gripper_power=0.0)
    
    # Maintain timing
    while time.time() < start_time + (i + 1) * dt:
        time.sleep(0.001)

# Stop at the end
robot.set_mobile_base_speed_and_gripper_power(0.0, 0.0, 0.0)

# Print final pose
final_pose = robot.get_poses()[0]
print("Final robot pose:", final_pose)

# Process collected data
x_actual = [pose[0] for pose in pose_list]
y_actual = [pose[1] for pose in pose_list]

# Plot actual vs desired trajectory
plt.figure(figsize=(10, 8))
plt.plot(y_actual, x_actual, 'b-', label='Actual Trajectory')
plt.plot(y_d, x_d, 'r--', label='Desired Trajectory')
plt.xlabel('Y position')
plt.ylabel('X position')
plt.title('Actual vs Desired Trajectory')
plt.legend()
plt.grid(True)
plt.show(block=True)

# Save trajectory data
with open('feedback_trajectory.csv', mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['X_desired', 'Y_desired', 'X_actual', 'Y_actual'])
    for i in range(len(x_d)):
        writer.writerow([x_d[i], y_d[i], x_actual[i], y_actual[i]])

# Plot velocities
plt.figure(figsize=(12, 6))
plt.subplot(2, 1, 1)
plt.plot(time_values, v_ff[:len(time_values)], 'r--', label='Feedforward v')
plt.plot(time_values, v_fb_values, 'b-', label='Feedback v')
plt.ylabel('Linear Velocity (v)')
plt.title('Velocity Profiles')
plt.legend()
plt.grid(True)

plt.subplot(2, 1, 2)
plt.plot(time_values, omega_ff[:len(time_values)], 'r--', label='Feedforward ω')
plt.plot(time_values, omega_fb_values, 'b-', label='Feedback ω')
plt.xlabel('Time (s)')
plt.ylabel('Angular Velocity (ω)')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show(block=True)

# Plot position errors
pos_errors = [np.sqrt((x_d[i]-x_actual[i])**2 + (y_d[i]-y_actual[i])**2) 
             for i in range(len(x_actual))]
plt.figure(figsize=(10, 6))
plt.plot(time_values, pos_errors)
plt.xlabel('Time (s)')
plt.ylabel('Position Error (m)')
plt.title('Position Error Over Time')
plt.grid(True)
plt.show(block=True)