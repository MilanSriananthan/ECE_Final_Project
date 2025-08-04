from mobile_manipulator_unicycle import MobileManipulatorUnicycle
import time
import numpy as np
from scipy.interpolate import CubicHermiteSpline
import matplotlib.pyplot as plt
import csv

# Connect to the robot
robot = MobileManipulatorUnicycle(robot_id=4, backend_server_ip="192.168.0.2")

# Define Hermite spline parameters
T = 20.0
N = 40  # Number of control steps
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

# Get velocity and angular velocity values from spline
dx = spline_x(t_vals, 1)
dy = spline_y(t_vals, 1)
ddx = spline_x(t_vals, 2)
ddy = spline_y(t_vals, 2)

v_vals = np.sqrt(dx**2 + dy**2)
omega_vals = (ddy * dx - ddx * dy) / (dx**2 + dy**2)

# Time per step
dt = T / N
start_time = time.time()

# LED colors: orange for movement
r, g, b = 255, 165, 0


all_poses = []
pose_list = []

# Main loop
for i in range(N):

    v = v_vals[i]
    omega = omega_vals[i]

    # Send velocity and LED commands
    robot.set_mobile_base_speed_and_gripper_power(v=v, omega=omega, gripper_power=0.0)
    current_pose = robot.get_poses()[0]
    #print(current_pose)
    pose_list.append(np.array(current_pose))

    # Maintain timing
    while time.time() < start_time + (i + 1) * dt:
        time.sleep(0.001)

# Stop at the end
robot.set_mobile_base_speed_and_gripper_power(0.0, 0.0, 0.0) # Green = done

# Print final pose
final_pose = robot.get_poses()[0]
print("Final robot pose:", final_pose)

pose_list = pose_list[1:]
print("HERE")
#print(pose_list)
x = [pose[0] for pose in pose_list]
y = [-pose[1] for pose in pose_list]

print(x)
print(y)

# Plot
plt.figure(figsize=(8,6))
plt.plot(y, x, 'bo-', label='Trajectory')  # Blue dots connected with lines
plt.xlabel('X position')
plt.ylabel('Y position')
plt.title('X vs Y Positions from Poses')
plt.legend()
plt.grid(True)
plt.show(block=True)



rows = zip(y, x)

with open('output_6.csv', mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['X', 'Y'])
    writer.writerows(rows)



plt.figure(figsize=(8,6))
plt.plot(t_vals, v_vals, 'bo-', label='Trajectory')  # Blue dots connected with lines
plt.xlabel('TIME')
plt.ylabel('V')
plt.title('V over Time')
plt.legend()
plt.grid(True)
plt.show(block=True)


plt.figure(figsize=(8,6))
plt.plot(t_vals, omega_vals, 'bo-', label='Trajectory')  # Blue dots connected with lines
plt.xlabel('TIME')
plt.ylabel('W')
plt.title('W over Time')
plt.legend()
plt.grid(True)
plt.show(block=True)


