from mobile_manipulator_unicycle_sim import MobileManipulatorUnicycleSim
import time
import random
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import CubicHermiteSpline


T = 20.0
t_vals = np.linspace(0, T, 40)

# Start and goal
x_s, y_s, theta_s = 0.0, 0.0, 0.0
x_g, y_g, theta_g = 1.0, 1.0, -np.pi / 6
v_mag = 0.1

dx_s = v_mag * np.cos(theta_s)
dy_s = v_mag * np.sin(theta_s)
dx_g = v_mag * np.cos(theta_g)
dy_g = v_mag * np.sin(theta_g)

spline_x = CubicHermiteSpline([0, T], [x_s, x_g], [dx_s, dx_g])
spline_y = CubicHermiteSpline([0, T], [y_s, y_g], [dy_s, dy_g])

# Derivatives
dx = spline_x(t_vals, 1)
dy = spline_y(t_vals, 1)
ddx = spline_x(t_vals, 2)
ddy = spline_y(t_vals, 2)

v_vals = np.sqrt(dx**2 + dy**2)
omega_vals = (ddy * dx - ddx * dy) / (dx**2 + dy**2)

# Two options to initialize the simulator
# (i) random robot and object locations
#robot = MobileManipulatorUnicycleSim(robot_id=1, backend_server_ip=None)
# (ii) specified robot and object locations
robot = MobileManipulatorUnicycleSim(robot_id=1, backend_server_ip=None, robot_pose=[x_s, y_s, theta_s], pickup_location=[-0.5, 1.75, 0.], dropoff_location=[-1.5, -1.5, 0.], obstacles_location=[[-2.0, -1.0, 0.0], [-1.6, -1.0, 0.0], [-1.2, -1.0, 0.0], [-0.8, 1.0, 0.0], [-0.4, 1.0, 0.0], [0.0, 1.0, 0.0], [0.4, 2.0, 0.0], [0.8, 2.0, 0.0], [1.2, 2.0, 0.0]])


'''
print("Move forward for 2 seconds")
start_time = time.time()
while time.time() - start_time < 2.:
    robot.set_mobile_base_speed_and_gripper_power(v=0.5, omega=0.0, gripper_power=0.0)
    time.sleep(0.05)

print("Move backward for 2 seconds")
start_time = time.time()
while time.time() - start_time < 2.:
    robot.set_mobile_base_speed_and_gripper_power(v=-0.5, omega=0.0, gripper_power=0.0)
    time.sleep(0.05)

print("Rotate CCW for 2 seconds")
start_time = time.time()
while time.time() - start_time < 2.:
    robot.set_mobile_base_speed_and_gripper_power(v=0.0, omega=10.0, gripper_power=0.0)
    time.sleep(0.05)

print("Rotate CW for 2 seconds")
start_time = time.time()
while time.time() - start_time < 2.:
    robot.set_mobile_base_speed_and_gripper_power(v=0.0, omega=-10.0, gripper_power=0.0)
    time.sleep(0.05)

print("Stop the base and the gripper")
robot.set_mobile_base_speed_and_gripper_power(0., 0., 0.)

print("Get the robot's current pose")
poses = robot.get_poses()
print(f"Robot, pickup, dropoff, obstacles poses: {poses}")
'''

print("Running feedforward control...")
all_poses = []
pose_list = []

dt = T / len(t_vals)
print(len(t_vals))
for v, omega in zip(v_vals, omega_vals):
    robot.set_mobile_base_speed_and_gripper_power(v=v, omega=omega, gripper_power=0.0)
    time.sleep(dt)
    current_pose = robot.get_poses()[0]
    #print(current_pose)
    pose_list.append(np.array(current_pose))

# Stop at the end
robot.set_mobile_base_speed_and_gripper_power(0.0, 0.0, 0.0)

# Print final pose
#print(pose_list)


x = [pose[0] for pose in pose_list]
y = [pose[1] for pose in pose_list]

#print(x)
#print(y)

# Plot
plt.figure(figsize=(8,6))
plt.plot(x, y, 'bo-', label='Trajectory')  # Blue dots connected with lines
plt.xlabel('X position')
plt.ylabel('Y position')
plt.title('X vs Y Positions from Poses')
plt.legend()
plt.grid(True)
plt.show(block=True)

#input("Press Enter to exit and close the plot...")

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

