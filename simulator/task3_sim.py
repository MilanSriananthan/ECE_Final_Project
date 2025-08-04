from mobile_manipulator_unicycle_sim import MobileManipulatorUnicycleSim
import time
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import CubicHermiteSpline

# --- Parameters ---
T = 20.0
N = 100
dt = T / N
t_vals = np.linspace(0, T, N)

# Start and goal
x_s, y_s, theta_s = 0.0, 0.0, 0.0
x_g, y_g, theta_g = 1.0, 1.0, -np.pi / 6
v_mag = 0.1

# Initial/final velocities
dx_s = v_mag * np.cos(theta_s)
dy_s = v_mag * np.sin(theta_s)
dx_g = v_mag * np.cos(theta_g)
dy_g = v_mag * np.sin(theta_g)

# Hermite spline
spline_x = CubicHermiteSpline([0, T], [x_s, x_g], [dx_s, dx_g])
spline_y = CubicHermiteSpline([0, T], [y_s, y_g], [dy_s, dy_g])

# Desired trajectory and derivatives
x_d = spline_x(t_vals)
y_d = spline_y(t_vals)
dx_d = spline_x(t_vals, 1)
dy_d = spline_y(t_vals, 1)
ddx_d = spline_x(t_vals, 2)
ddy_d = spline_y(t_vals, 2)

v_ff = np.sqrt(dx_d**2 + dy_d**2)
omega_ff = (ddy_d * dx_d - ddx_d * dy_d) / (dx_d**2 + dy_d**2 + 1e-5)

# --- Gains ---
Kp = 0.0001
Kd = 0.0001

# --- Simulator Init ---
robot = MobileManipulatorUnicycleSim(
    robot_id=1,
    backend_server_ip=None,
    robot_pose=[x_s, y_s, theta_s],
    pickup_location=[-0.5, 1.75, 0.],
    dropoff_location=[-1.5, -1.5, 0.],
    obstacles_location=[[-2.0, -1.0, 0.0], [-1.6, -1.0, 0.0], [-1.2, -1.0, 0.0],
                        [-0.8, 1.0, 0.0], [-0.4, 1.0, 0.0], [0.0, 1.0, 0.0],
                        [0.4, 2.0, 0.0], [0.8, 2.0, 0.0], [1.2, 2.0, 0.0]]
)

print("Running feedback control...")

# --- Storage ---
pose_list = []
v_cmd_list = []
omega_cmd_list = []
pos_error_list = []

prev_x, prev_y = None, None

for i in range(N):
    # Get current pose
    x, y, theta = robot.get_poses()[0]
    pose_list.append(np.array([x, y, theta]))

    # Estimate velocity
    if i > 0 and prev_x is not None:
        dx = (x - prev_x) / dt
        dy = (y - prev_y) / dt
    else:
        dx, dy = 0.0, 0.0
    prev_x, prev_y = x, y

    # Position and velocity error
    ex = x_d[i] - x
    ey = y_d[i] - y
    evx = dx_d[i] - dx
    evy = dy_d[i] - dy

    # Flat space feedback control
    ux = Kp * ex + Kd * evx
    uy = Kp * ey + Kd * evy

    # Compute new v, omega
    v = v_ff[i] + np.sqrt(ux**2 + uy**2) * np.sign(v_ff[i])
    omega = omega_ff[i] + (ux * dy_d[i] - uy * dx_d[i]) / (dx_d[i]**2 + dy_d[i]**2 + 1e-5)

    # Send command
    robot.set_mobile_base_speed_and_gripper_power(v=v, omega=omega, gripper_power=0.0)

    # Store
    v_cmd_list.append(v)
    omega_cmd_list.append(omega)
    pos_error_list.append(np.sqrt(ex**2 + ey**2))

    time.sleep(dt)

# Stop robot
robot.set_mobile_base_speed_and_gripper_power(0.0, 0.0, 0.0)

# --- Process and Plot ---
x_actual = [pose[0] for pose in pose_list]
y_actual = [pose[1] for pose in pose_list]

# Trajectory
plt.figure(figsize=(8, 6))
plt.plot(x_actual, y_actual, 'b-', label='Actual Trajectory')
plt.plot(x_d, y_d, 'r--', label='Desired Trajectory')
plt.xlabel("X")
plt.ylabel("Y")
plt.title("Trajectory: Feedback Control")
plt.legend()
plt.grid(True)
plt.show(block=True)

# Velocity plots
time_vals = np.linspace(0, T, N)

plt.figure(figsize=(12, 6))
plt.subplot(2, 1, 1)
plt.plot(time_vals, v_ff, 'r--', label='Feedforward v')
plt.plot(time_vals, v_cmd_list, 'b-', label='Feedback v')
plt.ylabel("v (m/s)")
plt.legend()
plt.grid(True)

plt.subplot(2, 1, 2)
plt.plot(time_vals, omega_ff, 'r--', label='Feedforward ω')
plt.plot(time_vals, omega_cmd_list, 'b-', label='Feedback ω')
plt.xlabel("Time (s)")
plt.ylabel("ω (rad/s)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show(block=True)

# Error plot
plt.figure(figsize=(8, 5))
plt.plot(time_vals, pos_error_list)
plt.xlabel("Time (s)")
plt.ylabel("Position Error (m)")
plt.title("Tracking Error Over Time")
plt.grid(True)
plt.show(block=True)
