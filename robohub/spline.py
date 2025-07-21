import numpy as np
from scipy.interpolate import CubicHermiteSpline
import matplotlib.pyplot as plt
# Start and goal positions
x_s, y_s, theta_s = 0.0, 0.0, 0.0
x_g, y_g, theta_g = 1.0, 1.0, -np.pi / 6
v_mag = 0.1  # Arbitrary speed for derivatives

dx_s = v_mag * np.cos(theta_s)
dy_s = v_mag * np.sin(theta_s)

dx_g = v_mag * np.cos(theta_g)
dy_g = v_mag * np.sin(theta_g)


T = 20.0
t = np.linspace(0, T, 500)

# Spline for x(t) and y(t)
spline_x = CubicHermiteSpline([0, T], [x_s, x_g], [dx_s, dx_g])
spline_y = CubicHermiteSpline([0, T], [y_s, y_g], [dy_s, dy_g])

x_t = spline_x(t)
y_t = spline_y(t)

# Optional: plot the path
plt.figure()
plt.plot(x_t, y_t, label='Planned Path')
plt.plot(x_s, y_s, 'go', label='Start')
plt.plot(x_g, y_g, 'ro', label='Goal')
plt.axis('equal')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Kinematically Feasible Path')
plt.legend()
plt.grid()
plt.show()


def compute_feedforward_controls(T=20.0):
    t = np.linspace(0, T, 500)

    # First derivatives
    dx = spline_x(t, 1)
    dy = spline_y(t, 1)

    # Second derivatives
    ddx = spline_x(t, 2)
    ddy = spline_y(t, 2)

    # Feedforward controls
    v = np.sqrt(dx**2 + dy**2)
    omega = (ddy * dx - ddx * dy) / (dx**2 + dy**2)

    return t, v, omega




def plot_controls():
    t, v, omega = compute_feedforward_controls()

    plt.figure(figsize=(12, 4))

    plt.subplot(1, 2, 1)
    plt.plot(t, v)
    plt.title("Feedforward Linear Velocity v(t)")
    plt.xlabel("Time [s]")
    plt.ylabel("v [m/s]")
    plt.grid()

    plt.subplot(1, 2, 2)
    plt.plot(t, omega)
    plt.title("Feedforward Angular Velocity ω(t)")
    plt.xlabel("Time [s]")
    plt.ylabel("ω [rad/s]")
    plt.grid()

    plt.tight_layout()
    plt.show()



plot_controls()

