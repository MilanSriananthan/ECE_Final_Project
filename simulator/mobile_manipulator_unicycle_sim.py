import math
import random
import time
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np

class MobileManipulatorUnicycleSim:
    def __init__(self, robot_id, backend_server_ip=None, robot_pose=[], pickup_location=[], dropoff_location=[], obstacles_location=[]):
        # constants
        self.MAX_LINEAR_SPEED = 0.1 # meters / second
        self.MAX_ANGULAR_SPEED = 30 * math.pi / 180.0 # 30 degrees / second
        self.TIMEOUT_SET_MOBILE_BASE_SPEED = 0 # milliseconds
        self.TIMEOUT_GET_POSES = 0 # milliseconds
        self.ENV_SIZE = 5 # x,y can vary from -ENV_SIZE/2 to ENV_SIZE/2
        self.ROBOT_SIZE = [0.24, 0.32] # [w, l]
        self.GRIPPER_SIZE = 0.1
        self.LOC_RADIUS = 0.1

        # initialize variables for object poses
        if robot_pose:
            self.robot_pose = robot_pose
        else:
            self.robot_pose = [-self.ENV_SIZE / 2.0 + self.ENV_SIZE * random.random(), -self.ENV_SIZE / 2.0 + self.ENV_SIZE * random.random(), 2 * math.pi * random.random()]
        if pickup_location:
            self.pickup_location = pickup_location
        else:
            self.pickup_location = [-self.ENV_SIZE / 2.0 + self.ENV_SIZE * random.random(), -self.ENV_SIZE / 2.0 + self.ENV_SIZE * random.random(), 0.]
        if dropoff_location:
            self.dropoff_location = dropoff_location
        else:
            self.dropoff_location = [-self.ENV_SIZE / 2.0 + self.ENV_SIZE * random.random(), -self.ENV_SIZE / 2.0 + self.ENV_SIZE * random.random(), 0.]
        if obstacles_location:
            self.obstacles_location = obstacles_location
        else:
            self.obstacles_location = [[-self.ENV_SIZE / 2.0 + self.ENV_SIZE * random.random(), -self.ENV_SIZE / 2.0 + self.ENV_SIZE * random.random(), 0.] for _ in range(9)]

        # initialize counters
        self.last_time_set_mobile_base_speed = int(round(time.time()*1000))
        self.last_time_get_poses = int(round(time.time()*1000))

        # init plot
        self.figure = []
        self.axes = []
        self.patches = []
        self.__init_plot()
    
    def __init_plot(self):
        self.figure, self.axes = plt.subplots()
        p_env = patches.Rectangle(np.array([-self.ENV_SIZE / 2.0, -self.ENV_SIZE / 2.0]), self.ENV_SIZE, self.ENV_SIZE, fill=False)
        p_pu_loc = patches.Circle(np.array(self.pickup_location[:2]), radius=self.LOC_RADIUS, facecolor='b')
        p_do_loc = patches.Circle(np.array(self.dropoff_location[:2]), radius=self.LOC_RADIUS, facecolor='g')
        p_o_locs = []
        for i in range(len(self.obstacles_location)):
            p_o_locs.append(patches.Circle(np.array(self.obstacles_location[i][:2]), radius=self.LOC_RADIUS, facecolor='r'))

        R = np.array([[0.0, 1.0], [-1.0, 0.0]]) @ np.array([[math.cos(self.robot_pose[2]), -math.sin(self.robot_pose[2])], [math.sin(self.robot_pose[2]), math.cos(self.robot_pose[2])]])
        t = np.array([self.robot_pose[0], self.robot_pose[1]])
        p_robot = patches.Polygon(t + np.array([0, self.ROBOT_SIZE[1]]) @ R.T / 2.0 + (np.array([[-self.ROBOT_SIZE[0] / 2.0, 0.0], [self.ROBOT_SIZE[0] / 2.0, 0.0], [self.ROBOT_SIZE[0] / 2.0, self.ROBOT_SIZE[1]], [-self.ROBOT_SIZE[0] / 2.0, self.ROBOT_SIZE[1]]]) - np.array([0, self.ROBOT_SIZE[1]]) / 2.0) @ R.T, facecolor='k')
        p_gripper = patches.Polygon(t + np.array([0, self.ROBOT_SIZE[1]]) @ R.T / 2.0 + (np.array([0, self.ROBOT_SIZE[1]]) + np.array([[-self.GRIPPER_SIZE / 2.0, 0.0], [self.GRIPPER_SIZE / 2.0, 0.0], [self.GRIPPER_SIZE / 2.0, self.GRIPPER_SIZE], [0.8 * self.GRIPPER_SIZE / 2.0, self.GRIPPER_SIZE], [0.8 * self.GRIPPER_SIZE / 2.0, 0.0], [-0.8 * self.GRIPPER_SIZE / 2.0, 0.0], [-0.8 * self.GRIPPER_SIZE / 2.0, self.GRIPPER_SIZE], [-self.GRIPPER_SIZE / 2.0, self.GRIPPER_SIZE], [-self.GRIPPER_SIZE / 2.0, 0.0]]) - np.array([0, self.ROBOT_SIZE[1]]) / 2.0) @ R.T, facecolor='k')
        
        self.patches.append(p_robot)
        self.patches.append(p_gripper)
        self.patches.append(p_pu_loc)
        self.patches.append(p_do_loc)
        for i in range(len(p_o_locs)):
            self.patches.append(p_o_locs[i])
        
        self.axes.add_patch(p_env)
        self.axes.add_patch(p_robot)
        self.axes.add_patch(p_gripper)
        self.axes.add_patch(p_pu_loc)
        self.axes.add_patch(p_do_loc)
        for i in range(len(p_o_locs)):
            self.axes.add_patch(p_o_locs[i])
        
        self.axes.set_xlim(-0.6*self.ENV_SIZE, 0.6*self.ENV_SIZE)
        self.axes.set_ylim(-0.6*self.ENV_SIZE, 0.6*self.ENV_SIZE)
        self.axes.set_axis_off()
        self.axes.axis('equal')

        plt.ion()
        plt.show()

    def __update_plot(self):
        R = np.array([[0.0, 1.0], [-1.0, 0.0]]) @ np.array([[math.cos(self.robot_pose[2]), -math.sin(self.robot_pose[2])], [math.sin(self.robot_pose[2]), math.cos(self.robot_pose[2])]])
        t = np.array([self.robot_pose[0], self.robot_pose[1]])
        xy_robot = t + np.array([0, self.ROBOT_SIZE[1]]) @ R.T / 2.0 + (np.array([[-self.ROBOT_SIZE[0] / 2.0, 0.0], [self.ROBOT_SIZE[0] / 2.0, 0.0], [self.ROBOT_SIZE[0] / 2.0, self.ROBOT_SIZE[1]], [-self.ROBOT_SIZE[0] / 2.0, self.ROBOT_SIZE[1]]]) - np.array([0, self.ROBOT_SIZE[1]]) / 2.0) @ R.T
        xy_gripper = t + np.array([0, self.ROBOT_SIZE[1]]) @ R.T / 2.0 + (np.array([0, self.ROBOT_SIZE[1]]) + np.array([[-self.GRIPPER_SIZE / 2.0, 0.0], [self.GRIPPER_SIZE / 2.0, 0.0], [self.GRIPPER_SIZE / 2.0, self.GRIPPER_SIZE], [0.8 * self.GRIPPER_SIZE / 2.0, self.GRIPPER_SIZE], [0.8 * self.GRIPPER_SIZE / 2.0, 0.0], [-0.8 * self.GRIPPER_SIZE / 2.0, 0.0], [-0.8 * self.GRIPPER_SIZE / 2.0, self.GRIPPER_SIZE], [-self.GRIPPER_SIZE / 2.0, self.GRIPPER_SIZE], [-self.GRIPPER_SIZE / 2.0, 0.0]]) - np.array([0, self.ROBOT_SIZE[1]]) / 2.0) @ R.T
        
        self.patches[0].xy = xy_robot
        self.patches[1].xy = xy_gripper
        self.patches[2].center = self.pickup_location
        self.patches[3].center = self.dropoff_location
        for i in range(4, len(self.patches)):
            self.patches[i].center = self.obstacles_location[i-4]

        self.figure.canvas.draw_idle()
        self.figure.canvas.flush_events()

    def set_arm_pose(self, x, y, wait_s=2.6):
        pass

    def set_mobile_base_speed_and_gripper_power(self, v : float, omega : float, gripper_power : float):
        delta_time_set_mobile_base_speed = int(round(time.time()*1000)) - self.last_time_set_mobile_base_speed
        if delta_time_set_mobile_base_speed > self.TIMEOUT_SET_MOBILE_BASE_SPEED:
            v, omega = self.__saturate_speeds(v, omega)
            self.robot_pose[0] = self.robot_pose[0] + (v * math.cos(self.robot_pose[2])) * delta_time_set_mobile_base_speed / 1000.0
            self.robot_pose[1] = self.robot_pose[1] + (v * math.sin(self.robot_pose[2])) * delta_time_set_mobile_base_speed / 1000.0
            self.robot_pose[2] = self.robot_pose[2] + omega * delta_time_set_mobile_base_speed / 1000.0
            self.last_time_set_mobile_base_speed = int(round(time.time()*1000))
            # update plot
            self.__update_plot()

    def set_leds(self, red_level : int, green_level : int, blue_level : int):
        pass

    def __saturate_speeds(self, v, omega):
        v = max(-self.MAX_LINEAR_SPEED, min(v, self.MAX_LINEAR_SPEED))
        omega = max(-self.MAX_ANGULAR_SPEED, min(omega, self.MAX_ANGULAR_SPEED))
        return v, omega
    
    def get_poses(self):
        while int(round(time.time()*1000)) - self.last_time_get_poses < self.TIMEOUT_GET_POSES:
            pass
        self.last_time_get_poses = int(round(time.time()*1000))
        return [self.robot_pose, self.pickup_location, self.dropoff_location, self.obstacles_location]