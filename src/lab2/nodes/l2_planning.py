#!/usr/bin/env python3
#Standard Libraries
import numpy as np
import yaml
import time
import pygame_utils
import matplotlib.image as mpimg
from skimage.draw import disk
from scipy.linalg import block_diag


def load_map(filename):
    im = mpimg.imread("../maps/" + filename)
    if len(im.shape) > 2:
        im = im[:,:,0]
    im_np = np.array(im)  #Whitespace is true, black is false
    #im_np = np.logical_not(im_np)    
    return im_np


def load_map_yaml(filename):
    with open("../maps/" + filename, "r") as stream:
            map_settings_dict = yaml.safe_load(stream)
    return map_settings_dict

#Node for building a graph
class Node:
    def __init__(self, point, parent_id, cost):
        self.point = point # A 3 by 1 vector [x, y, theta]
        self.parent_id = parent_id # The parent node id that leads to this node (There should only every be one parent in RRT)
        self.cost = cost # The cost to come to this node
        self.children_ids = [] # The children node ids of this node
        return

#Path Planner 
class PathPlanner:
    #A path planner capable of perfomring RRT and RRT*
    def __init__(self, map_filename, map_setings_filename, goal_point, stopping_dist):
        #Get map information
        self.occupancy_map = load_map(map_filename)
        self.map_shape = self.occupancy_map.shape
        self.map_settings_dict = load_map_yaml(map_setings_filename)

        #Get the metric bounds of the map
        self.bounds = np.zeros([2,2]) #m
        self.bounds[0, 0] = self.map_settings_dict["origin"][0]
        self.bounds[1, 0] = self.map_settings_dict["origin"][1]
        self.bounds[0, 1] = self.map_settings_dict["origin"][0] + self.map_shape[1] * self.map_settings_dict["resolution"]
        self.bounds[1, 1] = self.map_settings_dict["origin"][1] + self.map_shape[0] * self.map_settings_dict["resolution"]

        #Robot information
        self.robot_radius = 0.22 #m
        self.vel_max = 0.25 #m/s (Feel free to change!)
        self.rot_vel_max = 0.2 #rad/s (Feel free to change!)

        #Goal Parameters
        self.goal_point = goal_point #m
        self.stopping_dist = stopping_dist #m

        #Trajectory Simulation Parameters
        self.timestep = 1.0 #s
        self.num_substeps = 10

        #Planning storage
        self.nodes = [Node(np.zeros((3,1)), -1, 0)]

        #RRT* Specific Parameters
        self.lebesgue_free = np.sum(self.occupancy_map) * self.map_settings_dict["resolution"] **2
        self.zeta_d = np.pi
        self.gamma_RRT_star = 2 * (1 + 1/2) ** (1/2) * (self.lebesgue_free / self.zeta_d) ** (1/2)
        self.gamma_RRT = self.gamma_RRT_star + .1
        self.epsilon = 2.5
        
        #Pygame window for visualization
        self.window = pygame_utils.PygameWindow(
            "Path Planner", (1000, 1000), self.occupancy_map.shape, self.map_settings_dict, self.goal_point, self.stopping_dist)
        return

    #Functions required for RRT
    def sample_map_space(self):
        #Return an [x,y] coordinate to drive the robot towards

        # Sample random x and y within map bounds
        x = np.random.uniform(self.bounds[0, 0], self.bounds[0, 1])
        y = np.random.uniform(self.bounds[1, 0], self.bounds[1, 1])

        return np.array([[x], [y]])
    
    def check_if_duplicate(self, point):
        #Check if point is a duplicate of an already existing node

        # Define a threshold for considering points as duplicates
        # Use 2x the map resolution as threshold
        threshold = self.map_settings_dict["resolution"] * 2.0

        # Check distance to all existing nodes
        for node in self.nodes:
            node_pos = node.point[:2, :]  # Extract x, y position
            distance = np.linalg.norm(point - node_pos)
            if distance < threshold:
                return True

        return False
    
    def closest_node(self, point):
        #Returns the index of the closest node

        min_distance = float('inf')
        closest_node_id = 0

        # Iterate through all nodes to find the closest one
        for i, node in enumerate(self.nodes):
            node_pos = node.point[:2, :]  # Extract x, y position
            distance = np.linalg.norm(point - node_pos)

            if distance < min_distance:
                min_distance = distance
                closest_node_id = i

        return closest_node_id
    
    def simulate_trajectory(self, node_i, point_s):
        #Simulates the non-holonomic motion of the robot.
        #This function drives the robot from node_i towards point_s. This function does has many solutions!
        #node_i is a 3 by 1 vector [x;y;theta] this can be used to construct the SE(2) matrix T_{OI} in course notation
        #point_s is the sampled point vector [x; y]

        # Store the starting pose for trajectory_rollout to access
        self._current_start_pose = node_i

        # Compute control velocities to drive toward the sampled point
        vel, rot_vel = self.robot_controller(node_i, point_s)

        # Simulate the trajectory using the computed velocities
        robot_traj = self.trajectory_rollout(vel, rot_vel)
        return robot_traj
    
    def robot_controller(self, node_i, point_s):
        #This controller determines the velocities that will nominally move the robot from node i to node s
        #Max velocities should be enforced

        # Extract current pose
        x_current = node_i[0, 0]
        y_current = node_i[1, 0]
        theta_current = node_i[2, 0]

        # Extract target position
        # Handle both 2x1 and 1D arrays
        if point_s.ndim == 1:
            x_target = point_s[0]
            y_target = point_s[1]
        else:
            x_target = point_s[0, 0] if point_s.shape[1] == 1 else point_s[0, 0]
            y_target = point_s[1, 0] if point_s.shape[1] == 1 else point_s[1, 0]

        # Compute vector to target
        dx = x_target - x_current
        dy = y_target - y_current
        distance = np.sqrt(dx**2 + dy**2)

        # Compute desired heading (angle to target)
        theta_desired = np.arctan2(dy, dx)

        # Compute heading error
        theta_error = theta_desired - theta_current

        # Normalize angle to [-pi, pi]
        theta_error = np.arctan2(np.sin(theta_error), np.cos(theta_error))

        # Proportional controller for angular velocity
        # Higher gain for faster turning
        k_rot = 2.0
        rot_vel = k_rot * theta_error

        # Clamp to max rotational velocity
        rot_vel = np.clip(rot_vel, -self.rot_vel_max, self.rot_vel_max)

        # Proportional controller for linear velocity
        # Move faster when pointing in the right direction
        # Reduce speed when heading error is large (need to turn more)
        k_vel = 1.0
        vel = k_vel * self.vel_max * np.cos(theta_error)

        # Scale by distance (slow down when close)
        if distance < 1.0:
            vel = vel * distance

        # Clamp to max velocity and prevent backward motion
        vel = np.clip(vel, 0, self.vel_max)

        return vel, rot_vel
    
    def trajectory_rollout(self, vel, rot_vel):
        # Given your chosen velocities determine the trajectory of the robot for your given timestep
        # The returned trajectory should be a series of points to check for collisions

        # Unicycle kinematic model:
        # dx/dt = v * cos(theta)
        # dy/dt = v * sin(theta)
        # dtheta/dt = omega

        # Get the starting pose from the stored instance variable
        start_pose = self._current_start_pose

        # Time step for each substep
        dt = self.timestep / self.num_substeps

        # Initialize trajectory array: 3 x num_substeps
        trajectory = np.zeros((3, self.num_substeps))

        # Extract starting pose
        x = start_pose[0, 0]
        y = start_pose[1, 0]
        theta = start_pose[2, 0]

        # Integrate forward using unicycle kinematics
        for i in range(self.num_substeps):
            # Store current pose
            trajectory[0, i] = x
            trajectory[1, i] = y
            trajectory[2, i] = theta

            # Update pose using unicycle model (Euler integration)
            x = x + vel * np.cos(theta) * dt
            y = y + vel * np.sin(theta) * dt
            theta = theta + rot_vel * dt

            # Normalize theta to [-pi, pi]
            theta = np.arctan2(np.sin(theta), np.cos(theta))

        return trajectory
    
    def point_to_cell(self, point):
        #Convert a series of [x,y] points in the map to the indices for the corresponding cell in the occupancy map
        #point is a 2 by N matrix of points of interest

        # Get map parameters
        resolution = self.map_settings_dict["resolution"]
        origin_x = self.map_settings_dict["origin"][0]
        origin_y = self.map_settings_dict["origin"][1]
        map_height = self.map_shape[0]

        # Handle both single point (2,1) and multiple points (2,N)
        if point.ndim == 1:
            point = point.reshape(-1, 1)

        # Extract x and y coordinates
        x = point[0, :]
        y = point[1, :]

        # Convert world coordinates to pixel coordinates
        # Column: x increases to the right (same as image columns)
        col = (x - origin_x) / resolution

        # Row: y increases upward in world, but rows increase downward in image
        # Bottom of map (low y) corresponds to high row index
        # Top of map (high y) corresponds to low row index (0)
        row = (map_height - 1) - (y - origin_y) / resolution

        # Round to nearest integer and convert to int
        col = np.round(col).astype(int)
        row = np.round(row).astype(int)

        # Stack row and column indices
        cell_indices = np.vstack([row, col])

        return cell_indices

    def points_to_robot_circle(self, points):
        #Convert a series of [x,y] points to robot map footprints for collision detection
        #Hint: The disk function is included to help you with this function

        # Convert points to cell indices using point_to_cell
        cell_indices = self.point_to_cell(points)

        # Calculate robot radius in pixels
        resolution = self.map_settings_dict["resolution"]
        robot_radius_pixels = self.robot_radius / resolution

        # Initialize lists to store all occupied cells
        all_rows = []
        all_cols = []

        # For each point, get all cells that the robot's circular footprint would occupy
        num_points = cell_indices.shape[1]
        for i in range(num_points):
            center_row = cell_indices[0, i]
            center_col = cell_indices[1, i]

            # Use disk function to get all cells within the robot's circular footprint
            # disk returns (rr, cc) - arrays of row and column indices
            rr, cc = disk((center_row, center_col), robot_radius_pixels, shape=self.map_shape)

            # Append to our lists
            all_rows.extend(rr)
            all_cols.extend(cc)

        # Convert to numpy arrays
        all_rows = np.array(all_rows)
        all_cols = np.array(all_cols)

        return all_rows, all_cols
    #Note: If you have correctly completed all previous functions, then you should be able to create a working RRT function

    #RRT* specific functions
    def ball_radius(self):
        #Close neighbor distance
        card_V = len(self.nodes)
        return min(self.gamma_RRT * (np.log(card_V) / card_V ) ** (1.0/2.0), self.epsilon)
    
    def connect_node_to_point(self, node_i, point_f):
        #Given two nodes find the non-holonomic path that connects them
        #Settings
        #node is a 3 by 1 node
        #point is a 2 by 1 point
        print("TO DO: Implement a way to connect two already existing nodes (for rewiring).")
        return np.zeros((3, self.num_substeps))
    
    def cost_to_come(self, trajectory_o):
        #The cost to get to a node from lavalle 
        print("TO DO: Implement a cost to come metric")
        return 0
    
    def update_children(self, node_id):
        #Given a node_id with a changed cost, update all connected nodes with the new cost
        print("TO DO: Update the costs of connected nodes after rewiring.")
        return

    #Planner Functions
    def rrt_planning(self):
        #This function performs RRT on the given map and robot
        #You do not need to demonstrate this function to the TAs, but it is left in for you to check your work
        for i in range(10000): #Most likely need more iterations than this to complete the map!
            #Sample map space
            point = self.sample_map_space()

            #Get the closest point
            closest_node_id = self.closest_node(point)

            #Simulate driving the robot towards the closest point
            trajectory_o = self.simulate_trajectory(self.nodes[closest_node_id].point, point)

            #Check for collisions
            # Get all cells the robot would occupy along the trajectory
            rows, cols = self.points_to_robot_circle(trajectory_o[:2, :])

            # Check for collision
            collision = False
            if len(rows) > 0:
                # Check if all indices are within map bounds
                valid_mask = (rows >= 0) & (rows < self.map_shape[0]) & \
                             (cols >= 0) & (cols < self.map_shape[1])

                # If any indices are out of bounds, it's a collision
                if not np.all(valid_mask):
                    collision = True
                else:
                    # Check if any cell is an obstacle
                    # occupancy_map: True = free space, False = obstacle
                    cells_free = self.occupancy_map[rows, cols]
                    if not np.all(cells_free):
                        collision = True

            # If no collision, add the endpoint as a new node
            if not collision:
                # Get the endpoint of the trajectory
                endpoint = trajectory_o[:, -1].reshape(3, 1)

                # Check if it's a duplicate
                if not self.check_if_duplicate(endpoint[:2, :]):
                    # Calculate cost (distance from parent)
                    parent_point = self.nodes[closest_node_id].point
                    distance = np.linalg.norm(endpoint[:2, :] - parent_point[:2, :])
                    new_cost = self.nodes[closest_node_id].cost + distance

                    # Create new node
                    new_node = Node(endpoint, closest_node_id, new_cost)
                    new_node_id = len(self.nodes)
                    self.nodes.append(new_node)

                    # Update parent's children list
                    self.nodes[closest_node_id].children_ids.append(new_node_id)

                    # Update visualization
                    # Pass as a list (function expects array-like object it can modify)
                    self.window.add_se2_pose([endpoint[0, 0], endpoint[1, 0], endpoint[2, 0]])

                    #Check if goal has been reached
                    goal_distance = np.linalg.norm(endpoint[:2, :] - self.goal_point)
                    if goal_distance < self.stopping_dist:
                        print(f"Goal reached after {i+1} iterations!")
                        print(f"Total nodes: {len(self.nodes)}")
                        return self.nodes

            # Print progress every 500 iterations
            if (i + 1) % 500 == 0:
                print(f"Iteration {i+1}: {len(self.nodes)} nodes, closest to goal: {np.min([np.linalg.norm(n.point[:2, :] - self.goal_point) for n in self.nodes]):.2f}m")

        print(f"Completed {i+1} iterations without reaching goal")
        print(f"Total nodes: {len(self.nodes)}")
        return self.nodes
    
    def rrt_star_planning(self):
        #This function performs RRT* for the given map and robot        
        for i in range(1): #Most likely need more iterations than this to complete the map!
            #Sample
            point = self.sample_map_space()

            #Closest Node
            closest_node_id = self.closest_node(point)

            #Simulate trajectory
            trajectory_o = self.simulate_trajectory(self.nodes[closest_node_id].point, point)

            #Check for Collision
            print("TO DO: Check for collision.")

            #Last node rewire
            print("TO DO: Last node rewiring")

            #Close node rewire
            print("TO DO: Near point rewiring")

            #Check for early end
            print("TO DO: Check for early end")
        return self.nodes
    
    def recover_path(self, node_id = -1):
        path = [self.nodes[node_id].point]
        current_node_id = self.nodes[node_id].parent_id
        while current_node_id > -1:
            path.append(self.nodes[current_node_id].point)
            current_node_id = self.nodes[current_node_id].parent_id
        path.reverse()
        return path

def main():
    #Set map information
    map_filename = "willowgarageworld_05res.png"
    map_setings_filename = "willowgarageworld_05res.yaml"

    #robot information
    goal_point = np.array([[10], [10]]) #m
    stopping_dist = 0.5 #m

    #RRT precursor
    path_planner = PathPlanner(map_filename, map_setings_filename, goal_point, stopping_dist)

    # Use RRT planning (change to rrt_star_planning() once Task 4/5 are complete)
    nodes = path_planner.rrt_planning()
    node_path_metric = np.hstack(path_planner.recover_path())

    #Leftover test functions
    np.save("shortest_path.npy", node_path_metric)


if __name__ == '__main__':
    main()
