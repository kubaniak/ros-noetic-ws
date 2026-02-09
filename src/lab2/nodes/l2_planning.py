#!/usr/bin/env python3
#Standard Libraries
import numpy as np
import yaml
import pygame
import time
from tqdm import trange
import pygame_utils
import matplotlib.image as mpimg
from skimage.draw import disk
from scipy.linalg import block_diag


def normalize_angle(angle):
    return np.arctan2(np.sin(angle), np.cos(angle))


def heading(p1, p2):
    return np.arctan2(p2[1] - p1[1], p2[0] - p1[0])


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
        self.robot_radius = 0.4 #m (inflated from 0.225 for safety margin)
        self.vel_max = 0.15 #m/s (Feel free to change!)
        self.rot_vel_max = 0.35 #rad/s (Feel free to change!)
        self.min_dTheta_for_just_rotation = 0.8 * np.pi
        self.min_dTheta_for_closest = 0.35 * np.pi
        self.search_dist_around_node = 3.5  # sample within ±3.5m of frontier node

        #Goal Parameters
        self.goal_point = goal_point #m
        self.stopping_dist = stopping_dist #m

        #Trajectory Simulation Parameters
        self.timestep = 3 #s
        self.num_substeps = 20

        #Planning storage
        self.nodes = [Node(np.zeros((3,1)), -1, 0)]
        
        # Pre-allocate array for fast lookup: [x, y, theta]
        # We start with size 10,000 and can resize if needed, or set to expected max iterations
        self.max_nodes = 1000000
        self.node_coords = np.zeros((self.max_nodes, 3))
        # Initialize first node (0,0,0)
        self.node_coords[0] = np.zeros(3)
        self.num_nodes = 1
        
        # Optimization: Track node closest to goal to avoid O(N) search every step
        self.closest_to_goal_id = 0
        self.min_goal_dist = np.linalg.norm(self.nodes[0].point[:2] - self.goal_point)

        # Optimization: Pre-compute disk footprint for collision checking
        radius_pixels = int(np.ceil(self.robot_radius / self.map_settings_dict["resolution"]))
        self.disk_rr, self.disk_cc = disk((0, 0), radius_pixels)
        
        # Density Control Parameters
        self.density_radius = 1.5  # Check density within 1.5m radius
        self.max_density = 50       # Max 50 nodes allowed in that radius

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
    def sample_map_space(self, bounds=None):
        #Return an [x,y] coordinate to drive the robot towards
        xmin, xmax = self.bounds[0, :]
        ymin, ymax = self.bounds[1, :]
        if bounds is not None:
            xmin = max(xmin, bounds[0, 0])
            xmax = min(xmax, bounds[0, 1])
            ymin = max(ymin, bounds[1, 0])
            ymax = min(ymax, bounds[1, 1])
        x = np.random.uniform(xmin, xmax)
        y = np.random.uniform(ymin, ymax)
        return np.array([[x], [y]])
    
    def check_if_duplicate(self, point, parent_id):
        # Optimization: Only check if we haven't moved significantly from the parent
        # Checking against ALL nodes is O(N) and causes massive slowdowns.
        parent_point = self.node_coords[parent_id]
        dist = np.linalg.norm(parent_point[:2] - point[:2].flatten())
        if dist < 0.05: # If moved less than 5cm, consider duplicate
            return True
        return False
    
    def check_density(self, point):
        # Checks if adding 'point' would exceed local density limit
        # Returns True if density is TOO HIGH (reject point), False otherwise
        
        active_coords = self.node_coords[:self.num_nodes]
        dists = np.linalg.norm(active_coords[:, :2] - point[:2].flatten(), axis=1)
        
        # Count how many existing nodes are within density_radius
        count = np.sum(dists < self.density_radius)
        
        if count >= self.max_density:
            return True # Reject: Too dense here
        return False # Accept: Sparse enough
    
    def closest_node(self, point, consider_heading=True):
        #Returns the index of the closest node
        #point is a 2 by 1 vector [x; y]
        
        active_coords = self.node_coords[:self.num_nodes]
        xy = active_coords[:, :2]
        thetas = active_coords[:, 2]
        p_flat = point[:2].flatten()
        
        # Distance to all nodes
        dists = np.linalg.norm(xy - p_flat, axis=1)
        
        if consider_heading:
            # Vectorized heading calculation
            dx = p_flat[0] - xy[:, 0]
            dy = p_flat[1] - xy[:, 1]
            headings = np.arctan2(dy, dx)
            
            # Heading filter
            angle_diffs = np.abs(normalize_angle(headings - thetas))
            invalid_mask = angle_diffs > self.min_dTheta_for_closest
            
            # Set invalid nodes to infinity distance so they aren't picked
            dists[invalid_mask] = np.inf
            
        return np.argmin(dists)
    
    def simulate_trajectory(self, node_i, point_s):
        #Simulates the non-holonomic motion of the robot.
        #This function drives the robot from node_i towards point_s. This function does has many solutions!
        #node_i is a 3 by 1 vector [x;y;theta] this can be used to construct the SE(2) matrix T_{OI} in course notation
        #point_s is the sampled point vector [x; y]
        vel, rot_vel = self.robot_controller(node_i, point_s)
        return self.trajectory_rollout(vel, rot_vel, node_i)
    
    def robot_controller(self, node_i, point_s):
        #This controller determines the velocities that will nominally move the robot from node i to node s
        #Max velocities should be enforced
        x_0, y_0, theta_0 = node_i.flatten()[:3]
        x, y = point_s.flatten()[:2]

        dx, dy = x - x_0, y - y_0
        dist = np.sqrt(dx**2 + dy**2)
        theta_target = np.arctan2(dy, dx)
        dTheta = normalize_angle(theta_target - theta_0)

        # If facing very far from target, just rotate in place
        if abs(dTheta) > self.min_dTheta_for_just_rotation:
            vel = 0
            rot_vel = self.rot_vel_max * np.sign(dTheta)
        else:
            # Proportional controller
            K_v, K_w = 0.5, 1.0
            vel = np.clip(K_v * dist, 0, self.vel_max)
            rot_vel = np.clip(K_w * dTheta, -self.rot_vel_max, self.rot_vel_max)

        return vel, rot_vel
    
    def trajectory_rollout(self, vel, rot_vel, starting_pose):
        # Closed-form unicycle integration in world frame
        # starting_pose is [x, y, theta] (3,1) or (3,)
        t = np.linspace(0, self.timestep, self.num_substeps)
        x_i, y_i, theta_i = np.array(starting_pose).flatten()

        thetas = rot_vel * t
        if rot_vel == 0:
            xs = vel * t * np.cos(theta_i)
            ys = vel * t * np.sin(theta_i)
        else:
            xs = (vel / rot_vel) * (np.sin(theta_i + rot_vel * t) - np.sin(theta_i))
            ys = (vel / rot_vel) * (-np.cos(theta_i + rot_vel * t) + np.cos(theta_i))

        xs = xs + x_i
        ys = ys + y_i
        thetas = normalize_angle(thetas + theta_i)

        return np.vstack((xs, ys, thetas))
    
    def point_to_cell(self, point):
        #Convert a series of [x,y] points in the map to the indices for the corresponding cell in the occupancy map
        #point is a 2 by N matrix of points of interest
        origin = self.map_settings_dict["origin"]
        resolution = self.map_settings_dict["resolution"]

        # x increases rightward -> column increases rightward
        col = np.floor((point[0, :] - origin[0]) / resolution).astype(int)
        # y increases upward in world, but row increases downward in image
        # Bottom row (row = height-1) corresponds to y = origin[1]
        pixel_y = np.floor((point[1, :] - origin[1]) / resolution).astype(int)
        row = (self.map_shape[0] - 1) - pixel_y

        return np.vstack([row, col])

    def points_to_robot_circle(self, points):
        #Convert a series of [x,y] points to robot map footprints for collision detection
        #Hint: The disk function is included to help you with this function
        radius_pixels = int(np.ceil(self.robot_radius / self.map_settings_dict["resolution"]))

        cells = self.point_to_cell(points)

        # Broadcast cells against disk offsets
        # cells shape: (2, N_points), disk shape: (N_disk,)
        # result shape: (N_disk * N_points,)
        rr = (cells[0, :, None] + self.disk_rr[None, :]).flatten()
        cc = (cells[1, :, None] + self.disk_cc[None, :]).flatten()

        # Valid bounds check
        valid = (rr >= 0) & (rr < self.map_shape[0]) & (cc >= 0) & (cc < self.map_shape[1])
        return rr[valid], cc[valid]
    #Note: If you have correctly completed all previous functions, then you should be able to create a working RRT function

    #RRT* specific functions
    def ball_radius(self):
        #Close neighbor distance
        card_V = len(self.nodes)
        return min(self.gamma_RRT * (np.log(card_V) / card_V ) ** (1.0/2.0), self.epsilon)
    
    def connect_node_to_point(self, node_i, point_f):
        #Given two nodes find the non-holonomic path that connects them
        #Turn-then-drive: rotate to face target, then drive straight to it
        #node_i is a 3 by 1 node, point_f is a 2 by 1 point
        x, y, theta = node_i.flatten()[:3]
        x_f, y_f = point_f.flatten()[:2]
        sec_per_step = self.timestep / self.num_substeps
        vel_step = self.vel_max * sec_per_step
        rot_step = self.rot_vel_max * sec_per_step

        # Rotation phase: rotate to face target
        target_heading = heading([x, y], [x_f, y_f])
        heading_error = normalize_angle(target_heading - theta)
        steps_to_rotate = max(int(np.ceil(abs(heading_error / rot_step))), 1)

        traj_r = []
        for _ in range(steps_to_rotate - 1):
            theta += rot_step * np.sign(heading_error)
            traj_r.append([x, y, theta])
        theta = target_heading
        traj_r.append([x, y, theta])
        traj_r = np.array(traj_r).T

        # Translation phase: drive straight to target
        dist = np.hypot(x_f - x, y_f - y)
        steps_to_translate = max(int(np.ceil(dist / vel_step)), 1)

        traj_t = []
        for _ in range(steps_to_translate - 1):
            x += vel_step * np.cos(theta)
            y += vel_step * np.sin(theta)
            traj_t.append([x, y, theta])
        traj_t.append([x_f, y_f, target_heading])
        traj_t = np.array(traj_t).T

        return np.hstack([traj_r, traj_t])
    
    def cost_to_come(self, trajectory_o):
        #The cost to get to a node from lavalle
        #Path length: sum of Euclidean distances between consecutive trajectory points
        diffs = np.diff(trajectory_o[:2, :], axis=1)
        return np.sum(np.linalg.norm(diffs, axis=0))
    
    def update_children(self, node_id):
        #Given a node_id with a changed cost, update all connected nodes with the new cost
        for child_id in self.nodes[node_id].children_ids:
            #Recompute trajectory from this node to the child
            trajectory = self.connect_node_to_point(
                self.nodes[node_id].point,
                self.nodes[child_id].point[:2]
            )
            #Update child cost = parent cost + edge cost
            self.nodes[child_id].cost = self.nodes[node_id].cost + self.cost_to_come(trajectory)
            #Recursively propagate to grandchildren
            self.update_children(child_id)

    def is_ancestor(self, ancestor_id, node_id):
        #Check if ancestor_id is an ancestor of node_id to prevent cycles
        current = self.nodes[node_id].parent_id
        while current > -1:
            if current == ancestor_id:
                return True
            current = self.nodes[current].parent_id
        return False

    #Planner Functions
    def rrt_planning(self):
        #This function performs RRT on the given map and robot
        #You do not need to demonstrate this function to the TAs, but it is left in for you to check your work
        points_to_draw = []
        edges_to_draw = []
        for i in trange(self.max_nodes):
            if i % 500 == 0 and i > 0:
                print(f"RRT* iter {i}: nodes={len(self.nodes)}. Closest node to goal is {self.closest_to_goal_id} at distance {self.min_goal_dist:.2f}m")
                # Optimization: Batch draw all buffered points at once
                for p in points_to_draw:
                    self.window.add_point(p, radius=2)
                for (p1, p2) in edges_to_draw:
                    self.window.add_line(p1, p2)
                
                points_to_draw = []
                edges_to_draw = []
                
            # check if all nodes within radius of goal can make direct connection with it without collision. If so, connect and end early.
            if self.closest_to_goal_id is not None:
                active_coords = self.node_coords[:self.num_nodes]
                dists_to_goal = np.linalg.norm(active_coords[:, :2] - self.goal_point.flatten(), axis=1)
                candidate_ids = np.where(dists_to_goal < self.search_dist_around_node)[0]
                for candidate_id in candidate_ids:
                    traj_to_goal = self.connect_node_to_point(self.nodes[candidate_id].point, self.goal_point)
                    rows, cols = self.points_to_robot_circle(traj_to_goal[:2, :])
                    if len(rows) > 0 and np.all(self.occupancy_map[rows, cols] > 0.5):
                        goal_state = traj_to_goal[:, -1].reshape(3, 1)
                        goal_node = Node(goal_state, candidate_id, 0)
                        
                        self.nodes[candidate_id].children_ids.append(len(self.nodes))
                        self.nodes.append(goal_node)
                        
                        # Draw final connection to goal
                        self.window.add_line(self.nodes[candidate_id].point[:2].flatten(), goal_state[:2].flatten())
                        
                        print(f"RRT*: Goal connected directly from node {candidate_id} at iteration {i} with {len(self.nodes)} nodes!")
                        for p in points_to_draw:
                            self.window.add_point(p, radius=2)
                        for (p1, p2) in edges_to_draw:
                            self.window.add_line(p1, p2)
                        return self.nodes
            
            rand_val = np.random.random()
            if rand_val < 0.30:
                # Global sampling: anywhere in map
                point = self.sample_map_space(bounds=None)
            elif rand_val < 0.45:
                # Goal sampling: strict focus on goal
                goal_radius = 1.5
                gx, gy = self.goal_point.flatten()
                bounds = np.array([[gx - goal_radius, gx + goal_radius],
                                   [gy - goal_radius, gy + goal_radius]])
                point = self.sample_map_space(bounds=bounds)
            else:
                # Local sampling: around current best node
                near_x, near_y = self.nodes[self.closest_to_goal_id].point[:2, 0]
                d = self.search_dist_around_node
                bounds = np.array([[near_x - d, near_x + d],
                                   [near_y - d, near_y + d]])
                point = self.sample_map_space(bounds=bounds)

            #Closest Node
            closest_node_id = self.closest_node(point)

            #Simulate trajectory
            trajectory_o = self.simulate_trajectory(self.nodes[closest_node_id].point, point)

            #Get new node from trajectory endpoint
            new_point = trajectory_o[:, -1:].copy()

            #Check if duplicate (Optimized)
            if self.check_if_duplicate(new_point, closest_node_id):
                continue
            
            #Check for Density (Prevent Local Minima Sampling)
            if self.check_density(new_point):
                continue # Skip if area is already dense

            #Check for collisions along trajectory
            rows, cols = self.points_to_robot_circle(trajectory_o[:2, :])
            if len(rows) == 0 or not np.all(self.occupancy_map[rows, cols] > 0.5):
                continue

            #Add new node to tree
            new_node = Node(new_point, closest_node_id, 0)
            self.nodes[closest_node_id].children_ids.append(len(self.nodes))
            self.nodes.append(new_node)
            
            self.node_coords[self.num_nodes] = new_point.flatten()
            self.num_nodes += 1
            
            # Optimization: Buffer new points to draw later
            points_to_draw.append(new_point[:2].flatten())
            # Buffer the edge to the parent
            parent_point = self.nodes[closest_node_id].point[:2].flatten()
            edges_to_draw.append((parent_point, new_point[:2].flatten()))
            
            # Optimization: Update tracked closest-to-goal node
            dist_to_goal_now = np.linalg.norm(new_point[:2] - self.goal_point)
            if dist_to_goal_now < self.min_goal_dist:
                self.min_goal_dist = dist_to_goal_now
                self.closest_to_goal_id = len(self.nodes) - 1

            #Check if goal has been reached
            dist_to_goal = np.linalg.norm(new_point[:2] - self.goal_point)
            if dist_to_goal < self.stopping_dist:
                print("RRT: Goal reached at iteration %d with %d nodes!" % (i, len(self.nodes)))
                break

        return self.nodes
    
    def rrt_star_planning(self):
        #This function performs RRT* for the given map and robot
        points_to_draw = []
        edges_to_draw = []
        for i in trange(self.max_nodes):
            if i % 500 == 0 and i > 0:
                print(f"RRT* iter {i}: nodes={len(self.nodes)}. Closest node to goal is {self.closest_to_goal_id} at distance {self.min_goal_dist:.2f}m")
                # Optimization: Batch draw all buffered points at once
                for p in points_to_draw:
                    self.window.add_point(p, radius=2)
                for (p1, p2) in edges_to_draw:
                    self.window.add_line(p1, p2)
                
                points_to_draw = []
                edges_to_draw = []
            
            # check if all nodes within radius of goal can make direct connection with it without collision. If so, connect and end early.
            if self.closest_to_goal_id is not None:
                active_coords = self.node_coords[:self.num_nodes]
                dists_to_goal = np.linalg.norm(active_coords[:, :2] - self.goal_point.flatten(), axis=1)
                candidate_ids = np.where(dists_to_goal < self.search_dist_around_node)[0]
                for candidate_id in candidate_ids:
                    traj_to_goal = self.connect_node_to_point(self.nodes[candidate_id].point, self.goal_point)
                    rows, cols = self.points_to_robot_circle(traj_to_goal[:2, :])
                    if len(rows) > 0 and np.all(self.occupancy_map[rows, cols] > 0.5):
                        goal_state = traj_to_goal[:, -1].reshape(3, 1)
                        goal_node = Node(goal_state, candidate_id, 0)
                        
                        self.nodes[candidate_id].children_ids.append(len(self.nodes))
                        self.nodes.append(goal_node)
                        
                        # Draw final connection to goal
                        self.window.add_line(self.nodes[candidate_id].point[:2].flatten(), goal_state[:2].flatten())
                        
                        print(f"RRT*: Goal connected directly from node {candidate_id} at iteration {i} with {len(self.nodes)} nodes!")
                        for p in points_to_draw:
                            self.window.add_point(p, radius=2)
                        for (p1, p2) in edges_to_draw:
                            self.window.add_line(p1, p2)
                        return self.nodes
            
            rand_val = np.random.random()
            if rand_val < 0.30:
                # Global sampling: anywhere in map
                point = self.sample_map_space(bounds=None)
            elif rand_val < 0.45:
                # Goal sampling: strict focus on goal
                goal_radius = 1.5
                gx, gy = self.goal_point.flatten()
                bounds = np.array([[gx - goal_radius, gx + goal_radius],
                                   [gy - goal_radius, gy + goal_radius]])
                point = self.sample_map_space(bounds=bounds)
            else:
                # Local sampling: around current best node
                near_x, near_y = self.nodes[self.closest_to_goal_id].point[:2, 0]
                d = self.search_dist_around_node
                bounds = np.array([[near_x - d, near_x + d],
                                   [near_y - d, near_y + d]])
                point = self.sample_map_space(bounds=bounds)

            #Closest Node
            closest_node_id = self.closest_node(point)

            #Simulate trajectory
            trajectory_o = self.simulate_trajectory(self.nodes[closest_node_id].point, point)

            #Get new node from trajectory endpoint
            new_point = trajectory_o[:, -1:].copy()

            #Check if duplicate (Optimized)
            if self.check_if_duplicate(new_point, closest_node_id):
                continue
            
            #Check for Density (Prevent Local Minima Sampling)
            if self.check_density(new_point):
                    continue 

            #Check for collisions along trajectory
            rows, cols = self.points_to_robot_circle(trajectory_o[:2, :])
            if len(rows) == 0 or not np.all(self.occupancy_map[rows, cols] > 0.5):
                continue

            #Find all nodes within ball radius for rewiring
            ball_rad = self.ball_radius()
            
            # Use pre-allocated numpy array instead of list comprehension
            active_coords = self.node_coords[:self.num_nodes]
            dists = np.linalg.norm(active_coords[:, :2] - new_point[:2].flatten(), axis=1)
            near_node_ids = np.where(dists < ball_rad)[0].tolist()

            #Best parent selection
            best_parent_id = closest_node_id
            best_cost = self.nodes[closest_node_id].cost + self.cost_to_come(trajectory_o)

            for near_id in near_node_ids:
                traj = self.connect_node_to_point(self.nodes[near_id].point, new_point[:2])
                rows, cols = self.points_to_robot_circle(traj[:2, :])
                if len(rows) == 0 or not np.all(self.occupancy_map[rows, cols] > 0.5):
                    continue
                cost = self.nodes[near_id].cost + self.cost_to_come(traj)
                if cost < best_cost:
                    best_parent_id = near_id
                    best_cost = cost

            #Add new node with best parent
            new_node = Node(new_point, best_parent_id, best_cost)
            new_node_id = len(self.nodes)
            self.nodes[best_parent_id].children_ids.append(new_node_id)
            self.nodes.append(new_node)
            
            self.node_coords[self.num_nodes] = new_point.flatten()
            self.num_nodes += 1
            
            # Optimization: Buffer new points to draw later
            points_to_draw.append(new_point[:2].flatten())
            # Buffer the edge to the parent
            parent_point = self.nodes[best_parent_id].point[:2].flatten()
            edges_to_draw.append((parent_point, new_point[:2].flatten()))
            
            # Optimization: Update tracked closest-to-goal node
            dist_to_goal_now = np.linalg.norm(new_point[:2] - self.goal_point)
            if dist_to_goal_now < self.min_goal_dist:
                self.min_goal_dist = dist_to_goal_now
                self.closest_to_goal_id = new_node_id

            #Rewire nearby nodes through new node
            for near_id in near_node_ids:
                if near_id == best_parent_id:
                    continue
                #Cycle detection: skip if near_id is ancestor of new_node
                if self.is_ancestor(near_id, new_node_id):
                    continue
                traj = self.connect_node_to_point(new_point, self.nodes[near_id].point[:2])
                rows, cols = self.points_to_robot_circle(traj[:2, :])
                if len(rows) == 0 or not np.all(self.occupancy_map[rows, cols] > 0.5):
                    continue
                new_cost = new_node.cost + self.cost_to_come(traj)
                if new_cost < self.nodes[near_id].cost:
                    old_parent_id = self.nodes[near_id].parent_id
                    self.nodes[old_parent_id].children_ids.remove(near_id)
                    self.nodes[near_id].parent_id = new_node_id
                    self.nodes[near_id].cost = new_cost
                    new_node.children_ids.append(near_id)
                    self.update_children(near_id)
                    # Draw new rewired edge
                    edges_to_draw.append((new_point[:2].flatten(), self.nodes[near_id].point[:2].flatten()))

            #Check for early end
            if dist_to_goal_now < self.stopping_dist:
                print(f"RRT*: Goal reached at iteration {i} with {len(self.nodes)} nodes!")
                break

        return self.nodes
    
    def recover_path(self, node_id = -1):
        path = [self.nodes[node_id].point]
        current_node_id = self.nodes[node_id].parent_id
        while current_node_id > -1:
            path.append(self.nodes[current_node_id].point)
            current_node_id = self.nodes[current_node_id].parent_id
        path.reverse()

        # Build adjusted path where the heading at each waypoint is the
        # midpoint between the incoming and outgoing directions.
        adjusted = []
        for i in range(len(path)):
            x, y = path[i][:2, 0]
            if i == 0:
                # First waypoint: heading toward the next
                h = heading(path[0][:2, 0], path[1][:2, 0])
            elif i == len(path) - 1:
                # Last waypoint: heading from the previous
                h = heading(path[-2][:2, 0], path[-1][:2, 0])
            else:
                # Midpoint heading between incoming and outgoing directions
                h_prev = heading(path[i-1][:2, 0], path[i][:2, 0])
                h_next = heading(path[i][:2, 0], path[i+1][:2, 0])
                # Average on the unit circle to avoid wrap issues
                h = np.arctan2(np.sin(h_prev) + np.sin(h_next), np.cos(h_prev) + np.cos(h_next))

            adjusted.append(np.array([[x], [y], [h]]))

        return adjusted

def main():
    #Set map information
    map_filename = "willowgarageworld_05res.png"
    map_setings_filename = "willowgarageworld_05res.yaml"

    #robot information
    goal_point = np.array([[42.05], [-44]]) #m
    stopping_dist = 0.5 #m

    #RRT* planning
    path_planner = PathPlanner(map_filename, map_setings_filename, goal_point, stopping_dist)
    nodes = path_planner.rrt_star_planning()
    node_path_metric = np.hstack(path_planner.recover_path())
    
    #Save path for trajectory rollout
    np.save("path.npy", node_path_metric)
    print(f"Path saved with {node_path_metric.shape[1]} waypoints")
    
    # Visualizing the Final Path in Green
    print("Visualizing final path...")
    for i in range(node_path_metric.shape[1] - 1):
        p1 = node_path_metric[:2, i].flatten()
        p2 = node_path_metric[:2, i + 1].flatten()
        # Draw green line (0, 255, 0)
        path_planner.window.add_line(p1, p2, width=3, color=(0, 255, 0))
    
    # Keep window open for a bit
    print("Done! Close the window to exit.")
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        time.sleep(0.1)


if __name__ == '__main__':
    main()
