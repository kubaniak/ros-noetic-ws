#!/usr/bin/env python3
#Standard Libraries
import numpy as np
import yaml
import pygame
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
        self.vel_max = 0.5 #m/s (Feel free to change!)
        self.rot_vel_max = 0.2 #rad/s (Feel free to change!)

        #Goal Parameters
        self.goal_point = goal_point #m
        self.stopping_dist = stopping_dist #m

        #Trajectory Simulation Parameters
        self.timestep = 0.5 #s (reduced from 1.0 to avoid large jumps)
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
        #Goal biasing: 5% chance of sampling the goal directly
        if np.random.random() < 0.05:
            return self.goal_point.copy()
        x = np.random.uniform(self.bounds[0, 0], self.bounds[0, 1])
        y = np.random.uniform(self.bounds[1, 0], self.bounds[1, 1])
        return np.array([[x], [y]])
    
    def check_if_duplicate(self, point):
        #Check if point is a duplicate of an already existing node
        threshold = 0.1 #m
        for node in self.nodes:
            if np.linalg.norm(point[:2, 0] - node.point[:2, 0]) < threshold:
                return True
        return False
    
    def closest_node(self, point):
        #Returns the index of the closest node
        #point is a 2 by 1 vector [x; y]
        all_points = np.hstack([n.point[:2] for n in self.nodes])  # 2 x N
        dists = np.linalg.norm(all_points - point[:2], axis=0)
        return int(np.argmin(dists))
    
    def simulate_trajectory(self, node_i, point_s):
        #Simulates the non-holonomic motion of the robot.
        #This function drives the robot from node_i towards point_s. This function does has many solutions!
        #node_i is a 3 by 1 vector [x;y;theta] this can be used to construct the SE(2) matrix T_{OI} in course notation
        #point_s is the sampled point vector [x; y]
        vel, rot_vel = self.robot_controller(node_i, point_s)
        robot_traj = self.trajectory_rollout(vel, rot_vel)

        # Transform from local frame (origin at 0,0,0) to world frame (origin at node_i)
        theta_i = node_i[2, 0]
        cos_t = np.cos(theta_i)
        sin_t = np.sin(theta_i)

        world_traj = np.zeros_like(robot_traj)
        world_traj[0, :] = cos_t * robot_traj[0, :] - sin_t * robot_traj[1, :] + node_i[0, 0]
        world_traj[1, :] = sin_t * robot_traj[0, :] + cos_t * robot_traj[1, :] + node_i[1, 0]
        world_traj[2, :] = robot_traj[2, :] + theta_i

        return world_traj
    
    def robot_controller(self, node_i, point_s):
        #This controller determines the velocities that will nominally move the robot from node i to node s
        #Max velocities should be enforced
        dx = point_s[0, 0] - node_i[0, 0]
        dy = point_s[1, 0] - node_i[1, 0]

        # Desired heading toward target
        angle_to_target = np.arctan2(dy, dx)
        # Angle error relative to current heading
        angle_error = angle_to_target - node_i[2, 0]
        # Normalize to [-pi, pi]
        angle_error = (angle_error + np.pi) % (2 * np.pi) - np.pi

        # Rotational velocity: proportional to angle error, clamped
        rot_vel = np.clip(angle_error / self.timestep, -self.rot_vel_max, self.rot_vel_max)
        # Linear velocity: scale by cos(angle_error) so robot slows when not facing target
        vel = self.vel_max * np.clip(np.cos(angle_error), 0.0, 1.0)

        return vel, rot_vel
    
    def trajectory_rollout(self, vel, rot_vel):
        # Given your chosen velocities determine the trajectory of the robot for your given timestep
        # The returned trajectory should be a series of points to check for collisions
        # Simulates from (0, 0, 0) in the robot's local frame using unicycle kinematics
        dt = self.timestep / self.num_substeps
        traj = np.zeros((3, self.num_substeps))
        x, y, theta = 0.0, 0.0, 0.0
        for i in range(self.num_substeps):
            x += vel * np.cos(theta) * dt
            y += vel * np.sin(theta) * dt
            theta += rot_vel * dt
            traj[:, i] = [x, y, theta]
        return traj
    
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

        all_rows = []
        all_cols = []
        for i in range(cells.shape[1]):
            rr, cc = disk((cells[0, i], cells[1, i]), radius_pixels, shape=self.occupancy_map.shape)
            all_rows.append(rr)
            all_cols.append(cc)

        return np.concatenate(all_rows), np.concatenate(all_cols)
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
        #Ignores the ending theta constraint as per the lab hint
        return self.simulate_trajectory(node_i, point_f)
    
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

            #Check if trajectory stays within map bounds
            if np.any(trajectory_o[0, :] < self.bounds[0, 0]) or \
               np.any(trajectory_o[0, :] > self.bounds[0, 1]) or \
               np.any(trajectory_o[1, :] < self.bounds[1, 0]) or \
               np.any(trajectory_o[1, :] > self.bounds[1, 1]):
                continue

            #Check for collisions along trajectory
            rows, cols = self.points_to_robot_circle(trajectory_o[:2, :])
            if len(rows) == 0 or not np.all(self.occupancy_map[rows, cols] > 0.5):
                continue

            #Get new node from trajectory endpoint
            new_point = trajectory_o[:, -1:].copy()

            #Skip if robot barely moved (e.g. pure rotation)
            if np.linalg.norm(new_point[:2, 0] - self.nodes[closest_node_id].point[:2, 0]) < 0.01:
                continue

            #Check if duplicate
            if self.check_if_duplicate(new_point):
                continue

            #Add new node to tree
            new_node = Node(new_point, closest_node_id, 0)
            self.nodes[closest_node_id].children_ids.append(len(self.nodes))
            self.nodes.append(new_node)

            #Visualize
            self.window.add_point(new_point[:2].flatten(), radius=2)

            if i % 100 == 0:
                print("RRT iteration %d, nodes: %d" % (i, len(self.nodes)))

            #Check if goal has been reached
            dist_to_goal = np.linalg.norm(new_point[:2] - self.goal_point)
            if dist_to_goal < self.stopping_dist:
                print("RRT: Goal reached at iteration %d with %d nodes!" % (i, len(self.nodes)))
                break

        return self.nodes
    
    def rrt_star_planning(self):
        #This function performs RRT* for the given map and robot
        print(f"\n=== RRT* PLANNING DEBUG ===")
        print(f"Start: {self.nodes[0].point.flatten()}")
        print(f"Goal: {self.goal_point.flatten()}")
        print(f"Distance to goal: {np.linalg.norm(self.nodes[0].point[:2] - self.goal_point):.2f}m")
        print(f"Stopping distance: {self.stopping_dist}m")
        print(f"Map bounds: x=[{self.bounds[0,0]:.1f}, {self.bounds[0,1]:.1f}], y=[{self.bounds[1,0]:.1f}, {self.bounds[1,1]:.1f}]")
        print(f"Robot radius: {self.robot_radius}m")
        print(f"Max vel: {self.vel_max}m/s, max rot: {self.rot_vel_max}rad/s")
        print(f"Timestep: {self.timestep}s")

        # Verify starting position is collision-free
        start_cell = self.point_to_cell(self.nodes[0].point[:2, :])
        start_rows, start_cols = self.points_to_robot_circle(self.nodes[0].point[:2, :])
        if len(start_rows) > 0:
            start_free = np.sum(self.occupancy_map[start_rows, start_cols] > 0.5)
            print(f"Start position check: {start_free}/{len(start_rows)} pixels are free")
            if start_free < len(start_rows):
                print(f"  WARNING: Start position has obstacles!")
        print(f"===========================\n")

        iteration_stats = {'bounds': 0, 'collision': 0, 'small_move': 0, 'duplicate': 0, 'success': 0}

        for i in range(10000): #Most likely need more iterations than this to complete the map!
            #Sample
            point = self.sample_map_space()

            #Closest Node
            closest_node_id = self.closest_node(point)

            #Simulate trajectory
            trajectory_o = self.simulate_trajectory(self.nodes[closest_node_id].point, point)

            #Check if trajectory stays within map bounds
            if np.any(trajectory_o[0, :] < self.bounds[0, 0]) or \
               np.any(trajectory_o[0, :] > self.bounds[0, 1]) or \
               np.any(trajectory_o[1, :] < self.bounds[1, 0]) or \
               np.any(trajectory_o[1, :] > self.bounds[1, 1]):
                iteration_stats['bounds'] += 1
                continue

            #Check for Collision
            rows, cols = self.points_to_robot_circle(trajectory_o[:2, :])
            if len(rows) == 0 or not np.all(self.occupancy_map[rows, cols] > 0.5):
                iteration_stats['collision'] += 1
                continue

            #Get new node from trajectory endpoint
            new_point = trajectory_o[:, -1:].copy()

            #Skip if robot barely moved (e.g. pure rotation)
            if np.linalg.norm(new_point[:2, 0] - self.nodes[closest_node_id].point[:2, 0]) < 0.01:
                iteration_stats['small_move'] += 1
                continue

            #Check if duplicate
            if self.check_if_duplicate(new_point):
                iteration_stats['duplicate'] += 1
                continue

            iteration_stats['success'] += 1

            #Last node rewire - find best parent within ball radius
            ball_rad = self.ball_radius()

            # Find all nodes within ball radius
            all_points = np.hstack([n.point[:2] for n in self.nodes])
            dists = np.linalg.norm(all_points - new_point[:2], axis=0)
            near_node_ids = np.where(dists < ball_rad)[0].tolist()

            # Find the best parent (minimum cost)
            best_parent_id = closest_node_id
            best_cost = self.nodes[closest_node_id].cost + self.cost_to_come(trajectory_o)
            best_trajectory = trajectory_o

            for near_id in near_node_ids:
                # Try to connect from this near node to new point
                traj = self.connect_node_to_point(self.nodes[near_id].point, new_point[:2])

                # Check bounds
                if np.any(traj[0, :] < self.bounds[0, 0]) or \
                   np.any(traj[0, :] > self.bounds[0, 1]) or \
                   np.any(traj[1, :] < self.bounds[1, 0]) or \
                   np.any(traj[1, :] > self.bounds[1, 1]):
                    continue

                # Check collision
                rows, cols = self.points_to_robot_circle(traj[:2, :])
                if len(rows) == 0 or not np.all(self.occupancy_map[rows, cols] > 0.5):
                    continue

                # Calculate cost through this node
                cost = self.nodes[near_id].cost + self.cost_to_come(traj)

                if cost < best_cost:
                    best_parent_id = near_id
                    best_cost = cost
                    best_trajectory = traj

            # Add new node with best parent
            new_node = Node(new_point, best_parent_id, best_cost)
            new_node_id = len(self.nodes)
            self.nodes[best_parent_id].children_ids.append(new_node_id)
            self.nodes.append(new_node)

            #Close node rewire - check if nearby nodes can improve by using new node as parent
            for near_id in near_node_ids:
                if near_id == best_parent_id:
                    continue

                # Try to connect from new node to this near node
                traj = self.connect_node_to_point(new_point, self.nodes[near_id].point[:2])

                # Check bounds
                if np.any(traj[0, :] < self.bounds[0, 0]) or \
                   np.any(traj[0, :] > self.bounds[0, 1]) or \
                   np.any(traj[1, :] < self.bounds[1, 0]) or \
                   np.any(traj[1, :] > self.bounds[1, 1]):
                    continue

                # Check collision
                rows, cols = self.points_to_robot_circle(traj[:2, :])
                if len(rows) == 0 or not np.all(self.occupancy_map[rows, cols] > 0.5):
                    continue

                # Calculate cost through new node
                new_cost = new_node.cost + self.cost_to_come(traj)

                # If better, rewire
                if new_cost < self.nodes[near_id].cost:
                    # Remove from old parent's children
                    old_parent_id = self.nodes[near_id].parent_id
                    self.nodes[old_parent_id].children_ids.remove(near_id)

                    # Set new parent
                    self.nodes[near_id].parent_id = new_node_id
                    self.nodes[near_id].cost = new_cost
                    new_node.children_ids.append(near_id)

                    # Update all children costs recursively
                    self.update_children(near_id)

            #Visualize
            self.window.add_point(new_point[:2].flatten(), radius=2)

            if i % 100 == 0:
                success_rate = 100 * iteration_stats['success'] / max(i, 1)
                print(f"RRT* iter {i}: nodes={len(self.nodes)}, ball_r={ball_rad:.3f}, success_rate={success_rate:.1f}%")
                print(f"  Stats: bounds={iteration_stats['bounds']}, collision={iteration_stats['collision']}, " +
                      f"small_move={iteration_stats['small_move']}, duplicate={iteration_stats['duplicate']}")
                # Show current best path to goal
                closest_to_goal = min(range(len(self.nodes)),
                                     key=lambda j: np.linalg.norm(self.nodes[j].point[:2] - self.goal_point))
                dist = np.linalg.norm(self.nodes[closest_to_goal].point[:2] - self.goal_point)
                print(f"  Closest node to goal: {closest_to_goal} at distance {dist:.2f}m")

                # Debug: show a sample trajectory
                if i == 100:
                    print(f"\n  DEBUG: Sample trajectory analysis at iteration {i}:")
                    sample_pt = self.sample_map_space()
                    closest_id = self.closest_node(sample_pt)
                    print(f"    Sampled point: ({sample_pt[0,0]:.2f}, {sample_pt[1,0]:.2f})")
                    print(f"    Closest node {closest_id}: {self.nodes[closest_id].point.flatten()}")
                    traj = self.simulate_trajectory(self.nodes[closest_id].point, sample_pt)
                    print(f"    Trajectory: start={traj[:2,0].flatten()}, end={traj[:2,-1].flatten()}")
                    print(f"    Trajectory length: {self.cost_to_come(traj):.3f}m")
                    rows, cols = self.points_to_robot_circle(traj[:2, :])
                    if len(rows) > 0:
                        free_pixels = np.sum(self.occupancy_map[rows, cols] > 0.5)
                        print(f"    Collision check: {free_pixels}/{len(rows)} pixels are free")
                    print()

            #Check for early end
            dist_to_goal = np.linalg.norm(new_point[:2] - self.goal_point)
            if dist_to_goal < self.stopping_dist:
                print(f"\n*** RRT*: GOAL REACHED at iteration {i} with {len(self.nodes)} nodes! ***")
                print(f"Final node: {new_point.flatten()}")
                print(f"Distance to goal: {dist_to_goal:.3f}m")
                break

        if i == 9999:
            print(f"\n*** RRT*: Max iterations reached without finding goal ***")
            print(f"Total nodes created: {len(self.nodes)}")
            print(f"Closest distance to goal: {min(np.linalg.norm(n.point[:2] - self.goal_point) for n in self.nodes):.2f}m")

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
    goal_point = np.array([[42.05], [-44]]) #m
    stopping_dist = 0.5 #m

    #RRT* planning
    path_planner = PathPlanner(map_filename, map_setings_filename, goal_point, stopping_dist)
    nodes = path_planner.rrt_star_planning()
    node_path_metric = np.hstack(path_planner.recover_path())

    #Save path for trajectory rollout
    np.save("path.npy", node_path_metric)
    print(f"Path saved with {node_path_metric.shape[1]} waypoints")


if __name__ == '__main__':
    main()
