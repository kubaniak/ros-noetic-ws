#!/usr/bin/env python3
import numpy as np
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

# Temporarily reduce iterations for testing
import l2_planning

# Monkey patch to reduce iterations for testing
original_main = l2_planning.main

def test_main():
    map_filename = "willowgarageworld_05res.png"
    map_setings_filename = "willowgarageworld_05res.yaml"

    # Use a closer goal for faster testing
    goal_point = np.array([[10], [-10]])  # Closer goal
    stopping_dist = 0.5

    print("Creating path planner with closer goal for testing...")
    print(f"Goal: {goal_point.flatten()}, Distance from origin: ~14m")

    path_planner = l2_planning.PathPlanner(map_filename, map_setings_filename, goal_point, stopping_dist)

    # Temporarily modify max iterations
    print("Running RRT* with 2000 iterations...")

    # Store original code, run limited version
    for i in range(2000):
        point = path_planner.sample_map_space()
        closest_node_id = path_planner.closest_node(point)
        trajectory_o = path_planner.simulate_trajectory(path_planner.nodes[closest_node_id].point, point)

        if np.any(trajectory_o[0, :] < path_planner.bounds[0, 0]) or \
           np.any(trajectory_o[0, :] > path_planner.bounds[0, 1]) or \
           np.any(trajectory_o[1, :] < path_planner.bounds[1, 0]) or \
           np.any(trajectory_o[1, :] > path_planner.bounds[1, 1]):
            continue

        rows, cols = path_planner.points_to_robot_circle(trajectory_o[:2, :])
        if len(rows) == 0 or not np.all(path_planner.occupancy_map[rows, cols] > 0.5):
            continue

        new_point = trajectory_o[:, -1:].copy()

        if np.linalg.norm(new_point[:2, 0] - path_planner.nodes[closest_node_id].point[:2, 0]) < 0.01:
            continue

        if path_planner.check_if_duplicate(new_point):
            continue

        # RRT* logic
        ball_rad = path_planner.ball_radius()
        all_points = np.hstack([n.point[:2] for n in path_planner.nodes])
        dists = np.linalg.norm(all_points - new_point[:2], axis=0)
        near_node_ids = np.where(dists < ball_rad)[0].tolist()

        best_parent_id = closest_node_id
        best_cost = path_planner.nodes[closest_node_id].cost + path_planner.cost_to_come(trajectory_o)
        best_trajectory = trajectory_o

        for near_id in near_node_ids:
            traj = path_planner.connect_node_to_point(path_planner.nodes[near_id].point, new_point[:2])
            if np.any(traj[0, :] < path_planner.bounds[0, 0]) or \
               np.any(traj[0, :] > path_planner.bounds[0, 1]) or \
               np.any(traj[1, :] < path_planner.bounds[1, 0]) or \
               np.any(traj[1, :] > path_planner.bounds[1, 1]):
                continue
            rows, cols = path_planner.points_to_robot_circle(traj[:2, :])
            if len(rows) == 0 or not np.all(path_planner.occupancy_map[rows, cols] > 0.5):
                continue
            cost = path_planner.nodes[near_id].cost + path_planner.cost_to_come(traj)
            if cost < best_cost:
                best_parent_id = near_id
                best_cost = cost
                best_trajectory = traj

        new_node = l2_planning.Node(new_point, best_parent_id, best_cost)
        new_node_id = len(path_planner.nodes)
        path_planner.nodes[best_parent_id].children_ids.append(new_node_id)
        path_planner.nodes.append(new_node)

        # Rewiring
        for near_id in near_node_ids:
            if near_id == best_parent_id:
                continue
            traj = path_planner.connect_node_to_point(new_point, path_planner.nodes[near_id].point[:2])
            if np.any(traj[0, :] < path_planner.bounds[0, 0]) or \
               np.any(traj[0, :] > path_planner.bounds[0, 1]) or \
               np.any(traj[1, :] < path_planner.bounds[1, 0]) or \
               np.any(traj[1, :] > path_planner.bounds[1, 1]):
                continue
            rows, cols = path_planner.points_to_robot_circle(traj[:2, :])
            if len(rows) == 0 or not np.all(path_planner.occupancy_map[rows, cols] > 0.5):
                continue
            new_cost = new_node.cost + path_planner.cost_to_come(traj)
            if new_cost < path_planner.nodes[near_id].cost:
                old_parent_id = path_planner.nodes[near_id].parent_id
                path_planner.nodes[old_parent_id].children_ids.remove(near_id)
                path_planner.nodes[near_id].parent_id = new_node_id
                path_planner.nodes[near_id].cost = new_cost
                new_node.children_ids.append(near_id)
                path_planner.update_children(near_id)

        path_planner.window.add_point(new_point[:2].flatten(), radius=2)

        if i % 200 == 0:
            print(f"RRT* iteration {i}, nodes: {len(path_planner.nodes)}, ball_radius: {ball_rad:.3f}")

        dist_to_goal = np.linalg.norm(new_point[:2] - path_planner.goal_point)
        if dist_to_goal < path_planner.stopping_dist:
            print(f"RRT*: Goal reached at iteration {i} with {len(path_planner.nodes)} nodes!")
            break

    print(f"\nPath generation complete!")
    print(f"Total nodes: {len(path_planner.nodes)}")

    node_path_metric = np.hstack(path_planner.recover_path())
    np.save("path.npy", node_path_metric)
    print(f"Path saved with {node_path_metric.shape[1]} waypoints")
    print(f"Path file: {os.path.abspath('path.npy')}")

if __name__ == '__main__':
    test_main()
