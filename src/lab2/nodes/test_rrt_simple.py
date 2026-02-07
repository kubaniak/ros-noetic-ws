#!/usr/bin/env python3
"""Test RRT on simple map"""

import sys
sys.path.insert(0, '/home/rob521_4/catkin_ws/src/lab2/nodes')

from l2_planning import PathPlanner
import numpy as np

def main():
    print("=" * 60)
    print("Testing RRT on Simple Map")
    print("=" * 60)

    # Use simple map for initial testing
    map_filename = "simple_map.png"
    map_settings_filename = "willowgarageworld_05res.yaml"  # Use same settings

    # Set goal point (adjust based on simple map)
    goal_point = np.array([[5], [5]])  # Shorter distance for testing
    stopping_dist = 0.5

    print(f"\nMap: {map_filename}")
    print(f"Goal: ({goal_point[0,0]}, {goal_point[1,0]}) m")
    print(f"Stopping distance: {stopping_dist} m")
    print("\nRunning RRT planning...")
    print("-" * 60)

    try:
        # Create path planner
        path_planner = PathPlanner(map_filename, map_settings_filename, goal_point, stopping_dist)

        print(f"Map bounds: X=[{path_planner.bounds[0,0]:.1f}, {path_planner.bounds[0,1]:.1f}], "
              f"Y=[{path_planner.bounds[1,0]:.1f}, {path_planner.bounds[1,1]:.1f}]")
        print(f"Start node: ({path_planner.nodes[0].point[0,0]}, {path_planner.nodes[0].point[1,0]})")

        # Run RRT planning
        nodes = path_planner.rrt_planning()

        # Recover path
        path = path_planner.recover_path()
        node_path_metric = np.hstack(path)

        # Save path
        np.save("shortest_path.npy", node_path_metric)

        print("-" * 60)
        print(f"\n✓ Path planning complete!")
        print(f"  Total nodes created: {len(nodes)}")
        print(f"  Path length: {len(path)} waypoints")
        print(f"  Path saved to: shortest_path.npy")
        print(f"  Path shape: {node_path_metric.shape}")

        # Print first and last few waypoints
        print(f"\nFirst waypoint: ({node_path_metric[0,0]:.2f}, {node_path_metric[1,0]:.2f})")
        print(f"Last waypoint: ({node_path_metric[0,-1]:.2f}, {node_path_metric[1,-1]:.2f})")

    except Exception as e:
        print(f"\n✗ Error during planning: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    main()
