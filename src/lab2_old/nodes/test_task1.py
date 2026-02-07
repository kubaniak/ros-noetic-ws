#!/usr/bin/env python3
"""Test script for Task 1: Collision Detection"""

import numpy as np
import sys
sys.path.append('/home/rob521_4/catkin_ws/src/lab2/nodes')
from l2_planning import PathPlanner

def test_point_to_cell():
    """Test the point_to_cell function"""
    print("=" * 60)
    print("Testing point_to_cell function")
    print("=" * 60)

    # Initialize path planner
    map_filename = "willowgarageworld_05res.png"
    map_settings_filename = "willowgarageworld_05res.yaml"
    goal_point = np.array([[10], [10]])
    stopping_dist = 0.5

    path_planner = PathPlanner(map_filename, map_settings_filename, goal_point, stopping_dist)

    print(f"\nMap shape: {path_planner.map_shape}")
    print(f"Resolution: {path_planner.map_settings_dict['resolution']} m/pixel")
    print(f"Origin: {path_planner.map_settings_dict['origin']}")
    print(f"Robot radius: {path_planner.robot_radius} m")

    # Test 1: Origin point should map to bottom-left corner
    origin_x = path_planner.map_settings_dict["origin"][0]
    origin_y = path_planner.map_settings_dict["origin"][1]
    test_point1 = np.array([[origin_x], [origin_y]])

    cell1 = path_planner.point_to_cell(test_point1)
    print(f"\nTest 1 - Origin point:")
    print(f"  World coords: ({origin_x}, {origin_y})")
    print(f"  Cell indices: row={cell1[0,0]}, col={cell1[1,0]}")
    print(f"  Expected: row={path_planner.map_shape[0]-1}, col=0")

    # Test 2: A point in the middle
    mid_x = origin_x + path_planner.map_shape[1] * path_planner.map_settings_dict["resolution"] / 2
    mid_y = origin_y + path_planner.map_shape[0] * path_planner.map_settings_dict["resolution"] / 2
    test_point2 = np.array([[mid_x], [mid_y]])

    cell2 = path_planner.point_to_cell(test_point2)
    print(f"\nTest 2 - Middle point:")
    print(f"  World coords: ({mid_x:.2f}, {mid_y:.2f})")
    print(f"  Cell indices: row={cell2[0,0]}, col={cell2[1,0]}")
    print(f"  Expected approximately: row={path_planner.map_shape[0]//2}, col={path_planner.map_shape[1]//2}")

    # Test 3: Multiple points
    test_points = np.array([[origin_x, mid_x], [origin_y, mid_y]])
    cells = path_planner.point_to_cell(test_points)
    print(f"\nTest 3 - Multiple points:")
    print(f"  Input shape: {test_points.shape}")
    print(f"  Output shape: {cells.shape}")
    print(f"  Cell indices:\n{cells}")

def test_points_to_robot_circle():
    """Test the points_to_robot_circle function"""
    print("\n" + "=" * 60)
    print("Testing points_to_robot_circle function")
    print("=" * 60)

    # Initialize path planner
    map_filename = "willowgarageworld_05res.png"
    map_settings_filename = "willowgarageworld_05res.yaml"
    goal_point = np.array([[10], [10]])
    stopping_dist = 0.5

    path_planner = PathPlanner(map_filename, map_settings_filename, goal_point, stopping_dist)

    # Test with a single point
    test_point = np.array([[0.0], [0.0]])
    rows, cols = path_planner.points_to_robot_circle(test_point)

    print(f"\nTest - Single point at (0.0, 0.0):")
    print(f"  Robot radius: {path_planner.robot_radius} m")
    print(f"  Robot radius in pixels: {path_planner.robot_radius / path_planner.map_settings_dict['resolution']:.2f}")
    print(f"  Number of cells occupied: {len(rows)}")
    print(f"  Expected approximately: {np.pi * (path_planner.robot_radius / path_planner.map_settings_dict['resolution'])**2:.0f} cells")

    # Verify all indices are within bounds
    if len(rows) > 0:
        print(f"  Row range: [{rows.min()}, {rows.max()}]")
        print(f"  Col range: [{cols.min()}, {cols.max()}]")
        print(f"  All rows in bounds: {(rows >= 0).all() and (rows < path_planner.map_shape[0]).all()}")
        print(f"  All cols in bounds: {(cols >= 0).all() and (cols < path_planner.map_shape[1]).all()}")

    print("\n" + "=" * 60)
    print("Task 1 implementation complete!")
    print("=" * 60)

if __name__ == '__main__':
    test_point_to_cell()
    test_points_to_robot_circle()
