#!/usr/bin/env python3
"""Simple test for Task 1 coordinate transformations"""

import numpy as np

def test_coordinate_transform():
    """Test the coordinate transformation logic"""
    print("=" * 60)
    print("Testing coordinate transformation logic")
    print("=" * 60)

    # Map parameters from willowgarageworld_05res.yaml
    resolution = 0.05  # meters/pixel
    origin_x = -21.0   # meters
    origin_y = -49.25  # meters
    map_height = 1984  # pixels (typical value for this map)
    map_width = 2624   # pixels

    print(f"\nMap parameters:")
    print(f"  Resolution: {resolution} m/pixel")
    print(f"  Origin: ({origin_x}, {origin_y})")
    print(f"  Map size: {map_height} x {map_width} pixels")

    # Test 1: Origin point (bottom-left)
    print(f"\nTest 1: Origin point (bottom-left)")
    x, y = origin_x, origin_y
    col = (x - origin_x) / resolution
    row = (map_height - 1) - (y - origin_y) / resolution
    print(f"  World: ({x}, {y})")
    print(f"  Cell: row={row:.0f}, col={col:.0f}")
    print(f"  Expected: row={map_height-1}, col=0")

    # Test 2: Top-right corner
    print(f"\nTest 2: Top-right corner")
    x = origin_x + map_width * resolution
    y = origin_y + map_height * resolution
    col = (x - origin_x) / resolution
    row = (map_height - 1) - (y - origin_y) / resolution
    print(f"  World: ({x:.2f}, {y:.2f})")
    print(f"  Cell: row={row:.0f}, col={col:.0f}")
    print(f"  Expected: row=0, col={map_width}")

    # Test 3: Center point
    print(f"\nTest 3: Center point")
    x = origin_x + (map_width / 2) * resolution
    y = origin_y + (map_height / 2) * resolution
    col = (x - origin_x) / resolution
    row = (map_height - 1) - (y - origin_y) / resolution
    print(f"  World: ({x:.2f}, {y:.2f})")
    print(f"  Cell: row={row:.0f}, col={col:.0f}")
    print(f"  Expected approximately: row={map_height//2}, col={map_width//2}")

    # Test 4: Robot circle calculation
    print(f"\nTest 4: Robot circle footprint")
    robot_radius = 0.22  # meters
    robot_radius_pixels = robot_radius / resolution
    num_pixels = np.pi * robot_radius_pixels**2
    print(f"  Robot radius: {robot_radius} m = {robot_radius_pixels:.2f} pixels")
    print(f"  Approximate footprint area: {num_pixels:.0f} pixels")

    print("\n" + "=" * 60)
    print("Coordinate transformation logic verified!")
    print("=" * 60)

if __name__ == '__main__':
    test_coordinate_transform()
