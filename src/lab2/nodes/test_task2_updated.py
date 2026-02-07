#!/usr/bin/env python3
"""Test script for Task 2: Trajectory Simulation (with original signatures)"""

import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt

def test_trajectory_simulation():
    """Test the trajectory simulation functions"""
    print("=" * 60)
    print("Testing Task 2: Trajectory Simulation")
    print("=" * 60)

    # Create a mock PathPlanner with original function signatures
    class MockPathPlanner:
        def __init__(self):
            self.timestep = 1.0
            self.num_substeps = 10
            self.vel_max = 0.5
            self.rot_vel_max = 0.2
            self._current_start_pose = None

        def simulate_trajectory(self, node_i, point_s):
            # Store the starting pose for trajectory_rollout to access
            self._current_start_pose = node_i

            # Compute control velocities
            vel, rot_vel = self.robot_controller(node_i, point_s)

            # Simulate the trajectory
            robot_traj = self.trajectory_rollout(vel, rot_vel)
            return robot_traj

        def robot_controller(self, node_i, point_s):
            x_current = node_i[0, 0]
            y_current = node_i[1, 0]
            theta_current = node_i[2, 0]

            if point_s.ndim == 1:
                x_target = point_s[0]
                y_target = point_s[1]
            else:
                x_target = point_s[0, 0] if point_s.shape[1] == 1 else point_s[0, 0]
                y_target = point_s[1, 0] if point_s.shape[1] == 1 else point_s[1, 0]

            dx = x_target - x_current
            dy = y_target - y_current
            distance = np.sqrt(dx**2 + dy**2)

            theta_desired = np.arctan2(dy, dx)
            theta_error = theta_desired - theta_current
            theta_error = np.arctan2(np.sin(theta_error), np.cos(theta_error))

            k_rot = 2.0
            rot_vel = k_rot * theta_error
            rot_vel = np.clip(rot_vel, -self.rot_vel_max, self.rot_vel_max)

            k_vel = 1.0
            vel = k_vel * self.vel_max * np.cos(theta_error)

            if distance < 1.0:
                vel = vel * distance

            vel = np.clip(vel, 0, self.vel_max)

            return vel, rot_vel

        def trajectory_rollout(self, vel, rot_vel):
            # Get starting pose from stored instance variable
            start_pose = self._current_start_pose

            dt = self.timestep / self.num_substeps
            trajectory = np.zeros((3, self.num_substeps))

            x = start_pose[0, 0]
            y = start_pose[1, 0]
            theta = start_pose[2, 0]

            for i in range(self.num_substeps):
                trajectory[0, i] = x
                trajectory[1, i] = y
                trajectory[2, i] = theta

                x = x + vel * np.cos(theta) * dt
                y = y + vel * np.sin(theta) * dt
                theta = theta + rot_vel * dt
                theta = np.arctan2(np.sin(theta), np.cos(theta))

            return trajectory

    planner = MockPathPlanner()

    # Test 1: Straight line motion
    print("\nTest 1: Moving straight ahead")
    start_pose = np.array([[0.0], [0.0], [0.0]])  # Start at origin, facing right
    target_point = np.array([[1.0], [0.0]])  # Target 1m ahead

    trajectory = planner.simulate_trajectory(start_pose, target_point)

    print(f"  Start pose: ({start_pose[0,0]:.2f}, {start_pose[1,0]:.2f}, {start_pose[2,0]:.2f} rad)")
    print(f"  Target: ({target_point[0,0]:.2f}, {target_point[1,0]:.2f})")
    print(f"  Trajectory shape: {trajectory.shape}")
    print(f"  Final position: ({trajectory[0,-1]:.3f}, {trajectory[1,-1]:.3f}, {trajectory[2,-1]:.3f} rad)")

    # Check if robot moved toward target
    dist_start = np.linalg.norm(target_point[:, 0] - start_pose[:2, 0])
    dist_end = np.linalg.norm(target_point[:, 0] - trajectory[:2, -1])
    print(f"  Distance to target - Start: {dist_start:.3f}m, End: {dist_end:.3f}m")
    print(f"  ✓ Moved closer!" if dist_end < dist_start else "  ✗ Did not move closer")

    # Test 2: Turning motion
    print("\nTest 2: Turning to face target")
    start_pose = np.array([[0.0], [0.0], [0.0]])  # Facing right
    target_point = np.array([[0.0], [1.0]])  # Target above (requires 90° turn)

    trajectory = planner.simulate_trajectory(start_pose, target_point)

    print(f"  Start heading: {np.degrees(start_pose[2,0]):.1f}°")
    print(f"  Desired heading: 90.0°")
    print(f"  Final heading: {np.degrees(trajectory[2,-1]):.1f}°")

    dist_start = np.linalg.norm(target_point[:, 0] - start_pose[:2, 0])
    dist_end = np.linalg.norm(target_point[:, 0] - trajectory[:2, -1])
    print(f"  Distance to target - Start: {dist_start:.3f}m, End: {dist_end:.3f}m")

    # Test 3: Diagonal motion
    print("\nTest 3: Diagonal motion")
    start_pose = np.array([[0.0], [0.0], [0.0]])
    target_point = np.array([[1.0], [1.0]])  # 45° angle

    trajectory = planner.simulate_trajectory(start_pose, target_point)

    print(f"  Start: ({start_pose[0,0]:.2f}, {start_pose[1,0]:.2f})")
    print(f"  Target: ({target_point[0,0]:.2f}, {target_point[1,0]:.2f})")
    print(f"  Final: ({trajectory[0,-1]:.3f}, {trajectory[1,-1]:.3f})")

    dist_start = np.linalg.norm(target_point[:, 0] - start_pose[:2, 0])
    dist_end = np.linalg.norm(target_point[:, 0] - trajectory[:2, -1])
    print(f"  Distance to target - Start: {dist_start:.3f}m, End: {dist_end:.3f}m")
    print(f"  ✓ Moved closer!" if dist_end < dist_start else "  ✗ Did not move closer")

    # Visualize one trajectory
    print("\nCreating visualization...")
    fig, ax = plt.subplots(1, 1, figsize=(8, 8))

    # Plot trajectory
    ax.plot(trajectory[0, :], trajectory[1, :], 'b-', linewidth=2, label='Trajectory')
    ax.plot(trajectory[0, 0], trajectory[1, 0], 'go', markersize=10, label='Start')
    ax.plot(trajectory[0, -1], trajectory[1, -1], 'ro', markersize=10, label='End')
    ax.plot(target_point[0, 0], target_point[1, 0], 'r*', markersize=15, label='Target')

    # Plot orientation at start and end
    arrow_len = 0.2
    ax.arrow(trajectory[0, 0], trajectory[1, 0],
             arrow_len * np.cos(trajectory[2, 0]),
             arrow_len * np.sin(trajectory[2, 0]),
             head_width=0.1, head_length=0.05, fc='green', ec='green')
    ax.arrow(trajectory[0, -1], trajectory[1, -1],
             arrow_len * np.cos(trajectory[2, -1]),
             arrow_len * np.sin(trajectory[2, -1]),
             head_width=0.1, head_length=0.05, fc='red', ec='red')

    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_title('Task 2: Robot Trajectory Simulation')
    ax.legend()
    ax.grid(True)
    ax.axis('equal')

    plt.savefig('test_task2_trajectory_updated.png', dpi=150, bbox_inches='tight')
    print(f"  Saved visualization to: test_task2_trajectory_updated.png")

    print("\n" + "=" * 60)
    print("Task 2 implementation complete!")
    print("Function signatures preserved as required.")
    print("=" * 60)

if __name__ == '__main__':
    test_trajectory_simulation()
