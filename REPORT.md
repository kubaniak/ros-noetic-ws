# Lab 3: LiDAR Mapping with Wheel Odometry
## Implementation Details

### Dead-reckoning Pose Estimator (`l3_estimate_robot_motion.py`)
To estimate the robot's pose and velocity based on wheel encoder data:
1. First, we calculated the difference in encoder ticks between the current and last recorded measurements. The difference was computed carefully taking the 32-bit integer overflow into account.
2. We mapped the encoder tick differences to actual wheel displacement using the given conversion `RAD_PER_TICK` and `WHEEL_RADIUS` parameters for both the left and right wheels (`d_l`, `d_r`).
3. Using the differential drive kinematics equations, we computed the displacement in the center of the wheels `d_c = (d_l + d_r) / 2.0` and the change in heading `d_theta = (d_r - d_l) / (2.0 * BASELINE)`.
4. We then updated the `self.pose` correctly by shifting X and Y by `d_c * cos(theta + d_theta/2.0)` and `d_c * sin(theta + d_theta/2.0)` respectively.
5. The velocity (`self.twist`) was updated by dividing `d_c` and `d_theta` by the time delta of the timestamp (`del_time`).

### Occupancy Grid Mapping (`l3_mapping.py`)
To construct an occupancy grid map based on the 2D laser scan data:
1. In `scan_cb`, we retrieve the current robot base coordinate in the map frame `odom_map`.
2. We then calculate the base cell indices `x_start` and `y_start` of the map using the current `odom_map` location relative to the map origin.
3. Looping through each valid valid range measurement inside `scan_msg.ranges`:
   - An infinite or out-of-bounds reading means the ray did not hit an obstacle up to `range_max`, so we cap the distance to `scan_msg.range_max`.
   - The absolute angle of the LiDAR ray is found by taking the angle of the ray in the base_scan frame, rotating it by the base_scan orientation relative to base_link, and rotating by the robot's heading relative to the odom_map frame (`odom_map[2] + scan_euler[2] + angle_in_scan`).
   - We pass these parameters into `ray_trace_update`.
4. Inside `ray_trace_update`, we find the grid indices of the endpoint of the ray using `range_mes`, `CELL_SIZE`, and the global `angle`. We use `skimage.draw.line` to get all cells between the robot position and the endpoint.
5. We decrease `log_odds` for all cells on the ray by `BETA` to indicate increased probability of free space.
6. If the ray hits an obstacle (i.e. if `range_mes < 3.49`), we consider the last `NUM_PTS_OBSTACLE` cells on the ray to contain the obstacle and increase the `log_odds` by `ALPHA + BETA` (cancelling out the free space assumption of `- BETA`).
7. Finally, we convert the `log_odds` back to probability values ranging from 0 to 100 and write them to `self.np_map`.

---

## TODO List: Experiments for the Real Robot (Section 4.2)
The tasks below summarize what needs to be tested on the real Turtlebot3 during the lab session:

- [ ] **Task 4: Calibrate the TurtleBot**
  - Drive the robot in a straight line for a known distance (e.g., 4 ft / 1.2192 m) to calibrate the wheel radius. Compare against the provided `straight_line.bag`.
  - Rotate the robot in place for a known number of rotations (e.g., 3 rotations) to calculate the wheel baseline. Compare against `three_rotations.bag`.
  - Record the values to answer questions in the report regarding biases, uncertainty, and deviation from factory calibration data.

- [ ] **Task 5: Motion Estimation**
  - On the TurtleBot PC, bring up the robot (`roslaunch turtlebot3_bringup turtlebot3_robot.launch --screen`).
  - Run the node (`rosrun rob521_lab3 l3_estimate_robot_motion.py`).
  - Launch rviz (`roslaunch rob521_lab3 wheel_odom_rviz.launch`) and compare the estimated `/wheel_odom` pose with the TurtleBot's onboard `/odom` pose.
  - Experiment 1: Teleoperate the robot in a circle with a radius of approximately 1 meter and return to the start point.
  - Experiment 2: Drive the robot in a more complex path. Take note of any motions that lead to significant odometry errors.
  - Save the `.bag` file (`motion_estimate.bag`) and plot trajectories using `l3_plot_motion_estimate.py` for both experiments to include in the report.
  - *Note: Discuss possible sources of errors in the report.*

- [ ] **Task 6: Mapping (Live Demo)**
  - Ensure the Turtlebot is brought up and running.
  - Launch mapping rviz (`roslaunch rob521_lab3 mapping_rviz.launch`).
  - Start the mapping node (`rosrun rob521_lab3 l3_mapping.py`).
  - Teleoperate the robot to explore and map the maze in Myhal 570.
  - Provide a live mapping demo to the TA.
  - Capture a picture of the map and analyze differences in error sources compared to the simulation map.
