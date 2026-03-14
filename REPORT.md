# Lab 3: Vehicle Calibration, Odometry, and Mapping

## Implementation Strategy

### 1. `l3_estimate_wheel_radius.py`
The wheel radius is estimated by measuring the total distance driven by the robot in a straight line, and correlating it with the total rotations measured by the left and right wheel encoders.
We average the encoder count increments to reduce error: `avg_encoder_ticks = (del_left_encoder + del_right_encoder) / 2.0`.
We then convert ticks to rotations by dividing by `TICKS_PER_ROTATION = 4096`.
The radius is derived using the standard formula `radius = DRIVEN_DISTANCE / (rotations * 2 * pi)`.

### 2. `l3_estimate_wheel_baseline.py`
The wheel baseline (separation between the two wheels) is estimated by having the robot spin in place. The distance driven by each wheel during the rotation is computed using the previously estimated wheel radius.
The total distance is calculated as `rotations * WHEEL_RADIUS`, where `rotations = (del_right_encoder - del_left_encoder) / TICKS_PER_ROTATION`.
Since the robot performs a known number of rotations (`NUM_ROTATIONS`), we calculate the baseline as `separation = distance / NUM_ROTATIONS`. Note that `WHEEL_RADIUS` needs to be provided manually from the output of the first calibration step.

### 3. `l3_estimate_robot_motion.py`
To estimate robot motion, we implement standard differential drive kinematics (dead reckoning) using the calibrated `WHEEL_RADIUS` and `BASELINE` parameters.
1. The difference in encoder counts between consecutive time steps is converted to distances driven by each wheel (`d_l`, `d_r`).
2. We compute the linear distance `d = (d_r + d_l) / 2.0` and the angular change `th = (d_r - d_l) / (2.0 * BASELINE)`.
3. We use Euler integration to update the 2D position `(x, y)`: `x += d * cos(theta)`, `y += d * sin(theta)`, and update the heading `theta += th`.
4. We estimate the linear and angular velocities (`v = d/dt`, `w = th/dt`) and populate `self.pose` and `self.twist` for the `Odometry` message.

### 4. `l3_mapping.py`
The occupancy grid mapping algorithm is based on log-odds belief updating using ray tracing.
1. In `scan_cb`, we extract the robot's pose in the map frame and convert it to grid pixel coordinates `(x_start, y_start)`. We iterate through downsampled `scan_msg.ranges` to find valid measurements.
2. For each measurement, we compute the global ray angle (`robot_angle + ray_angle`) and call `ray_trace_update`.
3. In `ray_trace_update`, we compute the ray's endpoint `(x_end, y_end)` using the range measurement. We then trace the ray using `skimage.draw.line`.
4. We update the `log_odds` array: we decrement the log-odds for the ray's path representing free space (up to `NUM_PTS_OBSTACLE` before the end) by `BETA`, and we increment the log-odds at the end of the ray representing the obstacle by `ALPHA`.
5. Finally, we compute probabilities from log-odds and update `self.np_map` (0-100 values) for the `OccupancyGrid` message.

## TODO List for Real Robot Testing
1. **Calibrate Wheel Radius:**
   - Run `rosrun rob521_lab3 l3_estimate_wheel_radius.py`.
   - Run the straight-line driving bag/experiment (`straight_line.bag`).
   - Note the printed `Calibrated Radius`.
2. **Calibrate Wheel Baseline:**
   - Open `l3_estimate_wheel_baseline.py` and replace `WHEEL_RADIUS = 0.066 / 2` with the newly calibrated radius value.
   - Run `rosrun rob521_lab3 l3_estimate_wheel_baseline.py`.
   - Run the rotation bag/experiment (`three_rotations.bag`).
   - Note the printed `Calibrated Separation`.
3. **Configure Motion Estimator:**
   - Update `WHEEL_RADIUS` and `BASELINE` in `l3_estimate_robot_motion.py` using the values obtained in Steps 1 and 2.
4. **Test Odometry on Real Robot:**
   - Drive the real robot in a 1m circle and return to start.
   - Run `roslaunch rob521_lab3 wheel_odom_rviz.launch` to visualize and compare estimated `/wheel_odom` vs. TurtleBot's `/odom`.
   - Check if `motion_estimate.bag` accurately reflects the real-world trajectory. Use `l3_plot_motion_estimate.py` to generate the report plots.
5. **Test Mapping on Real Robot:**
   - Run `roslaunch rob521_lab3 mapping_rviz.launch`.
   - Run `rosrun rob521_lab3 l3_mapping.py`.
   - Teleoperate the robot to map the maze in Myhal 570.
   - Save the `map.png` to include in the final deliverable.
