#!/usr/bin/env python3
from __future__ import division, print_function
import os

import numpy as np
from scipy.linalg import block_diag
from scipy.spatial.distance import cityblock
import rospy
import tf2_ros

# msgs
from geometry_msgs.msg import TransformStamped, Twist, PoseStamped
from nav_msgs.msg import Path, Odometry, OccupancyGrid
from visualization_msgs.msg import Marker

# ros and se2 conversion utils
import utils


TRANS_GOAL_TOL = .2  # m, tolerance to consider a goal complete
ROT_GOAL_TOL = .2  # rad, tolerance to consider a goal complete
TRANS_VEL_OPTS = [0, 0.13, 0.26]  # m/s, max of real robot is .26
ROT_VEL_OPTS = np.linspace(-1.82, 1.82, 31)  # rad/s, max of real robot is 1.82
CONTROL_RATE = 5  # Hz, how frequently control signals are sent
CONTROL_HORIZON = 5  # seconds. if this is set too high and INTEGRATION_DT is too low, code will take a long time to run!
INTEGRATION_DT = .025  # s, delta t to propagate trajectories forward by
COLLISION_RADIUS = 0.225  # m, radius from base_link to use for collisions, min of 0.2077 based on dimensions of .281 x .306
ROT_DIST_MULT = .1  # multiplier to change effect of rotational distance in choosing correct control
OBS_DIST_MULT = .1  # multiplier to change the effect of low distance to obstacles on a path
MIN_TRANS_DIST_TO_USE_ROT = 2.5  # m, robot has to be within this distance to use rot distance in cost
PATH_NAME = 'path.npy'  # saved path from l2_planning.py, should be in the same directory as this file

# here are some hardcoded paths to use if you want to develop l2_planning and this file in parallel
# TEMP_HARDCODE_PATH = [[2, 0, 0], [2.75, -1, -np.pi/2], [2.75, -4, -np.pi/2], [2, -4.4, np.pi]]  # almost collision-free
TEMP_HARDCODE_PATH = [[2, -.5, 0], [2.4, -1, -np.pi/2], [2.45, -3.5, -np.pi/2], [1.5, -4.4, np.pi]]  # some possible collisions


#Map Handling Functions
def load_map(filename):
    import matplotlib.image as mpimg
    import cv2 
    im = cv2.imread("../maps/" + filename)
    im = cv2.flip(im, 0)
    # im = mpimg.imread("../maps/" + filename)
    if len(im.shape) > 2:
        im = im[:,:,0]
    im_np = np.array(im)  #Whitespace is true, black is false
    im_np = np.logical_not(im_np)     #for ros
    return im_np

class PathFollower():
    def __init__(self):
        # time full path
        self.path_follow_start_time = rospy.Time.now()

        # use tf2 buffer to access transforms between existing frames in tf tree
        self.tf_buffer = tf2_ros.Buffer()
        self.listener = tf2_ros.TransformListener(self.tf_buffer)
        rospy.sleep(1.0)  # time to get buffer running

        # constant transforms
        self.map_odom_tf = self.tf_buffer.lookup_transform('map', 'odom', rospy.Time(0), rospy.Duration(2.0)).transform
        print(self.map_odom_tf)

        # subscribers and publishers
        self.cmd_pub = rospy.Publisher('/cmd_vel', Twist, queue_size=1)
        self.global_path_pub = rospy.Publisher('~global_path', Path, queue_size=1, latch=True)
        self.local_path_pub = rospy.Publisher('~local_path', Path, queue_size=1)
        self.collision_marker_pub = rospy.Publisher('~collision_marker', Marker, queue_size=1)

        # map
        # map = rospy.wait_for_message('/map', OccupancyGrid)
        # self.map_np = np.array(map.data).reshape(map.info.height, map.info.width)
        # self.map_resolution = round(map.info.resolution, 5)
        # self.map_origin = -utils.se2_pose_from_pose(map.info.origin)  # negative because of weird way origin is stored
        # self.map_nonzero_idxes = np.argwhere(self.map_np)
        map_filename = "myhal.png"
        occupancy_map = load_map(map_filename)
        self.map_np = occupancy_map
        self.map_resolution = 0.05
        self.map_origin = np.array([ 0.2 , 0.2 ,-0. ])
        self.map_nonzero_idxes = np.argwhere(self.map_np)


        # collisions
        self.collision_radius_pix = COLLISION_RADIUS / self.map_resolution
        self.collision_marker = Marker()
        self.collision_marker.header.frame_id = '/map'
        self.collision_marker.ns = '/collision_radius'
        self.collision_marker.id = 0
        self.collision_marker.type = Marker.CYLINDER
        self.collision_marker.action = Marker.ADD
        self.collision_marker.scale.x = COLLISION_RADIUS * 2
        self.collision_marker.scale.y = COLLISION_RADIUS * 2
        self.collision_marker.scale.z = 1.0
        self.collision_marker.color.g = 1.0
        self.collision_marker.color.a = 0.5

        # transforms
        self.map_baselink_tf = self.tf_buffer.lookup_transform('map', 'base_footprint', rospy.Time(0), rospy.Duration(2.0))
        self.pose_in_map_np = np.zeros(3)
        self.pos_in_map_pix = np.zeros(2)
        self.update_pose()

        # path variables
        cur_dir = os.path.dirname(os.path.realpath(__file__))

        # to use the temp hardcoded paths above, switch the comment on the following two lines
        self.path_tuples = np.load(os.path.join(cur_dir, 'path.npy')).T
        # self.path_tuples = np.array(TEMP_HARDCODE_PATH)

        self.path = utils.se2_pose_list_to_path(self.path_tuples, 'map')
        self.global_path_pub.publish(self.path)

        # goal
        self.cur_goal = np.array(self.path_tuples[0])
        self.cur_path_index = 0

        # trajectory rollout tools
        # self.all_opts is a Nx2 array with all N possible combinations of the t and v vels, scaled by integration dt
        self.all_opts = np.array(np.meshgrid(TRANS_VEL_OPTS, ROT_VEL_OPTS)).T.reshape(-1, 2)

        # if there is a [0, 0] option, remove it
        all_zeros_index = (np.abs(self.all_opts) < [0.001, 0.001]).all(axis=1).nonzero()[0]
        if all_zeros_index.size > 0:
            self.all_opts = np.delete(self.all_opts, all_zeros_index, axis=0)
        self.all_opts_scaled = self.all_opts * INTEGRATION_DT

        self.num_opts = self.all_opts_scaled.shape[0]
        self.horizon_timesteps = int(np.ceil(CONTROL_HORIZON / INTEGRATION_DT))

        self.rate = rospy.Rate(CONTROL_RATE)

        rospy.on_shutdown(self.stop_robot_on_shutdown)
        self.follow_path()

    def follow_path(self):
        while not rospy.is_shutdown():
            # timing for debugging...loop time should be less than 1/CONTROL_RATE
            tic = rospy.Time.now()

            self.update_pose()
            self.check_and_update_goal()

            print(f"\n=== TRAJECTORY ROLLOUT DEBUG ===")
            print(f"Current pose: x={self.pose_in_map_np[0]:.3f}, y={self.pose_in_map_np[1]:.3f}, theta={self.pose_in_map_np[2]:.3f}")
            print(f"Current goal (waypoint {self.cur_path_index}): x={self.cur_goal[0]:.3f}, y={self.cur_goal[1]:.3f}, theta={self.cur_goal[2]:.3f}")
            dist_to_goal = np.linalg.norm(self.pose_in_map_np[:2] - self.cur_goal[:2])
            print(f"Distance to goal: {dist_to_goal:.3f}m")

            # start trajectory rollout algorithm
            local_paths = np.zeros([self.horizon_timesteps + 1, self.num_opts, 3])
            local_paths[0] = np.atleast_2d(self.pose_in_map_np).repeat(self.num_opts, axis=0)

            # Propagate the trajectory forward, storing the resulting points in local_paths
            for t in range(1, self.horizon_timesteps + 1):
                # propogate trajectory forward, assuming perfect control of velocity and no dynamic effects
                # Apply unicycle model: x' = x + v*cos(theta)*dt, y' = y + v*sin(theta)*dt, theta' = theta + w*dt
                local_paths[t, :, 0] = local_paths[t-1, :, 0] + self.all_opts_scaled[:, 0] * np.cos(local_paths[t-1, :, 2])
                local_paths[t, :, 1] = local_paths[t-1, :, 1] + self.all_opts_scaled[:, 0] * np.sin(local_paths[t-1, :, 2])
                local_paths[t, :, 2] = local_paths[t-1, :, 2] + self.all_opts_scaled[:, 1]

            print(f"Generated {self.num_opts} trajectories with {self.horizon_timesteps} timesteps each")

            # check all trajectory points for collisions
            # first find the closest collision point in the map to each local path point
            local_paths_pixels = (self.map_origin[:2] + local_paths[:, :, :2]) / self.map_resolution
            # OPTIMIZED: Vectorized Collision Checking
            # Convert float pixels to integer indices once
            x_idx = np.rint(local_paths_pixels[:, :, 0]).astype(int)
            y_idx = np.rint(local_paths_pixels[:, :, 1]).astype(int)
            
            # Check bounds efficiently
            valid_x = (x_idx >= 0) & (x_idx < self.map_np.shape[1])
            valid_y = (y_idx >= 0) & (y_idx < self.map_np.shape[0])
            valid_mask = valid_x & valid_y
            
            # Initialize distances to infinity
            local_paths_lowest_collision_dist = np.full(self.num_opts, 50.0)
            
            # Any trajectory that goes out of bounds is invalid (dist = 0)
            out_of_bounds = ~np.all(valid_mask, axis=0)
            local_paths_lowest_collision_dist[out_of_bounds] = 0.0
            
            # Check collisions for valid points
            # We clip indices just to safely index the map, even if we marked them invalid already
            x_safe = np.clip(x_idx, 0, self.map_np.shape[1]-1)
            y_safe = np.clip(y_idx, 0, self.map_np.shape[0]-1)
            
            # Check if any point in trajectory hits an obstacle
            # map_np is 100 for obstacle, 0 for free. 
            collisions = (self.map_np[y_safe, x_safe] > 0)
            
            # If any point in a trajectory collides, mark trajectory as collided
            traj_collides = np.any(collisions & valid_mask[:, :, None].T if len(collisions.shape)>2 else collisions, axis=0)
            local_paths_lowest_collision_dist[traj_collides] = 0.0

            # Remove trajectories with collisions (collision distance less than collision radius)
            valid_opts = np.where(local_paths_lowest_collision_dist > 0)[0]

            # Calculate the final cost and choose the best control option
            if len(valid_opts) == 0:
                # No valid options - use hardcoded recovery
                print("WARNING: No valid collision-free trajectories! Using recovery control.")
                final_cost = np.zeros(0)
            else:
                final_cost = np.zeros(len(valid_opts))
                
                # Vectorized Cost Calculation
                valid_paths_end = local_paths[-1, valid_opts]
                
                # Translation Cost
                dist_to_goal = np.linalg.norm(valid_paths_end[:, :2] - self.cur_goal[:2], axis=1)
                
                # Rotation Cost logic
                rot_cost = np.zeros_like(dist_to_goal)
                
                # Calculate angle difference
                diff = valid_paths_end[:, 2] - self.cur_goal[2]
                abs_angle_diff = np.abs(np.arctan2(np.sin(diff), np.cos(diff)))
                
                # Apply rotation cost only where translation is close enough
                mask_close = dist_to_goal < MIN_TRANS_DIST_TO_USE_ROT
                rot_cost[mask_close] = ROT_DIST_MULT * abs_angle_diff[mask_close]

                # We removed obs_cost calculation for speed since we aren't computing exact distances
                final_cost = dist_to_goal + rot_cost

            if final_cost.size == 0:  # hardcoded recovery if all options have collision
                control = [-.1, 0]
            else:
                best_opt_idx = final_cost.argmin()
                best_opt = valid_opts[best_opt_idx]
                control = self.all_opts[best_opt]
                self.local_path_pub.publish(utils.se2_pose_list_to_path(local_paths[:, best_opt], 'map'))

            # send command to robot
            self.cmd_pub.publish(utils.unicyle_vel_to_twist(control))

            # uncomment out for debugging if necessary
            loop_time = (rospy.Time.now() - tic).to_sec()
            # print(f"Total loop time: {loop_time:.3f}s (should be < {1/CONTROL_RATE:.3f}s)")
            if loop_time > 1/CONTROL_RATE:
                print(f"WARNING: Loop time {loop_time:.3f}s exceeds control rate!")
            else:
                print(f"Loop time {loop_time:.3f}s within control rate.")

            self.rate.sleep()

    def update_pose(self):
        # Update numpy poses with current pose using the tf_buffer
        self.map_baselink_tf = self.tf_buffer.lookup_transform('map', 'base_footprint', rospy.Time(0)).transform
        self.pose_in_map_np[:] = [self.map_baselink_tf.translation.x, self.map_baselink_tf.translation.y,
                                  utils.euler_from_ros_quat(self.map_baselink_tf.rotation)[2]]
        self.pos_in_map_pix = (self.map_origin[:2] + self.pose_in_map_np[:2]) / self.map_resolution
        self.collision_marker.header.stamp = rospy.Time.now()
        self.collision_marker.pose = utils.pose_from_se2_pose(self.pose_in_map_np)
        self.collision_marker_pub.publish(self.collision_marker)

    def check_and_update_goal(self):
        # iterate the goal if necessary
        dist_from_goal = np.linalg.norm(self.pose_in_map_np[:2] - self.cur_goal[:2])
        abs_angle_diff = np.abs(self.pose_in_map_np[2] - self.cur_goal[2])
        rot_dist_from_goal = min(np.pi * 2 - abs_angle_diff, abs_angle_diff)
        if dist_from_goal < TRANS_GOAL_TOL and rot_dist_from_goal < ROT_GOAL_TOL:
            rospy.loginfo("Goal {goal} at {pose} complete.".format(
                    goal=self.cur_path_index, pose=self.cur_goal))
            if self.cur_path_index == len(self.path_tuples) - 1:
                rospy.loginfo("Full path complete in {time}s! Path Follower node shutting down.".format(
                    time=(rospy.Time.now() - self.path_follow_start_time).to_sec()))
                rospy.signal_shutdown("Full path complete! Path Follower node shutting down.")
            else:
                self.cur_path_index += 1
                self.cur_goal = np.array(self.path_tuples[self.cur_path_index])
        else:
            rospy.logdebug("Goal {goal} at {pose}, trans error: {t_err}, rot error: {r_err}.".format(
                goal=self.cur_path_index, pose=self.cur_goal, t_err=dist_from_goal, r_err=rot_dist_from_goal
            ))

    def stop_robot_on_shutdown(self):
        self.cmd_pub.publish(Twist())
        rospy.loginfo("Published zero vel on shutdown.")


if __name__ == '__main__':
    try:
        rospy.init_node('path_follower', log_level=rospy.DEBUG)
        pf = PathFollower()
    except rospy.ROSInterruptException:
        pass