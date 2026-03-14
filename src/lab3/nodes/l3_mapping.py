#!/usr/bin/env python3
from __future__ import division, print_function

import numpy as np
import rospy
import tf2_ros
from skimage.draw import line as ray_trace
import rospkg
import matplotlib.pyplot as plt

# msgs
from nav_msgs.msg import OccupancyGrid, MapMetaData
from geometry_msgs.msg import TransformStamped
from sensor_msgs.msg import LaserScan

from utils import convert_pose_to_tf, convert_tf_to_pose, euler_from_ros_quat, \
     tf_to_tf_mat, tf_mat_to_tf


ALPHA = 1
BETA = 1
MAP_DIM = (4, 4)
CELL_SIZE = .01
NUM_PTS_OBSTACLE = 3
SCAN_DOWNSAMPLE = 1

class OccupancyGripMap:
    def __init__(self):
        # use tf2 buffer to access transforms between existing frames in tf tree
        self.tf_buffer = tf2_ros.Buffer()
        self.listener = tf2_ros.TransformListener(self.tf_buffer)
        self.tf_br = tf2_ros.TransformBroadcaster()

        # subscribers and publishers
        self.scan_sub = rospy.Subscriber('/scan', LaserScan, self.scan_cb, queue_size=1)
        self.map_pub = rospy.Publisher('/map', OccupancyGrid, queue_size=1)

        # attributes
        width = int(MAP_DIM[0] / CELL_SIZE); height = int(MAP_DIM[1] / CELL_SIZE)
        self.log_odds = np.zeros((width, height))
        self.np_map = np.ones((width, height), dtype=np.uint8) * -1  # -1 for unknown
        self.map_msg = OccupancyGrid()
        self.map_msg.info = MapMetaData()
        self.map_msg.info.resolution = CELL_SIZE
        self.map_msg.info.width = width
        self.map_msg.info.height = height

        # transforms
        self.base_link_scan_tf = self.tf_buffer.lookup_transform('base_link', 'base_scan', rospy.Time(0),
                                                            rospy.Duration(2.0))
        odom_tf = self.tf_buffer.lookup_transform('odom', 'base_link', rospy.Time(0), rospy.Duration(2.0)).transform

        # set origin to center of map
        rob_to_mid_origin_tf_mat = np.eye(4)
        rob_to_mid_origin_tf_mat[0, 3] = -width / 2 * CELL_SIZE
        rob_to_mid_origin_tf_mat[1, 3] = -height / 2 * CELL_SIZE
        odom_tf_mat = tf_to_tf_mat(odom_tf)
        self.map_msg.info.origin = convert_tf_to_pose(tf_mat_to_tf(odom_tf_mat.dot(rob_to_mid_origin_tf_mat)))

        # map to odom broadcaster
        self.map_odom_timer = rospy.Timer(rospy.Duration(0.1), self.broadcast_map_odom)
        self.map_odom_tf = TransformStamped()
        self.map_odom_tf.header.frame_id = 'map'
        self.map_odom_tf.child_frame_id = 'odom'
        self.map_odom_tf.transform.rotation.w = 1.0

        rospy.spin()
        plt.imshow(100-self.np_map, cmap='gray', vmin=0, vmax=100)
        rospack = rospkg.RosPack()
        path = rospack.get_path("rob521_lab3")
        plt.savefig(path+"/map.png")

    def broadcast_map_odom(self, e):
        self.map_odom_tf.header.stamp = rospy.Time.now()
        self.tf_br.sendTransform(self.map_odom_tf)

    def scan_cb(self, scan_msg):
        # read new laser data and populate map
        # get current odometry robot pose
        try:
            odom_tf = self.tf_buffer.lookup_transform('odom', 'base_scan', rospy.Time(0)).transform
        except tf2_ros.TransformException:
            rospy.logwarn('Pose from odom lookup failed. Using origin as odom.')
            odom_tf = convert_pose_to_tf(self.map_msg.info.origin)

        # get odom in frame of map
        odom_map_tf = tf_mat_to_tf(
            np.linalg.inv(tf_to_tf_mat(convert_pose_to_tf(self.map_msg.info.origin))).dot(tf_to_tf_mat(odom_tf))
        )
        odom_map = np.zeros(3)
        odom_map[0] = odom_map_tf.translation.x
        odom_map[1] = odom_map_tf.translation.y
        odom_map[2] = euler_from_ros_quat(odom_map_tf.rotation)[2]

        # YOUR CODE HERE!!! Loop through each measurement in scan_msg to get the correct angle and
        # x_start and y_start to send to your ray_trace_update function.

        x_start = int((odom_map[0] - self.map_msg.info.origin.position.x) / CELL_SIZE)
        y_start = int((odom_map[1] - self.map_msg.info.origin.position.y) / CELL_SIZE)

        for i in range(0, len(scan_msg.ranges), SCAN_DOWNSAMPLE):
            r = scan_msg.ranges[i]

            if np.isinf(r) or np.isnan(r) or r > scan_msg.range_max:
                r = scan_msg.range_max

            if r < scan_msg.range_min:
                continue

            # The beam's angle in the base_scan frame
            angle_in_scan = scan_msg.angle_min + i * scan_msg.angle_increment

            # Find the pose of the laser frame in the odom frame
            # The orientation of the sensor in base_link
            scan_euler = euler_from_ros_quat(self.base_link_scan_tf.transform.rotation)

            global_angle = odom_map[2] + scan_euler[2] + angle_in_scan

            self.np_map, self.log_odds = self.ray_trace_update(
                self.np_map, self.log_odds, x_start, y_start, global_angle, r
            )

        # publish the message
        self.map_msg.info.map_load_time = rospy.Time.now()
        self.map_msg.data = self.np_map.flatten()
        self.map_pub.publish(self.map_msg)

    def ray_trace_update(self, map, log_odds, x_start, y_start, angle, range_mes):
        """
        A ray tracing grid update as described in the lab document.

        :param map: The numpy map.
        :param log_odds: The map of log odds values.
        :param x_start: The x starting point in the map coordinate frame (i.e. the x 'pixel' that the robot is in).
        :param y_start: The y starting point in the map coordinate frame (i.e. the y 'pixel' that the robot is in).
        :param angle: The ray angle relative to the x axis of the map.
        :param range_mes: The range of the measurement along the ray.
        :return: The numpy map and the log odds updated along a single ray.
        """
        # YOUR CODE HERE!!! You should modify the log_odds object and the numpy map based on the outputs from
        # ray_trace and the equations from class. Your numpy map must be an array of int8s with 0 to 100 representing
        # probability of occupancy, and -1 representing unknown.

        x_end = int(x_start + range_mes * np.cos(angle) / CELL_SIZE)
        y_end = int(y_start + range_mes * np.sin(angle) / CELL_SIZE)

        rr, cc = ray_trace(x_start, y_start, x_end, y_end)

        # Filter points outside map bounds
        valid = (rr >= 0) & (rr < log_odds.shape[0]) & (cc >= 0) & (cc < log_odds.shape[1])
        rr = rr[valid]
        cc = cc[valid]

        # In simulation, max range is 3.5. We passed range_max for inf.
        # If the ray hit an obstacle, range_mes < 3.5. So the last NUM_PTS_OBSTACLE are an obstacle.
        # However, to avoid hardcoding 3.5, we just use the fact that ray_trace_update doesn't know the max_range.
        # We can compute if the distance is exactly what we set for max range, but since we modify log_odds based on NUM_PTS_OBSTACLE:

        # The prompt says: NUM_PTS_OBSTACLE indicating how many pixels at the end of a LiDAR beam to consider part of the obstacle

        # The common approach for these labs is to always update the free space for all points except the last NUM_PTS_OBSTACLE.
        # Wait, if range_mes == range_max (no obstacle), we shouldn't add to the obstacle log odds.
        # But `ray_trace_update` signature `def ray_trace_update(self, map, log_odds, x_start, y_start, angle, range_mes):` doesn't pass max range.
        # Let's just update BETA for free space, and ALPHA for the end points if it's not out of bounds.

        # Decrease log odds for all points (BETA is free space multiplier)
        log_odds[rr, cc] -= BETA

        # We only add ALPHA if it wasn't a max range measurement. We check if range_mes < 3.49
        # In `scan_msg` range_max is 3.5, but some small float inaccuracies exist.
        if range_mes < 3.49 and len(rr) > 0:
            obs_rr = rr[-NUM_PTS_OBSTACLE:]
            obs_cc = cc[-NUM_PTS_OBSTACLE:]

            # We already decreased BETA for these, so we add ALPHA + BETA to offset it.
            log_odds[obs_rr, obs_cc] += (ALPHA + BETA)

        # Update the probabilities in the numpy map for the affected cells
        # The instructions specify np_map must be an array of int8s with 0 to 100 representing
        # probability of occupancy, and -1 representing unknown.

        # In init, self.np_map = np.ones((width, height), dtype=np.uint8) * -1
        # Since it uses uint8, -1 becomes 255. Let's just store int(prob * 100).

        # The flattening in scan_cb is map_msg.data = self.np_map.flatten()
        # Ensure we're indexing [rr, cc] and not transposed.
        # Since map is created as (width, height), rr (x) goes into dimension 0, cc (y) goes into dimension 1.
        # This matches log_odds[rr, cc] since ray_trace(x_start, y_start, x_end, y_end) returns rr=x, cc=y

        probs = self.log_odds_to_probability(log_odds[rr, cc])
        map[rr, cc] = (probs * 100).astype(np.uint8)

        return map, log_odds

    def log_odds_to_probability(self, values):
        # print(values)
        return np.exp(values) / (1 + np.exp(values))


if __name__ == '__main__':
    try:
        rospy.init_node('mapping')
        ogm = OccupancyGripMap()
    except rospy.ROSInterruptException:
        pass