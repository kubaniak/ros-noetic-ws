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

        # Robot position in map pixel coordinates
        x_start = int(odom_map[0] / CELL_SIZE)
        y_start = int(odom_map[1] / CELL_SIZE)

        # Loop through each LiDAR beam (with downsampling)
        for i in range(0, len(scan_msg.ranges), SCAN_DOWNSAMPLE):
            range_mes = scan_msg.ranges[i]
            # Skip invalid readings
            if range_mes == 0.0 or np.isnan(range_mes):
                continue
            # Cap at max range so inf rays still mark free space up to range_max
            if np.isinf(range_mes):
                range_mes = scan_msg.range_max

            # Beam angle in map frame: beam index angle + robot heading in map
            angle = scan_msg.angle_min + i * scan_msg.angle_increment + odom_map[2]

            self.np_map, self.log_odds = self.ray_trace_update(
                self.np_map, self.log_odds, x_start, y_start, angle, range_mes
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
        # Compute endpoint pixel of the ray
        x_end = int(x_start + (range_mes / CELL_SIZE) * np.cos(angle))
        y_end = int(y_start + (range_mes / CELL_SIZE) * np.sin(angle))

        # Get all pixel indices along the ray (skimage line)
        rr, cc = ray_trace(x_start, y_start, x_end, y_end)

        # Clip to map bounds
        in_bounds = (rr >= 0) & (rr < map.shape[0]) & (cc >= 0) & (cc < map.shape[1])
        rr = rr[in_bounds]
        cc = cc[in_bounds]

        if len(rr) == 0:
            return map, log_odds

        # All pixels except the last NUM_PTS_OBSTACLE are free
        free_rr = rr[:-NUM_PTS_OBSTACLE] if len(rr) > NUM_PTS_OBSTACLE else rr
        free_cc = cc[:-NUM_PTS_OBSTACLE] if len(cc) > NUM_PTS_OBSTACLE else cc
        log_odds[free_rr, free_cc] -= BETA

        # Last NUM_PTS_OBSTACLE pixels are occupied (only if ray hit something, not at max range)
        if not np.isinf(scan_msg.range_max) and range_mes < scan_msg.range_max:
            occ_rr = rr[-NUM_PTS_OBSTACLE:]
            occ_cc = cc[-NUM_PTS_OBSTACLE:]
            log_odds[occ_rr, occ_cc] += ALPHA

        # Convert log odds to probability, scale to 0-100 int8
        map[rr, cc] = (self.log_odds_to_probability(log_odds[rr, cc]) * 100).astype(np.int8)

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