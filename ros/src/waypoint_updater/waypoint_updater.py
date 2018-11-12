#!/usr/bin/env python
import numpy as np
import rospy
from geometry_msgs.msg import PoseStamped
from styx_msgs.msg import Lane, Waypoint
from scipy.spatial import KDTree
from std_msgs.msg import Int32

import math

'''
This node will publish waypoints from the car's current position to some `x` distance ahead.
'''

LOOKAHEAD_WPS = 50 # Number of waypoints we will publish. You can change this number
STOPAHEAD_WPS = 30
MAX_DECEL = 5

class WaypointUpdater(object):
    def __init__(self):
        rospy.init_node('waypoint_updater')

        self.base_lane = None
        self.pose = None
        #self.base_waypoints = None 
        self.stopline_wp_idx = -1
        self.waypoints_2d = None
        self.waypoint_tree = None
	self.getdata = False
	self.last_final_lane = None
	self.pre_closest_idx = None

        rospy.Subscriber('/current_pose', PoseStamped, self.pose_cb, queue_size=1)
        rospy.Subscriber('/base_waypoints', Lane, self.waypoints_cb, queue_size=1)
        rospy.Subscriber('/traffic_waypoint', Int32, self.traffic_cb)

        self.final_waypoints_pub = rospy.Publisher('final_waypoints', Lane, queue_size=1)

        self.loop()

    def loop(self):
        rate = rospy.Rate(50)
        while not rospy.is_shutdown():
            if self.pose and self.base_lane:
                self.publish_waypoints()
            rate.sleep()

    def get_closest_waypoint_idx(self):
        x = self.pose.pose.position.x
        y = self.pose.pose.position.y
        closest_idx = self.waypoint_tree.query([x, y], 1)[1]
        # Check if closest is ahead or behind vehicle
        closest_coord = self.waypoints_2d[closest_idx]
        prev_coord = self.waypoints_2d[closest_idx - 1]

        # Equation for hyperplane through closest_coords
        cl_vect = np.array(closest_coord)
        prev_vect = np.array(prev_coord)
        pos_vect = np.array([x, y])
        val = np.dot(cl_vect-prev_vect, pos_vect-cl_vect)
        if val > 0:
            closest_idx = (closest_idx + 1)  % len(self.waypoints_2d)
	"""
        dl_2d = lambda a, x, y: math.sqrt((a[0]-x)**2 + (a[1]-y)**2)
	print("length:",len(self.waypoints_2d))
	if not self.pre_closest_idx:
	    closest_idx = 0
	    temp_dis = 10000
	    for i, waypoint_2d in enumerate(self.waypoints_2d):
		if dl_2d(waypoint_2d, x, y) < temp_dis:
		    temp_dis = dl_2d(waypoint_2d, x, y);
		    closest_idx = i;
	else:
	    closest_idx = 0
	    temp_dis = 10000
	    for i in range(max(self.pre_closest_idx-20,0), min(self.pre_closest_idx+20, len(self.waypoints_2d))):
	    	if dl_2d(self.waypoints_2d[i], x, y) < temp_dis:
		    temp_dis = dl_2d(self.waypoints_2d[i], x, y);
		    closest_idx = i;
	    
	self.pre_closest_idx = closest_idx;
	"""
        return closest_idx

    """
    def publish_waypoints(self, closest_idx):
        lane = Lane()
        lane.header = self.base_waypoints.header
        lane.waypoints = self.base_waypoints.waypoints[closest_idx:closest_idx+LOOKAHEAD_WPS]
        self.final_waypoints_pub.publish(lane)
    """
    def publish_waypoints(self):

        # If get /traffic_waypoint message, then calulate lane points in current step
	if self.getdata:
	    final_lane = self.generate_lane()
	    self.last_final_lane = final_lane
            self.final_waypoints_pub.publish(final_lane)

	#If loss /traffic_waypoint message, then use last lane calculated with /traffic_waypoint message
	elif self.last_final_lane == None:
	    pass;
	else:
	    self.final_waypoints_pub.publish(self.last_final_lane)

    def generate_lane(self):
        lane = Lane()
        closest_idx = self.get_closest_waypoint_idx()
        farthest_idx = closest_idx + LOOKAHEAD_WPS
        base_waypoints = self.base_lane.waypoints[closest_idx:farthest_idx]
        if self.stopline_wp_idx == -1 or (self.stopline_wp_idx >= closest_idx + STOPAHEAD_WPS):
            lane.waypoints = base_waypoints
        else:
            # rospy.loginfo("decelerating")
            lane.waypoints = self.decelerate_waypoints(base_waypoints, closest_idx)
        return lane

    def decelerate_waypoints(self, waypoints, closest_idx):
        temp = []
        for i, wp in enumerate(waypoints):
            p = Waypoint()
            p.pose = wp.pose
            stop_idx = max(self.stopline_wp_idx - closest_idx - 1, 0) # one waypoints back from lines so car stops in front of line
            dist = self.distance(waypoints, i, stop_idx)
            vel = math.sqrt(2 * MAX_DECEL * dist)
            if vel < 1:
                vel = 0
            p.twist.twist.linear.x = min(vel, wp.twist.twist.linear.x)
	    print("stop_idx:",self.stopline_wp_idx,"vel:",vel,"closest_idx:",closest_idx)
            temp.append(p)
        return temp

    def pose_cb(self, msg):
        self.pose = msg

    def waypoints_cb(self, waypoints):
        #self.base_waypoints = waypoints
        self.base_lane = waypoints
        if not self.waypoints_2d:
            self.waypoints_2d = [[waypoint.pose.pose.position.x, waypoint.pose.pose.position.y] for waypoint in waypoints.waypoints]
            self.waypoint_tree = KDTree(self.waypoints_2d)
    def traffic_cb(self, msg):
        # Callback for /traffic_waypoint message.
        self.stopline_wp_idx = msg.data
	self.getdata = True;

    def get_waypoint_velocity(self, waypoint):
        return waypoint.twist.twist.linear.x

    def set_waypoint_velocity(self, waypoints, waypoint, velocity):
        waypoints[waypoint].twist.twist.linear.x = velocity

    def distance(self, waypoints, wp1, wp2):
        dist = 0
        dl = lambda a, b: math.sqrt((a.x-b.x)**2 + (a.y-b.y)**2  + (a.z-b.z)**2)
        for i in range(wp1, wp2+1):
	    try:
            	dist += dl(waypoints[wp1].pose.pose.position, waypoints[i].pose.pose.position)
            	wp1 = i
	    except:
		print(i,wp1,wp2)
        return dist


if __name__ == '__main__':
    try:
        WaypointUpdater()
    except rospy.ROSInterruptException:
        rospy.logerr('Could not start waypoint updater node.')
