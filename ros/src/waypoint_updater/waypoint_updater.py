#!/usr/bin/env python

import rospy
from geometry_msgs.msg import PoseStamped
from styx_msgs.msg import Lane, Waypoint
from std_msgs.msg import Int32
from scipy.spatial import KDTree
import numpy as np
from copy import deepcopy

import math

'''
This node will publish waypoints from the car's current position to some `x` distance ahead.

As mentioned in the doc, you should ideally first implement a version which does not care
about traffic lights or obstacles.

Once you have created dbw_node, you will update this node to use the status of traffic lights too.

Please note that our simulator also provides the exact location of traffic lights and their
current status in `/vehicle/traffic_lights` message. You can use this message to build this node
as well as to verify your TL classifier.

TODO (for Yousuf and Aaron): Stopline location for each traffic light.
'''

# Be careful with this number. 200 generates excesive lag in the system and it doesn't process
# current values OK. Reduced to 30 and everything works much better

LOOKAHEAD_WPS = 40 # Number of waypoints we will publish. You can change this number. It was 200


class WaypointUpdater(object):
    def __init__(self):
        rospy.init_node('waypoint_updater')

        rospy.Subscriber('/current_pose', PoseStamped, self.pose_cb)
        rospy.Subscriber('/base_waypoints', Lane, self.waypoints_cb)

        # TODO: Add a subscriber for /traffic_waypoint and /obstacle_waypoint below

        rospy.Subscriber('/traffic_waypoint', Int32, self.traffic_cb)

        self.final_waypoints_pub = rospy.Publisher('final_waypoints', Lane, queue_size=1)


        # TODO: Add other member variables you need below

        self.base_waypoints = None
        self.waypoints_2d = None
        self.waypoint_tree = None
        self.pose = None

        self.light_wp = -1
        self.old_state = 0
        self.loop()

    def loop(self):
        rate = rospy.Rate(30)  # Perhaps reduce to 30 hertz. Originally it is 50
        while not rospy.is_shutdown():
            if self.pose and self.base_waypoints:
                closest_waypoint_idx = self.get_closest_waypoint_idx()
                self.publish_waypoints(closest_waypoint_idx)
            rate.sleep()

    def get_closest_waypoint_idx(self):
        x = self.pose.pose.position.x
        y = self.pose.pose.position.y
        closest_idx = self.waypoint_tree.query([x,y], 1)[1]

        # could bee before of after

        closest_coord = self.waypoints_2d[closest_idx]
        prev_coord = self.waypoints_2d[closest_idx-1]

        # We compute the vectors from prev to cl and cl to car
        #  If they are in the same direction (dot > 0) then closest is rear car, need following point
        #  If not is after pose,

        cl_vect = np.array(closest_coord)
        prev_vect = np.array(prev_coord)
        car_vect = np.array([x, y])

        prev_to_cl_vect = cl_vect - prev_vect
        cl_to_car_vect = car_vect - cl_vect

        val = np.dot(prev_to_cl_vect, cl_to_car_vect)

        if val > 0:
            closest_idx = (closest_idx + 1) % len(self.waypoints_2d)

        return closest_idx

    def publish_waypoints(self, closest_idx):
        lane = Lane()
        lane.header = self.base_waypoints.header

        # Just make a deepcopy so altering speed due to a traffic light or other reason
        # does  ot modify the original waypoints desired speed

        lane.waypoints = deepcopy(self.base_waypoints.waypoints[closest_idx:closest_idx+LOOKAHEAD_WPS])

        npoints = 60   # Horizon to detect lights and braking distance.

        # Check if there is a light and if it is in front of us and not too far away
        # state is used to display the Breaking or Accelerating messages just when changing

        if self.light_wp != -1 and self.light_wp - closest_idx < npoints and self.light_wp - closest_idx >= 0:
            light_idx = self.light_wp - closest_idx

            for i in range(len(lane.waypoints)):  #reduce speed in 10 waypoints
                if i < light_idx :
                    if self.old_state == 0:
                        rospy.logwarn("Braking")
                        self.old_state = 1

                    vel = self.get_waypoint_velocity(lane.waypoints[i])

                    # Given actual speed to maintain  nice std acceleration we must do
                    # v = v0 + sqrt(1-x/d)
                    # where v0 is actual speed and x is distance from here and d i stop distance
                    #
                    # We modify multiplying by the speed of each waypoint to take into account
                    # possible maximum speeds.
                    #

                    

                    # If we are before the stoping line reduce speed linearly, else just stopped
                    if i < light_idx :
                        vel = vel * math.sqrt(1.0 - ((light_idx-i)*1.0)/(light_idx*1.0))
                    else:
                        vel = 0.0

                    self.set_waypoint_velocity(lane.waypoints, i, vel)

        elif self.old_state == 1:
            self.old_state = 0
            rospy.logwarn("Accelerating")

        # Just in case we have given all round we copy points from the begining

        if len(lane.waypoints) < LOOKAHEAD_WPS  or False :
            morepoints = LOOKAHEAD_WPS - len(lane.waypoints)
            lane.waypoints += self.base_waypoints.waypoints[0:morepoints]

        self.final_waypoints_pub.publish(lane)
        #rospy.logwarn("Publishing waypoints")


    def pose_cb(self, msg):
        self.pose = msg
        #rospy.logwarn("Position received")

    def waypoints_cb(self, waypoints):
        self.base_waypoints = waypoints

        #for i in range(len(waypoints.waypoints)):
        #    self.set_waypoint_velocity(self.base_waypoints.waypoints, i, 2.0)

        # Just to check we receive the data

        rospy.logwarn("Waypoints "+str(len(self.base_waypoints.waypoints)))

        if not self.waypoints_2d:
            self.waypoints_2d = [[waypoint.pose.pose.position.x, waypoint.pose.pose.position.y] for waypoint in waypoints.waypoints]
            self.waypoint_tree = KDTree(self.waypoints_2d)

    def traffic_cb(self, msg):
        # TODO: Callback for /traffic_waypoint message. Implement
        self.light_wp = msg.data


    def obstacle_cb(self, msg):
        # TODO: Callback for /obstacle_waypoint message. We will implement it later
        pass

    def get_waypoint_velocity(self, waypoint):
        return waypoint.twist.twist.linear.x

    def set_waypoint_velocity(self, waypoints, waypoint, velocity):
        waypoints[waypoint].twist.twist.linear.x = velocity

    def distance(self, waypoints, wp1, wp2):
        dist = 0
        dl = lambda a, b: math.sqrt((a.x-b.x)**2 + (a.y-b.y)**2  + (a.z-b.z)**2)
        for i in range(wp1, wp2+1):
            dist += dl(waypoints[wp1].pose.pose.position, waypoints[i].pose.pose.position)
            wp1 = i
        return dist


if __name__ == '__main__':
    try:
        WaypointUpdater()
    except rospy.ROSInterruptException:
        rospy.logerr('Could not start waypoint updater node.')
