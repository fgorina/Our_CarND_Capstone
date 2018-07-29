# Planning Module

The planning subsystem is in charge of selecting the vehicle's path based on current position, velocity taking into account the state of upcoming traffic lights.  Main nodes are:

  - Waypoint loader
  - Waypoint updater

## Waypoint loader

In charge of loading a csv file with all waypoints the vehicle should go through (topic /base_waypoints).

## Waypoint updater

![dbw](./imgs/waypoint-updater-ros-graph.png)

Responsible for planning the next waypoints the vehicle should follow. It loads base waypoints, considers the vehicle current state, traffic lights and publishes to the topic /final_waypoints the next N number of waypoints the vehicle should go through. This list includes the (x,y) coordinate and the target linear velocity which is necessary for keeping the car within the correct lane and maintain the desired velocity and stop for red lights.

On every timestep a new path is planned. First, get the closest waypoint to the vehicle's current position and build a list containing the next 200 waypoints. Then, check the upcoming traffic lights. Adjust the speed of the waypoints immediately before the traffic light to slow the vehicle to a halt at the red light or continue on green lights.