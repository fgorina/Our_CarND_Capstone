This is the project repository for the final project of the Udacity Self-Driving Car Nanodegree: Programming a Real Self-Driving Car.

For more information about the project, see the original project introduction [here](https://classroom.udacity.com/nanodegrees/nd013/parts/6047fe34-d93c-4f50-8336-b70ef10cb4b2/modules/e1a23b06-329a-4684-a717-ad476f0d8dff/lessons/462c933d-9f24-42d3-8bdc-a08a5fc866e4/concepts/5ab4b122-83e6-436d-850f-9f4d26627fd9).

# Captsonte Project - System Integration

## Team: Speedy

  - Francisco Gorina (lead) : fgorina@gmail.com
  - Ricardo Rios : ricardo.rios.sv@gmail.com
  - Luciano Silveira : ladrians@gmail.com

## Installation

For installation check the original project readme file [here](https://github.com/udacity/CarND-Capstone/).
  
## Project Overview

The System Architecture is detailed as follows:

![](./imgs/final-project-ros-graph-v2.png)

All code is located in [./ros/src](). The main components are:

  - Perception
  - Planning
  - Control
  - Car\Simulation

### Perception

![](./imgs/tl-detector-ros-graph.png)

The Perception module can be divided in the following components:

  - Obstacle Detection
  - Traffic Light Detection

TODO, improve here

### Planning

The Planning module includes the following nodes:

  - Waypoint loader
  - Waypoint updater

TODO, improve here

### Control

The Control module is in charge of moving the vehicle:

  - DBW Node
  - Waypoint follower

TODO, improve here

### Car or Simulation

TODO improve here

#### Real world testing
1. Download [training bag](https://s3-us-west-1.amazonaws.com/udacity-selfdrivingcar/traffic_light_bag_file.zip) that was recorded on the Udacity self-driving car.
2. Unzip the file
```bash
unzip traffic_light_bag_file.zip
```
3. Play the bag file
```bash
rosbag play -l traffic_light_bag_file/traffic_light_training.bag
```
4. Launch your project in site mode
```bash
cd CarND-Capstone/ros
roslaunch launch/site.launch
```
5. Confirm that traffic light detection works on real life images
