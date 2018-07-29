This is the project repository for the final project of the Udacity Self-Driving Car Nanodegree: Programming a Real Self-Driving Car.

For more information about the [project](https://github.com/udacity/CarND-Capstone/), see the original project introduction [here](https://classroom.udacity.com/nanodegrees/nd013/parts/6047fe34-d93c-4f50-8336-b70ef10cb4b2/modules/e1a23b06-329a-4684-a717-ad476f0d8dff/lessons/462c933d-9f24-42d3-8bdc-a08a5fc866e4/concepts/5ab4b122-83e6-436d-850f-9f4d26627fd9).

# Capstone Project - System Integration

## Team: Speedy

  - [Francisco Gorina](https://github.com/fgorina) (lead) : fgorina@gmail.com
  - [Ricardo Rios](https://github.com/ricardoues) : ricardo.rios.sv@gmail.com
  - [Luciano Silveira](https://github.com/ladrians) : ladrians@gmail.com
  - Frank Wong : wfhit@139.com
  - Omar Barrera : oabj1986@gmail.com

## Installation

For installation check the original project readme file [here](https://github.com/udacity/CarND-Capstone/).
  
## Project Overview

The System Architecture is detailed as follows:

![Architecture](./imgs/final-project-ros-graph-v2.png)

All code is located in [src](./ros/src). The main components are:

  - Perception
  - Planning
  - Control
  - Car\Simulation

### Perception

The Perception module can be divided in the following components:

  - Obstacle Detection
  - Traffic Light Detection

More detail [here](perception.md).

### Planning

The Planning module includes the following nodes:

  - Waypoint loader
  - Waypoint updater

More detail [here](planning.md).

### Control

The Control module is in charge of moving the vehicle:

  - DBW Node
  - Waypoint follower
  
More detail [here](control.md).

### Testing

#### Simulation
1. Run the simulator and select a slow resolution screen (to increase performance) for example 400x300.
2. Select the 'Highway' track.
3. Enable the 'Camera' option.
4. Start your pipeline, go to the ros folder from your project, source it and run the correct launch file.
```bash
source devel/setup.sh
roslaunch launch/styx.launch
```
5. When ready uncheck the 'Manual' option from the simulator to get started.
6. For debugging purposes use the 'image_debug' topic.
```bash
rqt_image_view
```

#### Real world
1. Download [training bag](https://s3-us-west-1.amazonaws.com/udacity-selfdrivingcar/traffic_light_bag_file.zip) that was recorded on the Udacity self-driving car.
2. Unzip the file
```bash
unzip traffic_light_bag_file.zip
```
3. Start ROS
```bash
roscore
```
4. Play the bag file
```bash
rosbag play -l traffic_light_bag_file/traffic_light_training.bag
```
5. Launch your project in site mode
```bash
cd CarND-Capstone/ros
roslaunch launch/site.launch
```
6. Confirm that traffic light detection works on real life images
