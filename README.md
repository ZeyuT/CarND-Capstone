
# System Integration Project

## Introduction
This is the project repo for the final project of the Udacity Self-Driving Car Nanodegree: Programming a Real Self-Driving Car. In the project, I built ROS nodes to implement core functionality of the autonomous vehicle system, including traffic light detection, control, and waypoint following. This software system will be tested on Carla (Udacity’s Self Driving Lincoln MKZ) around a test track.

Please use **one** of the two installation options, either native **or** docker installation.

## Implementation

[//]: # (Image References)

[image1]: ./images/final-project-ros-graph.png
[image2]: ./images/example_simulator.jpg
[image3]: ./images/example_site.jpg
[image4]: ./examples/left_flip.jpg
[image5]: ./examples/MSE.png
[image6]: ./examples/ModelStructure.JPG 

The following is a system architecture diagram showing the ROS nodes and topics used in the project.
![SystemArchitecture][image1]
Notes that obstacle detection is not included in the project.

### Perception Subsystem

The subsystem detects external environments through varies of sensors. and publishes information to other subsystems. In the project, an onboard camera is used for detecting traffic light.

#### Traffic Light Detection Node

The node subscribes four types of topics to get messages of camera images, the entire list of waypoints, positions of all traffic lights ahead and their corresponding stop lines, and current vehicle position. Then it executes traffic light classification and publishes the upcoming traffic light status and the stop line's location. To avoid measurement noise, new traffic light status can not be published unless detections over the previous 4 successive steps are the same as the new traffic light status. 

The light classification takes place in traffic light classification node(./ros/src/tl_detector/light_classification/tf_classifer.py). It is seperated into two steps: region detection and color recognition.

For region detection step, I use [TensorFlow Object Detection API](https://github.com/tensorflow/models/tree/master/research/object_detection), which is an open source framework built on top of TensorFlow to construct, train and deploy object detection models. There are lots of detection models given by Object Detection API and they are already pre-trained on [COCO dataset](http://mscoco.org/). For the project, because of the high demand on processing speed, I use 
ssd_mobilenet_v1_coco model, which is the lightest and fastest model in all given models and is accurate enough for detection in the project. The model could classify 90 classes of objects and output regions of these objects. The class number of 'traffic light' is 10, which is the only class that is used in the project. Here are examples of region detection:
![example_simulator][image2]
![example_site][image3]

After region detection, I apply color recognition to the region with max score. There are some thresholds for region size and minimum max score to filter out those False Positive regions.

For color recognition, I simply count pixels of red and green in detected regions and consider each region to be a specific color if the number of pixels with the color is larger than COLOR_THRESHOLD. The recognition method is pretty fast and accurate enough for the project. After several trials, I found that the red color could be reliably recognized in HSV colorspace while green color could not be simply recognized through thresholding pixel values in all colorspaces I tried. Therefore, I set traffic light status to be two types: red and unknown(same as green).

### Planning Subsystem

The subsystem plans the vehicle’s trajectory based on the vehicle’s current position and velocity along with the state of upcoming traffic lights. The subsystem publishes the trajectory to the control subsystem in the form of a list of waypoints.

#### Waypoint Loader Node

This node is given by Udacity. It is used for loading a CSV file that contains all waypoints along the track and publishes them. The loaded CSV file can be switched between the simulator and the real testing track.

#### Waypoint Updater Node

This node subscribes to three topics to get the entire list of waypoints, vehicle’s current position and velocity, and the state of upcoming traffic lights and stop lines' position. The node plans a path and publishes a list of 50 waypoints to the control subsystem to follow at a rate of 50Hz.

Each time the node calculates the trajectory, if the upcoming traffic light is red and the vehicle gets close enough to the traffic light, then the node will plan a trajectory on which the vehicle will decelerate at the maximum rate and stop right behind the stop line of the upcoming traffic light. Otherwise, the node will plan trajectories only based on the waypoints list of the track.

### Control Subsystem 

The subsystem publishes commands for the vehicle’s steering, throttle, and brakes based on the list of waypoints published by the planning subsystem. Besides, the subsystem can be taken over by humans at any time.

#### Waypoint Follower Node

The node is given by Udacity.It parses the list of waypoints to follow and publishes target linear and angular velocities to the /twist_cmd topic.

#### DBW (Drive-by-Wire) Node

The node subscribes three topics to get target linear and angular velocities, current vehicle velocity and current control mode(manually or autonomously). The node publishes steering, throttle, and brakes command at a rate of 50Hz and therefore has three controllers.

_Steering controller_

The controller is given by Udacity. It can be used to convert target linear and angular velocity to steering commands, considering the vehicle’s steering ratio and wheelbase length.

_Throttle controller_

The controller applies PID control that considers the target linear velocity as the reference and adjusts the throttle.

_Braking controller_

The controller is a simple logic controller. It calculates brake torque (in Nm) based on the deceleration, the vehicle's mass and the wheel radius. The maximum brake torque is 700 Nm, which is the demand to fully stop Carla (Udacity’s Self Driving Lincoln MKZ).

_Low-pass filter_

The current velocity measured from sensors of the simulator is passed through a low-pass filter before being used in controllers above. The low-pass filter can filter out those abnormal velocities and reduce possible jitter from noise in velocity data.

## Environment

### Native Installation

* Be sure that your workstation is running Ubuntu 16.04 Xenial Xerus or Ubuntu 14.04 Trusty Tahir. [Ubuntu downloads can be found here](https://www.ubuntu.com/download/desktop).
* If using a Virtual Machine to install Ubuntu, use the following configuration as minimum:
  * 2 CPU
  * 2 GB system memory
  * 25 GB of free hard drive space

  The Udacity provided virtual machine has ROS and Dataspeed DBW already installed, so you can skip the next two steps if you are using this.

* Follow these instructions to install ROS
  * [ROS Kinetic](http://wiki.ros.org/kinetic/Installation/Ubuntu) if you have Ubuntu 16.04.
  * [ROS Indigo](http://wiki.ros.org/indigo/Installation/Ubuntu) if you have Ubuntu 14.04.
* [Dataspeed DBW](https://bitbucket.org/DataspeedInc/dbw_mkz_ros)
  * Use this option to install the SDK on a workstation that already has ROS installed: [One Line SDK Install (binary)](https://bitbucket.org/DataspeedInc/dbw_mkz_ros/src/81e63fcc335d7b64139d7482017d6a97b405e250/ROS_SETUP.md?fileviewer=file-view-default)
* Download the [Udacity Simulator](https://github.com/udacity/CarND-Capstone/releases).

### Docker Installation
[Install Docker](https://docs.docker.com/engine/installation/)

Build the docker container
```bash
docker build . -t capstone
```

Run the docker file
```bash
docker run -p 4567:4567 -v $PWD:/capstone -v /tmp/log:/root/.ros/ --rm -it capstone
```

### Port Forwarding
To set up port forwarding, please refer to the [instructions from term 2](https://classroom.udacity.com/nanodegrees/nd013/parts/40f38239-66b6-46ec-ae68-03afd8a601c8/modules/0949fca6-b379-42af-a919-ee50aa304e6a/lessons/f758c44c-5e40-4e01-93b5-1a82aa4e044f/concepts/16cf4a78-4fc7-49e1-8621-3450ca938b77)

### Usage

1. Clone the project repository
```bash
git clone https://github.com/udacity/CarND-Capstone.git
```

2. Install python dependencies
```bash
cd CarND-Capstone
pip install -r requirements.txt
```
3. Make and run styx
```bash
cd ros
catkin_make
source devel/setup.sh
roslaunch launch/styx.launch
```
4. Run the simulator

### Real world testing
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
