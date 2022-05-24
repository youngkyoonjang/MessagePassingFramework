# MessagePassingFramework
Message Passing Framework for Vision Prediction Stability in HRI

# Requirements 
RGB-D camera is required.
Specifically we have tested using following devices:
- Cameras: Intel RealSense D435, D455 and L515. (Intel RealSense cameras require sufficient power to capture images. Please connect it to the main desktop or laptop ports instaed of a USB hub.)
- GPUs: NVIDIA GeForce RTX 3080, NVIDIA GeForce RTX 3070-8.0GB, Nvidia GeForce RTX 3060 and NVIDIA GeForce RTX 2080 Ti (slow)
- PyTorch version: 1.8.2+cu111
res
# Prerequisite (or note): 
1. Install Docker: https://docs.docker.com/engine/install/ubuntu/
2. If you cannot clone this repository using 'git clone', please download the entire repository. It happens because it is currently private.
3. You need to install pip or pip3 (e.g., if you use python3, 'sudo apt-get install python3-pip')
4. The code is designed to use 'python' command. If your OS only support python3, do 'Find & Replace' from 'python ' to 'python3 '.
5. If you see error message related to'RuntimeError: Not compiled with GPU support' when testing --hand, you need to change stable version docker command.

# Running Instruction: 
1. Download github repository:
```
git clone https://github.com/youngkyoonjang/MessagePassingFramework.git
```
2. Change parameters depending on what you want to run in '0_Do_necessary_settings_build_and_run.py':
* If you want to run pose estimation using the first GPU: 
```python
--pose T:0 --object F:0 --hand F:0 --gaze F:0
```
* If you want to run both object detection on the first GPU and hand-object state estimation on the second GPU:
```python
--pose F:0 --object T:0 --hand T:1 --gaze F:0
```
* Keep in mind that each model requires GPU memory to load their models, so you are only able to run modules depending on your GPU memory capacity.
3. Do everything else: 
```python
python 0_Do_necessary_settings_build_and_run.py
```
* Please be patient. Building a Docker image for the first time can take an hour or more (or less). However, it does not rebuild the prebuilt image from the second attempt. So it runs very fast from the second trial.

# Use cases: 
## Overview
* There are some subscriber examples under the folder MessagePassingFrameowrk/tas_perception/scripts/.
* The names are {tas_h_subscriber, tas_o_subscriber and tas_p_subscriber}.py for hand states, object detection and body pose topics, respectively.
* Each subscriber example visualise the output of each corresponding vision module: <tas_h_subscriber.py, tas_hand.py>, <tas_o_subscriber.py, tas_mmdetection.py>, and <tas_p_subscriber.py, tas_mmpose.py>.
* When all three modules are switched on to run and process, the integrater example tas_hpo_integrator.py will visualise all the outputs at once.
* To launch the subscriber or integrator example process, you need to put the following code line in the main.launch file under root/tas_perception/launch/ folder.
## In case you want to use a single PC or a laptop with one GPU
1. Check the IP address for the network (wifi or wired) you are currently connecting to:
```
ifconfig
```
* In my case, I am connecting to a wifi network. Then it shows:
```
wlp7s0: flags=4163<UP,BROADCAST,RUNNING,MULTICAST>  mtu 1500
        inet 10.0.0.71  netmask 255.255.255.0  broadcast 10.0.0.255
        inet6 fe80::55b9:dd5c:9925:874c  prefixlen 64  scopeid 0x20<link>
        ether 00:0c:43:02:19:7d  txqueuelen 1000  (Ethernet)
```
* Then, copy the ip: 10.0.0.71
2. Set up your ROS_MASTER_URI and ROS_IP using the ip address:
* Edit 'Makefile' in the root folder
```python
		-e ROS_MASTER_URI=http://10.0.0.71:11311 \
		-e ROS_IP=10.0.0.71 \
```
3. Turn on the module you want to test (e.g., object detection module running on the first (only) GPU):
* Edit '0_Do_necessary_settings_build_and_run.py' in the root folder
```python
os.system('python 2_Switch_module_activation_and_assign_gpus.py --pose F:0 --object T:0 --hand F:0 --gaze F:0') ##Acvitate:T/F, gpu_id:0/1
```
4. Make sure the realsense camera launch script is not commented out:
* If the realsense script in the 'main.launch' file under the <root>/tas_perception/launch folder is commented out:
```python
    <!-- <include file="$(find realsense2_camera)/launch/rs_camera.launch">
        <arg name="align_depth" value="true"/>
        <arg name="initial_reset" value="true"/>
    </include> -->
```
* Remove the comment:
```python
    <include file="$(find realsense2_camera)/launch/rs_camera.launch">
        <arg name="align_depth" value="true"/>
        <arg name="initial_reset" value="true"/>
    </include>
```
* Make sure you haven't touched any other scripts.

5. Make sure the corresponding subscriber launch script is not commented out:
* If the object usbscriber (specifically in this example) script in the 'main.launch' file under the <root>/tas_perception/launch folder is commented out:
```python
    <!-- <node pkg="tas_perception" type="tas_o_subscriber.py" name="tas_o_subscriber" /> -->
```
* Remove the comment:
```python
    <node pkg="tas_perception" type="tas_o_subscriber.py" name="tas_o_subscriber" />
```
6. Now ready to build docker image and run:
```python
python 0_Do_necessary_settings_build_and_run.py
```
## In case you want to use two PCs (1 GPU for each PC) on the same (WiFi) network:
1. Check the IP address for the network (wifi or wired) you are currently connecting to:
* Follow the same procedure written (above) in the case of using single PC.
* For the second PC as an example:
```
ifconfig
```
* In my case, I am connecting to a wifi network. Then it shows:
```
wlp108s0: flags=4163<UP,BROADCAST,RUNNING,MULTICAST>  mtu 1500
        inet 10.0.0.136  netmask 255.255.255.0  broadcast 10.0.0.255
        inet6 fe80::35ab:251a:6659:cdd  prefixlen 64  scopeid 0x20<link>
        ether a4:b1:c1:81:b6:9f  txqueuelen 1000  (Ethernet)
```
* Then, copy the ip: 10.0.0.136
2. (will be updating soon)
* For Master and second PCs, follow the same procedure written in the case of using single PC.
	
# References
```
@InProceedings{Jang2022MPF,  
author = {Jang, Youngkyoon and Demiris, Yiannis},  
title = {Message Passing Framework for Vision Prediction Stability in Human Robot Interaction},  
booktitle = {IEEE International Conference on Robotics and Automation},  
year = {2022}  
}  
```
