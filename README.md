# MessagePassingFramework
This is the code to accompany the paper “Message Passing Framework for Vision Prediction Stability in Human Robot Interaction, ICRA 2022”; if you find this code useful, we would appreciate it if you cite this paper.

# Requirements 
RGB-D camera is required.
Specifically we have tested using following devices:
- Cameras: Intel RealSense D435, D455 and L515. (Intel RealSense cameras require sufficient power to capture images. Please connect it to the main desktop or laptop ports instaed of a USB hub.)
- GPUs: NVIDIA GeForce RTX 3080, NVIDIA GeForce RTX 3070-8.0GB, Nvidia GeForce RTX 3060 and NVIDIA GeForce RTX 2080 Ti (slow)
- PyTorch version: 1.8.2+cu111
res
# Prerequisite (or note): 
1. [Install **Docker**](https://docs.docker.com/engine/install/ubuntu/)
2. [Install **NVIDIA-Docker**](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html)
3. If you cannot clone this repository using 'git clone', please download the entire repository. It happens because it is currently private.
4. You need to install pip or pip3 (e.g., if you use python3, 'sudo apt-get install python3-pip')
5. The code is designed to use 'python' command. If your OS only support python3, do **use 'python3' instead of 'python' in the command line**. 
6. If you see error message related to'RuntimeError: Not compiled with GPU support' when testing --hand, you need to change stable version docker command.
* If you have an error related to tas_hand.py, it might be the case of downloading model failure. In this case, please download the 'faster_rcnn_1_8_89999.pth' file directly from the [100DOH](https://github.com/ddshan/hand_object_detector) github repository. Then, put it in the <root>/tas_perception/hand_object/models/ folder.

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
* Because we need a person bounding box (which is an object detection result) as an input for pose estimation, the object module turns on automatically when the pose estimation module is activated.
* Keep in mind that each model requires GPU memory to load their models, so you are only able to run modules depending on your GPU memory capacity.
3. Switch on/off example subscribers that you want to visualise:
* If you want to visualise pose estimation topic subscribing results: 
```python
--pose T --object F --hand F --integration F --MPF F
```
4. Switch on/off the realsense camera launch script:
* If you want to launch realsense camera on your machine: 
```python
--realsense T
```
5. Do everything else: 
```python
python( or python3) 0_Do_necessary_settings_build_and_run.py
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
os.system(my_python_path + ' 2_Switch_module_activation_and_assign_gpus.py --pose F:0 --object T:0 --hand F:0 --gaze F:0') ##Acvitate:T/F, gpu_id:0/1
```
4. Switch on the module you want to visualise subscribing results (e.g., object detection subscriber):
* Edit '0_Do_necessary_settings_build_and_run.py' in the root folder
```python
os.system(my_python_path + 3_Switch_subscriber_activation.py --pose F --object T --hand F --integration F --MPF F') ##Acvitate:T/F
```
5. Switch on the realsense camera launch script:
* Edit '0_Do_necessary_settings_build_and_run.py' in the root folder
```python
os.system(my_python_path + 4_Switch_on_off_realsense_launch.py --realsense T') ##Acvitate:T/F
```
6. Now ready to build docker image and run:
```python
python( or python3) 0_Do_necessary_settings_build_and_run.py
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
2. Set up your ROS_MASTER_URI and ROS_IP using the ip address:
* For Master PC, follow the same procedure written (above) in the case of using single PC.
* For second PC, edit 'Makefile' in the root folder
```python
		-e ROS_MASTER_URI=http://10.0.0.71:11311 \
		-e ROS_IP=10.0.0.136 \
```
3. Turn on the module you want to test (For second PC, for example, pose estimation module running on the first (only) GPU):
* Edit '0_Do_necessary_settings_build_and_run.py' in the root folder
```python
os.system(my_python_path + ' 2_Switch_module_activation_and_assign_gpus.py --pose T:0 --object F:0 --hand F:0 --gaze F:0') ##Acvitate:T/F, gpu_id:0/1
```
* The pose estimation and object detection modules must run on the same machine.

4. Switch on the module you want to visualise subscribing results (e.g., pose estimation subscriber):
* Edit '0_Do_necessary_settings_build_and_run.py' in the root folder
```python
os.system(my_python_path + ' 3_Switch_subscriber_activation.py --pose T --object F --hand F --integration F --MPF F') ##Acvitate:T/F
```
* You can subscribe for the topics published by other PCs on the same network.
	
5. Switch off the realsense camera launch script:
* Edit '0_Do_necessary_settings_build_and_run.py' in the root folder
```python
os.system(my_python_path + 4_Switch_on_off_realsense_launch.py --realsense F') ##Acvitate:T/F
```
	
6. Now ready to build docker image and run:
```python
python( or python3) 0_Do_necessary_settings_build_and_run.py
```
* Make sure the master PC is running before you run the second PC
# To-do list
- [ ] Resolve 'hand-object detector' model download  issue: it currently often fails to download the model from Google Drive automatically. I may need to put it somewhere else and then put other script download file automatically. At the moment, you can download directly from the origianl 100DOH repository.
- [x] Resolve 'hand-object detector's 'nms_thresh'-related issue. The error was 'RuntimeError: Not compiled with GPU support'. This was resolved by putting lines to make sure nvidia-docker is installed before building docker.
- [ ] Put explicit example case, how to turn of MPF. Simply, when switching on pose, object, hand modules (with proper splitting modules into two gpus) and switching on mpf subscriber only, it is able to run MPF on the subscriber script. <- it all depends on your choice of deciding which modules run on which GPUs and which PC will run the mpf subscriber.
- [ ] Update references when this repository uses more recent vision modules such as RT-GENE, which is not mentioned in the Message Passing Frameowrk (ICRA22) paper. This repository will be updated by adopting the state-of-the-art modules. So, the methods (or pretrained models) used in this repository may be different from what MPF paper describes. To reflect this, I will update when there are other modules being used after ICRA 22 presentation.
	
# References
```
@InProceedings{Jang2022MPF,  
author = {Jang, Youngkyoon and Demiris, Yiannis},  
title = {Message Passing Framework for Vision Prediction Stability in Human Robot Interaction},  
booktitle = {IEEE International Conference on Robotics and Automation},  
year = {2022}  
}  
```
