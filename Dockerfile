FROM nvidia/cuda:11.1-devel-ubuntu20.04

ARG ROS_PKG=ros_base
ENV ROS_DISTRO=noetic
ENV ROS_ROOT=/opt/ros/${ROS_DISTRO}
ENV ROS_PYTHON_VERSION=3

# # nvidia-container-runtime
ENV NVIDIA_VISIBLE_DEVICES all
ENV NVIDIA_DRIVER_CAPABILITIES all


RUN apt update && \
    DEBIAN_FRONTEND=noninteractive apt install -y software-properties-common && \
    add-apt-repository ppa:deadsnakes/ppa && \
    apt update && \
    DEBIAN_FRONTEND=noninteractive apt install -y python3.8 python3-pip python-is-python3


# ####################################################################################################
# ######################################### ROS INSTALLATION #########################################
# ####################################################################################################

WORKDIR /workspace

RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
          git \
		cmake \
		build-essential \
		curl \
		wget \
		gnupg2 \
		lsb-release \
		ca-certificates \
    && rm -rf /var/lib/apt/lists/*

RUN sh -c 'echo "deb http://packages.ros.org/ros/ubuntu $(lsb_release -sc) main" > /etc/apt/sources.list.d/ros-latest.list' \
     && curl -s https://raw.githubusercontent.com/ros/rosdistro/master/ros.asc | apt-key add -


# install bootstrap dependencies
RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        libpython3-dev \
        python3-rosdep \
        python3-rosinstall-generator \
        python3-vcstool \
        python3-catkin-tools \
        build-essential && \
    rosdep init && \
    rosdep update

#
# download/build the ROS source
#
RUN mkdir ros_catkin_ws && cd ros_catkin_ws && \
    rosinstall_generator ${ROS_PKG} vision_msgs --rosdistro ${ROS_DISTRO} --deps --tar > ${ROS_DISTRO}-${ROS_PKG}.rosinstall && \
    mkdir src && \
    vcs import --input ${ROS_DISTRO}-${ROS_PKG}.rosinstall ./src && \
    apt-get update && rosdep install --from-paths ./src --ignore-packages-from-source --rosdistro ${ROS_DISTRO} --skip-keys python3-pykdl -y && \
    python3 ./src/catkin/bin/catkin_make_isolated --install --install-space ${ROS_ROOT} -DCMAKE_BUILD_TYPE=Release 

RUN echo 'source /opt/ros/${ROS_DISTRO}/setup.bash' >> /root/.bashrc 

COPY ./ros_entrypoint.sh /ros_entrypoint.sh
RUN chmod u+x /ros_entrypoint.sh
ENTRYPOINT ["/ros_entrypoint.sh"]

WORKDIR /ros-ws
RUN catkin init

####it remove build cache: docker builder prune
# ####################################################################################################
# ####################################### PYTORCH INSTALLATION #######################################
# ####################################################################################################

RUN pip install torch==1.9.0+cu111 torchvision==0.10.0+cu111 torchaudio==0.9.0 -f https://download.pytorch.org/whl/torch_stable.html
# RUN pip install torch==1.8.0+cu111 torchvision==0.9.0+cu111 torchaudio==0.8.0 -f https://download.pytorch.org/whl/torch_stable.html



# ####################################################################################################
# ####################################### REALSENSE INSTALL ##########################################
# ####################################################################################################

RUN apt install -y ros-noetic-realsense2-camera


####################################################################################################
##################################### MMDETECTION INSTALLATION #####################################
####################################################################################################

ENV TORCH_NVCC_FLAGS="-Xfatbin -compress-all"

RUN apt-get install -y ffmpeg libsm6 libxext6 git ninja-build libglib2.0-0 \
    libsm6 libxrender-dev libxext6 

ENV FORCE_CUDA="1"
# Install MMCV
RUN pip install mmcv-full -f \
        https://download.openmmlab.com/mmcv/dist/cu111/torch1.9.0/index.html
# RUN pip install mmcv-full -f \
#         https://download.openmmlab.com/mmcv/dist/cu111/torch1.8.0/index.html

# Install MMDetection
RUN git clone https://github.com/open-mmlab/mmdetection.git /mmdetection
WORKDIR /mmdetection
RUN pip install -r requirements/build.txt
RUN pip install --no-cache-dir -e .

# SHELL ["/bin/bash", "-c"]
WORKDIR /ros-ws

####################################################################################################
####################################### MMPOSE INSTALLATION ########################################
####################################################################################################
RUN apt-get update && apt-get install -y git ninja-build libglib2.0-0 libsm6 libxrender-dev libxext6 libgl1-mesa-glx\
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Install MMDetection
RUN pip install cython
RUN pip install xtcocotools

# RUN conda clean --all

RUN git clone https://github.com/open-mmlab/mmpose.git /mmpose
WORKDIR /mmpose
RUN mkdir -p /mmpose/data
ENV FORCE_CUDA="1"
RUN pip install -r requirements/runtime.txt
RUN pip install -r requirements/tests.txt
RUN pip install --no-cache-dir -e .


# SHELL ["/bin/bash", "-c"]
WORKDIR /ros-ws

################################################################################################################
####################################### HAND_CONTACT_STATE INSTALLATION ########################################
################################################################################################################
## Download 
RUN git clone https://github.com/ddshan/hand_object_detector /hand_object_detector

WORKDIR /

## removing opencv-python line from /hand_object_detector/requirements.txt
COPY /tas_perception/scripts_helper_in_install/remove_opencv_from_a_file.py /hand_object_detector/remove_opencv_from_a_file.py
COPY /tas_perception/hand_object/net_utils.py /hand_object_detector/lib/model/utils/net_utils.py
WORKDIR /hand_object_detector
RUN python remove_opencv_from_a_file.py 

RUN pip install -r requirements.txt

## instead of using setup.py, pip install will find setup.py under the current folder to install it.
RUN cd lib && pip install --no-cache-dir -e .


SHELL ["/bin/bash", "-c"]
WORKDIR /ros-ws


# RUN pip install \
#     sklearn \
#     face_recognition



RUN mkdir src 

COPY tas_perception src/tas_perception

RUN source /opt/ros/noetic/setup.bash && catkin build


# CMD bash
CMD source devel/setup.bash && roslaunch tas_perception main.launch --screen
