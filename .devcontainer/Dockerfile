FROM osrf/ros:noetic-desktop-full

# Arguments to pass during build (defaults to standard 1000:1000)
ARG USER_ID=1000
ARG GROUP_ID=1000
ARG USER_NAME=rob521

ENV DEBIAN_FRONTEND=noninteractive
ENV TURTLEBOT3_MODEL=waffle_pi

# Install OS Packages etc
RUN apt-get update && apt-get install -y \
    git \
    curl \
    wget 

RUN echo "deb http://packages.osrfoundation.org/gazebo/ubuntu-stable focal main" > /etc/apt/sources.list.d/gazebo-stable.list
RUN wget https://packages.osrfoundation.org/gazebo.key -O - | sudo apt-key add -

RUN apt-get update && apt-get install -y \
    ros-noetic-moveit \
    ros-noetic-catkin \
    python3-catkin-tools \
    python3-wstool \
    gazebo11 \
    ros-noetic-joy \
    ros-noetic-teleop-twist-joy \
    ros-noetic-teleop-twist-keyboard \
    ros-noetic-laser-proc \
    ros-noetic-rgbd-launch \
    ros-noetic-rosserial-arduino \
    ros-noetic-rosserial-python \
    ros-noetic-rosserial-client \
    ros-noetic-rosserial-msgs \
    ros-noetic-amcl \
    ros-noetic-map-server \
    ros-noetic-move-base \
    ros-noetic-urdf \
    ros-noetic-xacro \
    ros-noetic-compressed-image-transport \
    ros-noetic-rqt* \
    ros-noetic-rviz \
    ros-noetic-gmapping \
    ros-noetic-navigation \
    ros-noetic-interactive-markers \
    ros-noetic-dynamixel-sdk \
    ros-noetic-turtlebot3-msgs \
    ros-noetic-turtlebot3 \
    git \
    nano \
    sudo \
    mesa-utils \
    x11-utils \
    x11-apps \
    && rm -rf /var/lib/apt/lists/*

# Setup LCAS User
RUN useradd -rm -d /home/lcas -s /bin/bash -g root -G sudo -u 1001 lcas
RUN echo '%sudo ALL=(ALL) NOPASSWD:ALL' >> /etc/sudoers

RUN echo "source /usr/share/gazebo/setup.bash" >> /home/lcas/.bashrc \
    && echo "source /opt/ros/noetic/setup.bash" >> /home/lcas/.bashrc

# add commit history to bust cache if and when needed.
ADD "https://api.github.com/repos/LCAS/cmp9764/commits?per_page=1&sha=main" .lcas_addons.commits
