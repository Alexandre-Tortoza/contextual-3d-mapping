FROM osrf/ros:humble-desktop

SHELL ["/bin/bash", "-c"]
ARG FAST_LIO_REF=2fffc570a25d0df172720bac034fbdb6a13d2162
ARG LIVOX_SDK2_REF=08f523c930b2f0ba1e98a6afaa8d7476bf479908
ARG LIVOX_ROS_DRIVER2_REF=4a1def929e5b59c7a8122d19fce6efba581ce9f7
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential \
        cmake \
        git \
        python3-colcon-common-extensions \
        ros-humble-pcl-conversions \
        ros-humble-pcl-ros \
    && rm -rf /var/lib/apt/lists/*
RUN git clone https://github.com/Livox-SDK/Livox-SDK2.git /tmp/Livox-SDK2 \
    && git -C /tmp/Livox-SDK2 checkout --detach "${LIVOX_SDK2_REF}" \
    && cmake -S /tmp/Livox-SDK2 -B /tmp/Livox-SDK2/build -DCMAKE_BUILD_TYPE=Release \
    && cmake --build /tmp/Livox-SDK2/build --parallel \
    && cmake --install /tmp/Livox-SDK2/build \
    && rm -rf /tmp/Livox-SDK2
RUN mkdir -p /opt/fast-lio/src \
    && git clone https://github.com/Livox-SDK/livox_ros_driver2.git /opt/fast-lio/src/livox_ros_driver2 \
    && git -C /opt/fast-lio/src/livox_ros_driver2 checkout --detach "${LIVOX_ROS_DRIVER2_REF}" \
    && cp /opt/fast-lio/src/livox_ros_driver2/package_ROS2.xml /opt/fast-lio/src/livox_ros_driver2/package.xml \
    && cp -r /opt/fast-lio/src/livox_ros_driver2/launch_ROS2 /opt/fast-lio/src/livox_ros_driver2/launch \
    && git clone https://github.com/Ericsii/FAST_LIO_ROS2.git /opt/fast-lio/src/FAST_LIO_ROS2 \
    && git -C /opt/fast-lio/src/FAST_LIO_ROS2 checkout --detach "${FAST_LIO_REF}" \
    && git -C /opt/fast-lio/src/FAST_LIO_ROS2 submodule update --init --recursive
RUN source /opt/ros/humble/setup.bash \
    && cd /opt/fast-lio \
    && colcon build --symlink-install --cmake-args -DCMAKE_BUILD_TYPE=Release -DROS_EDITION=ROS2 -DDISTRO_ROS=humble
WORKDIR /workspace
CMD ["bash"]
