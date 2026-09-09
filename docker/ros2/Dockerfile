FROM osrf/ros:humble-desktop

SHELL ["/bin/bash", "-c"]
ARG FAST_LIO_ROS2_REPOSITORY=https://github.com/Ericsii/FAST_LIO_ROS2.git
ARG FAST_LIO_REF=main
RUN apt-get update && apt-get install -y --no-install-recommends git python3-pip python3-colcon-common-extensions && rm -rf /var/lib/apt/lists/* \
    && mkdir -p /opt/fast-lio/src \
    && git clone --depth 1 --branch "${FAST_LIO_REF}" "${FAST_LIO_ROS2_REPOSITORY}" /opt/fast-lio/src/FAST_LIO_ROS2
RUN source /opt/ros/humble/setup.bash && cd /opt/fast-lio && rosdep update && rosdep install --from-paths src --ignore-src -r -y && colcon build
WORKDIR /workspace
CMD ["bash"]
