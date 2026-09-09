FROM osrf/ros:noetic-desktop-full

SHELL ["/bin/bash", "-c"]
ARG FAST_LIO_REF=main
RUN apt-get update && apt-get install -y --no-install-recommends git python3-pip && rm -rf /var/lib/apt/lists/* \
    && mkdir -p /opt/fast-lio/src \
    && git clone --depth 1 --branch "${FAST_LIO_REF}" https://github.com/hku-mars/FAST_LIO.git /opt/fast-lio/src/FAST_LIO
RUN source /opt/ros/noetic/setup.bash && cd /opt/fast-lio && catkin_make
WORKDIR /workspace
CMD ["bash"]
