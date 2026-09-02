# FAST-LIO Adapter

This directory contains the initial external LiDAR-inertial odometry integration.

Its responsibility is translation between the external runtime and state-estimation ports. ROS messages, backend configuration, process lifecycle, topic names, and dependency-specific details must remain here and must not leak into domain or downstream module contracts.
