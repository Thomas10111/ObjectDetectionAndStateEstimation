# ROS 2 Lyrical for the Jetson Orin, with Zenoh as the middleware.
#
# Build and run on the Orin with docker/run_orin.sh. Why Zenoh and shared
# memory: docs/rmw_comparison.md.
FROM ros:lyrical-perception

RUN apt-get update \
    && apt-get install -y --no-install-recommends ros-lyrical-rmw-zenoh-cpp \
    && rm -rf /var/lib/apt/lists/*

ENV RMW_IMPLEMENTATION=rmw_zenoh_cpp
ENV ZENOH_CONFIG_OVERRIDE=transport/shared_memory/enabled=true

RUN echo 'source /opt/ros/lyrical/setup.bash' >> /root/.bashrc

COPY entrypoint.sh /odse_entrypoint.sh
RUN chmod +x /odse_entrypoint.sh
ENTRYPOINT ["/odse_entrypoint.sh"]
CMD ["bash"]
