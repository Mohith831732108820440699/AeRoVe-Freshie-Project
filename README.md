unzip freshieproject.zip -d ~

cd ~/recon_ws/src
git clone https://github.com/PX4/px4_msgs.git --depth 1

cd ~/recon_ws
source /opt/ros/jazzy/setup.bash
colcon build --symlink-install

echo "source /opt/ros/jazzy/setup.bash" >> ~/.bashrc
echo "source ~/recon_ws/install/setup.bash" >> ~/.bashrc
source ~/.bashrc

mkdir -p "/home/$USER/.gz/fuel/fuel.gazebosim.org/openrobotics/models/x3 uav/4/"
cp ~/recon_ws/x3_model.sdf "/home/$USER/.gz/fuel/fuel.gazebosim.org/openrobotics/models/x3 uav/4/model.sdf"

THEN RUN THESE ON 4 SEPARATE TERMINALS

1: pkill -9 -f gz; sleep 3
gz sim ~/recon_ws/recon_mission.sdf

2: ros2 run ros_gz_bridge parameter_bridge \
  /model/x3/odometry@nav_msgs/msg/Odometry[gz.msgs.Odometry \
  /X3/gazebo/command/twist@geometry_msgs/msg/Twist]gz.msgs.Twist \
  /X3/enable@std_msgs/msg/Bool]gz.msgs.Boolean \
  /overhead_camera/image@sensor_msgs/msg/Image[gz.msgs.Image \
  /drone_camera/image@sensor_msgs/msg/Image[gz.msgs.Image

3: ros2 topic pub --once /X3/enable std_msgs/msg/Bool "data: true"
sleep 2
ros2 run recon_mission waypoint_nav

4: ros2 run recon_mission opencv_processor
