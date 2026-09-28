cmake_minimum_required(VERSION 3.0.2)
project(crazyflie_test_ros1)

find_package(catkin REQUIRED COMPONENTS
  geometry_msgs
  rospy
  std_msgs
)

catkin_package(
  CATKIN_DEPENDS geometry_msgs rospy std_msgs
)

catkin_install_python(
  PROGRAMS scripts/dog_tracker_flight.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

install(DIRECTORY config launch
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
)
