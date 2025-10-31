cmake_minimum_required(VERSION 2.8.3)
project(feature_tracker)

set(CMAKE_BUILD_TYPE "Release")
set(CMAKE_CXX_FLAGS "-std=c++11")
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -Wall -g")

find_package(catkin REQUIRED COMPONENTS
    roscpp
    std_msgs
    sensor_msgs
    cv_bridge
    camera_model
    yolov5_ros_msgs
    )

    
    find_package(OpenCV REQUIRED)
    
    
    set(LAPACKE_INCLUDE_DIR "/usr/local/include/")
    
    catkin_package()
    
    include_directories(
        ${EIGEN3_INCLUDE_DIR}
        ${LAPACK_INCLUDE_DIR}
        ${CBLAS_INCLUDE_DIR}
        ${catkin_INCLUDE_DIRS}
        )
        
        set(CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)
        find_package(Eigen3)
        include_directories(
            ${catkin_INCLUDE_DIRS}
            ${EIGEN3_INCLUDE_DIR}
            )
            
            add_executable(feature_tracker
            src/feature_tracker_node.cpp
            src/parameters.cpp
            src/feature_tracker.cpp
            src/ellipse_detect.cpp
            src/rect&tri_detect.cpp
            )
            
add_dependencies(feature_tracker yolov5_ros_msgs_gencpp)

target_link_libraries(feature_tracker ${catkin_LIBRARIES} ${OpenCV_LIBS}
-llapack 
-llapacke 
-lopenblas
-ltmglib )
