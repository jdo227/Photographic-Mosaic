cmake_minimum_required(VERSION 3.10)
# set the project name
project(Mosaic)
find_package(X11 REQUIRED)
find_package(Threads REQUIRED)
find_package(OpenCV REQUIRED)

if(COMMAND cmake_policy)
	cmake_policy(SET CMP0003 NEW)
endif(COMMAND cmake_policy)

# add the executable
add_executable(Mosaic main.cpp)

# Specify the C++ standard
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED True)

SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -pthread")
# PTHREAD
SET(THREADS_PREFER_PTHREAD_FLAG ON)
#Add lib for fs
TARGET_LINK_LIBRARIES(Mosaic stdc++fs)

# Set C++14 support
set_property(TARGET Mosaic PROPERTY CXX_STANDARD 14)
set_property(TARGET Mosaic PROPERTY CXX_STANDARD_REQUIRED ON)

# link opencv library
include_directories(${X11_INCLUDE_DIR})
target_link_libraries( Mosaic ${OpenCV_LIBS} )
target_link_libraries(Mosaic Threads::Threads)
target_link_libraries(Mosaic X11)
