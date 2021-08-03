cmake_minimum_required(VERSION 3.4)

find_package(OpenCV REQUIRED)
add_library(utils utils.cpp)
add_library(algorithm algorithm.cpp)

include_directories(${OpenCV_INCLUDE_DIRS} ${CMAKE_CURRENT_COURCE_DIR})

target_link_libraries(utils ${OpenCV_LIBS})
target_link_libraries(algorithm ${OpenCV_LIBS})
