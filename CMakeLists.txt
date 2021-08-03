cmake_minimum_required(VERSION 3.4)
project(boat_detector)
set(CMAKE_CXX_STANDARD 17)

find_package(OpenCV REQUIRED)
add_subdirectory(lib)

include_directories(${OpenCV_INCLUDE_DIRS} ${CMAKE_CURRENT_COURCE_DIR})

add_executable(boat_detector src/main.cpp)

target_link_libraries(boat_detector ${OpenCV_LIBS} utils algorithm)
target_include_directories(boat_detector PUBLIC "${PROJECT_BINARY_DIR}""${PROJECT_SOURCE_DIR}/lib")
