cmake_minimum_required(VERSION 2.8)
project( music_transcription )
set(OpenCV_DIR '/home/shriram/opencv-3.1.0/build')
find_package(OpenCV 3.1 REQUIRED)
include_directories(include ${OpenCV_INCLUDE_DIRS})

add_executable(build/frame_picker src/frame_picker.cpp)
target_link_libraries(build/frame_picker ${OpenCV_LIBS})
