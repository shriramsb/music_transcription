cmake_minimum_required(VERSION 2.8)
project( music_transcription )
set(OpenCV_DIR '/home/shriram/opencv-3.1.0/build')
find_package(OpenCV 3.1 REQUIRED)
include_directories(include ${OpenCV_INCLUDE_DIRS})

add_executable(build/frame_picker src/frame_picker.cpp)
target_link_libraries(build/frame_picker ${OpenCV_LIBS})

add_executable(build/keyboard_edge src/keyboard_edge.cpp)
target_link_libraries(build/keyboard_edge ${OpenCV_LIBS})

add_executable(build/perspective_transform src/perspective_transform.cpp)
target_link_libraries(build/perspective_transform ${OpenCV_LIBS})

add_executable(build/key_detection src/key_detection.cpp src/filehandle_vov.cpp src/my_contours_utility.cpp)
target_link_libraries(build/key_detection ${OpenCV_LIBS})

add_executable(build/key_press_detector src/key_press_detector.cpp src/filehandle_vov.cpp src/my_contours_utility.cpp)
target_link_libraries(build/key_press_detector ${OpenCV_LIBS})