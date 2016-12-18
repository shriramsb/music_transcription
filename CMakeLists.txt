cmake_minimum_required(VERSION 2.8)
project( music_transcription )
set(OpenCV_DIR '/home/shriram/opencv-3.1.0/build')
find_package( OpenCV 3.1 REQUIRED )
include_directories( include ${OpenCV_INCLUDE_DIRS} )

add_executable(keyboard_edge src/keyboard_edge.cpp)
add_executable(perspective_transform src/perspective_transform.cpp)
add_executable(key_detection src/key_detection.cpp)

target_link_libraries(keyboard_edge ${OpenCV_LIBS})
traget_link_libraries(perspective_transform ${OpenCV_LIBS})
traget_link_libraries(key_detection ${OpenCV_LIBS})