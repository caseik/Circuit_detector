cmake_minimum_required(VERSION 3.10)

project(CircuitDetector)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(OpenCV REQUIRED)

include_directories(
    ${OpenCV_INCLUDE_DIRS}
    .
    input
    pipeline
)

set(SOURCES
    main.cpp
    pipeline.cpp
    roi_detector.cpp
    symbol_classifier.cpp
    onnx_classifier.cpp
)

add_executable(circuit_detector ${SOURCES})

target_link_libraries(circuit_detector
    ${OpenCV_LIBS}
)