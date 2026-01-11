cmake_minimum_required(VERSION 3.16)

# Project Name
project(FaceDetector LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_AUTOMOC ON)
set(CMAKE_AUTORCC ON)
set(CMAKE_AUTOUIC ON)

# -----------------------------------------------------------------------------
# 1. Setup Qt
# -----------------------------------------------------------------------------
find_package(Qt6 COMPONENTS Widgets REQUIRED)

# -----------------------------------------------------------------------------
# 2. Setup OpenCV
# -----------------------------------------------------------------------------
find_package(OpenCV REQUIRED)

# -----------------------------------------------------------------------------
# 3. Setup ONNX Runtime
# -----------------------------------------------------------------------------
set(ONNXRUNTIME_ROOT "C:/3rdParty/onnxruntime-win-x64-1.23.2")

if(NOT EXISTS "${ONNXRUNTIME_ROOT}")
    message(FATAL_ERROR "Could not find ONNX Runtime at: ${ONNXRUNTIME_ROOT}")
endif()

include_directories(${ONNXRUNTIME_ROOT}/include)
link_directories(${ONNXRUNTIME_ROOT}/lib)

# -----------------------------------------------------------------------------
# 4. Add Executable (FIXED: Lowercase & No 'src/' prefix)
# -----------------------------------------------------------------------------
add_executable(FaceDetector
    main.cpp

    # Lowercase names as you requested
    facesystem.cpp
    facesystem.h

    mainwindow.cpp
    mainwindow.h
)

# -----------------------------------------------------------------------------
# 5. Link Libraries
# -----------------------------------------------------------------------------
target_link_libraries(FaceDetector PRIVATE
    Qt6::Widgets
    ${OpenCV_LIBS}
    onnxruntime
)

# -----------------------------------------------------------------------------
# 6. Post-Build: Copy DLLs
# -----------------------------------------------------------------------------
add_custom_command(TARGET FaceDetector POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
    "${ONNXRUNTIME_ROOT}/lib/onnxruntime.dll"
    $<TARGET_FILE_DIR:FaceDetector>
    COMMENT "Copying onnxruntime.dll..."
)

if(EXISTS "${CMAKE_SOURCE_DIR}/models")
    add_custom_command(TARGET FaceDetector POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_directory
        "${CMAKE_SOURCE_DIR}/models"
        "$<TARGET_FILE_DIR:FaceDetector>/models"
        COMMENT "Copying models folder..."
    )
endif()
