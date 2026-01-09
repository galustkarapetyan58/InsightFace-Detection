cmake_minimum_required(VERSION 3.16)
project(FaceDetector)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ==============================================================================
# 1. PATH CONFIGURATION
# ==============================================================================
# OpenCV Path (The one you found)
set(OpenCV_DIR "C:/3rdParty/opencv" CACHE PATH "Path to OpenCV" FORCE)

# ONNX Runtime Path
set(ONNX_ROOT "C:/3rdParty/onnxruntime-win-x64-1.23.2")

# ==============================================================================
# 2. FIND PACKAGES
# ==============================================================================
# Find OpenCV
find_package(OpenCV REQUIRED)

# Find Qt (Fixes the QCoreApplication error)
find_package(Qt6 REQUIRED COMPONENTS Core)

# Check ONNX Headers
if(NOT EXISTS "${ONNX_ROOT}/include/onnxruntime_cxx_api.h")
    message(FATAL_ERROR "ONNX headers not found! Check path: ${ONNX_ROOT}")
endif()

# ==============================================================================
# 3. BUILD SETUP
# ==============================================================================
add_executable(FaceDetector
    main.cpp
    facesystem.cpp
    facesystem.h
)

# ==============================================================================
# 4. LINKING (The most important part)
# ==============================================================================
# Include headers
target_include_directories(FaceDetector PRIVATE
    ${OpenCV_INCLUDE_DIRS}
    "${ONNX_ROOT}/include"
)

# Link libraries (OpenCV + ONNX + Qt)
target_link_libraries(FaceDetector PRIVATE
    ${OpenCV_LIBS}
    "${ONNX_ROOT}/lib/onnxruntime.lib"
    Qt6::Core  # <--- This links QCoreApplication
)

# ==============================================================================
# 5. POST-BUILD
# ==============================================================================
# Copy ONNX DLL to build folder
add_custom_command(TARGET FaceDetector POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
    "${ONNX_ROOT}/lib/onnxruntime.dll"
    $<TARGET_FILE_DIR:FaceDetector>
)
