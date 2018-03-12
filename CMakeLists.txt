CMAKE_MINIMUM_REQUIRED(VERSION 2.6)

# PROJECT CONFIGURATION: Project names and version numbers
SET (PROJECT_NAME zero-latency)
SET (VERSION_MAJOR 1)
SET (VERSION_MINOR 0)

PROJECT(${PROJECT_NAME})
STRING(TOUPPER ${PROJECT_NAME} PREFIX)

SET(DEFAULT_BUILD_TYPE "Release")

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  message(STATUS "Setting build type to '${DEFAULT_BUILD_TYPE}' as none was specified.")
  set(CMAKE_BUILD_TYPE "${DEFAULT_BUILD_TYPE}" CACHE
      STRING "Choose the type of build." FORCE)
  # Set the possible values of build type for cmake-gui
  set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS
    "Debug" "Release" "MinSizeRel" "RelWithDebInfo")
endif()

list(APPEND CMAKE_MODULE_PATH "${${PROJECT_NAME}_SOURCE_DIR}/CMakeModules")

find_package(CUDA)
find_package(nvpipe)

include_directories(${EGL_INCLUDE_DIR} ${GLEW_INCLUDE_DIRS} ${CUDA_INCLUDE_DIRS})

set(SOURCE_FILES_SERVER src/server.cpp)
set(SOURCE_FILES_CLIENT src/client.cpp)

add_executable(${PROJECT_NAME}-server ${SOURCE_FILES_SERVER})
target_link_libraries(${PROJECT_NAME}-server nvpipe ${EGL_LIBRARIES} ${GLEW_LIBRARIES} ${CUDA_LIBRARIES})

add_executable(${PROJECT_NAME}-client ${SOURCE_FILES_CLIENT})
target_link_libraries(${PROJECT_NAME}-client nvpipe ${EGL_LIBRARIES} ${GLEW_LIBRARIES} ${CUDA_LIBRARIES})
