cmake_minimum_required(VERSION 3.13 FATAL_ERROR)
project(libxstream
  VERSION 0.9.1
  LANGUAGES C)

option(LIBXSTREAM_SHARED "Build shared library instead of static" OFF)
set(LIBXSROOT "${PROJECT_SOURCE_DIR}/../libxs" CACHE PATH
  "Root of the LIBXS library (sibling directory by default)")

if(NOT IS_DIRECTORY "${LIBXSROOT}")
  message(FATAL_ERROR
    "LIBXS not found at ${LIBXSROOT}\n"
    "Set -DLIBXSROOT=<path> to the LIBXS source/install tree.")
endif()
set(LIBXS_INCLUDE_DIR "${LIBXSROOT}/include")

find_package(OpenCL REQUIRED)

message(STATUS "******** Build Summary ********")
message(STATUS "  CMake version         : ${CMAKE_VERSION}")
message(STATUS "  System                : ${CMAKE_SYSTEM_NAME}")
message(STATUS "  LIBXSTREAM_SHARED     : ${LIBXSTREAM_SHARED}")
message(STATUS "  LIBXSROOT             : ${LIBXSROOT}")
message(STATUS "  OpenCL                : ${OpenCL_VERSION_STRING}")

set(LIBXSTREAM_ROOT_DIR ${PROJECT_SOURCE_DIR})
set(LIBXSTREAM_INCLUDE_DIR ${LIBXSTREAM_ROOT_DIR}/include)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

file(GLOB LIBXSTREAM_SRCS LIST_DIRECTORIES false CONFIGURE_DEPENDS
  ${LIBXSTREAM_ROOT_DIR}/src/*.c)

if(LIBXSTREAM_SHARED)
  add_library(libxstream SHARED ${LIBXSTREAM_SRCS})
else()
  add_library(libxstream STATIC ${LIBXSTREAM_SRCS})
endif()
add_library(libxstream::libxstream ALIAS libxstream)
set_target_properties(libxstream PROPERTIES OUTPUT_NAME xstream)

target_compile_definitions(libxstream PRIVATE LIBXSTREAM_BUILD=1)
target_compile_definitions(libxstream PUBLIC __OPENCL __ACC)

target_include_directories(libxstream
  PUBLIC
    $<BUILD_INTERFACE:${LIBXSTREAM_INCLUDE_DIR}>
    $<BUILD_INTERFACE:${LIBXS_INCLUDE_DIR}>
    $<INSTALL_INTERFACE:include>
  PRIVATE
    ${LIBXSTREAM_ROOT_DIR}/src)

target_link_libraries(libxstream PUBLIC OpenCL::OpenCL)

find_package(OpenMP REQUIRED)
target_link_libraries(libxstream PUBLIC OpenMP::OpenMP_C)

# prefer pre-built LIBXS library, fall back to header-only (libxs_source.h)
find_library(LIBXS_LIBRARY NAMES xs libxs HINTS "${LIBXSROOT}/lib")
if(LIBXS_LIBRARY)
  target_compile_definitions(libxstream PUBLIC __LIBXS)
  target_link_libraries(libxstream PUBLIC "${LIBXS_LIBRARY}")
endif()

# transitive dependencies of LIBXS (pthread, dl, m, rt)
set(THREADS_PREFER_PTHREAD_FLAG ON)
find_package(Threads REQUIRED)
target_link_libraries(libxstream PUBLIC Threads::Threads)
target_link_libraries(libxstream PUBLIC ${CMAKE_DL_LIBS})

include(CheckLibraryExists)
check_library_exists(m sqrt "" LIBXSTREAM_HAS_LIBM)
if(LIBXSTREAM_HAS_LIBM)
  target_link_libraries(libxstream PUBLIC m)
endif()
check_library_exists(rt clock_gettime "" LIBXSTREAM_HAS_LIBRT)
if(LIBXSTREAM_HAS_LIBRT)
  target_link_libraries(libxstream PUBLIC rt)
endif()
