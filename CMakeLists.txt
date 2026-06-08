cmake_minimum_required(VERSION 3.13 FATAL_ERROR)
project(libxstream
  VERSION 0.9.1
  LANGUAGES C)

include(GNUInstallDirs)
include(CMakePackageConfigHelpers)

option(LIBXSTREAM_HEADER_ONLY "Header-only usage (no library target)" OFF)
option(LIBXSTREAM_SHARED "Build shared library instead of static" OFF)

find_package(libxs CONFIG REQUIRED)
find_package(OpenCL REQUIRED)
find_package(OpenMP REQUIRED COMPONENTS C)

set(LIBXSTREAM_LINK_LIBRARIES
  OpenCL::OpenCL
  OpenMP::OpenMP_C
  libxs::libxs)

message(STATUS "******** Build Summary ********")
message(STATUS "  CMake version         : ${CMAKE_VERSION}")
message(STATUS "  System                : ${CMAKE_SYSTEM_NAME}")
message(STATUS "  LIBXSTREAM_HEADER_ONLY: ${LIBXSTREAM_HEADER_ONLY}")
message(STATUS "  LIBXSTREAM_SHARED     : ${LIBXSTREAM_SHARED}")
message(STATUS "  libxs target          : libxs::libxs")
message(STATUS "  OpenCL                : ${OpenCL_VERSION_STRING}")

set(LIBXSTREAM_ROOT_DIR ${PROJECT_SOURCE_DIR})
set(LIBXSTREAM_INCLUDE_DIR ${LIBXSTREAM_ROOT_DIR}/libxstream)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Generate include/libxstream_version.h
set(LIBXSTREAM_VERSION_HEADER "${LIBXSTREAM_INCLUDE_DIR}/libxstream_version.h")
set(LIBXSTREAM_VERSION_SCRIPT "${CMAKE_CURRENT_SOURCE_DIR}/scripts/tool_version.sh")

if(NOT EXISTS "${LIBXSTREAM_VERSION_SCRIPT}")
  message(FATAL_ERROR "Missing ${LIBXSTREAM_VERSION_SCRIPT}")
endif()

find_program(SH_EXECUTABLE NAMES sh bash REQUIRED)

execute_process(
  COMMAND "${SH_EXECUTABLE}" "${LIBXSTREAM_VERSION_SCRIPT}" libxstream -1
  OUTPUT_FILE "${LIBXSTREAM_VERSION_HEADER}"
  RESULT_VARIABLE LIBXSTREAM_VERSION_RC
  WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
)

if(NOT LIBXSTREAM_VERSION_RC EQUAL 0)
  message(FATAL_ERROR "tool_version.sh failed with code ${LIBXSTREAM_VERSION_RC}")
endif()

set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS
  "${LIBXSTREAM_VERSION_SCRIPT}")

set(LIBXSTREAM_PUBLIC_INCLUDE_DIRS
  $<BUILD_INTERFACE:${LIBXSTREAM_ROOT_DIR}>
  $<BUILD_INTERFACE:${LIBXSTREAM_INCLUDE_DIR}>
  $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
  $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}/libxstream>)

set(LIBXSTREAM_PUBLIC_COMPILE_DEFINITIONS __OPENCL __ACC)
if(NOT LIBXS_HEADER_ONLY)
  list(APPEND LIBXSTREAM_PUBLIC_COMPILE_DEFINITIONS __LIBXS)
endif()

if(LIBXSTREAM_HEADER_ONLY)
  add_library(libxstream INTERFACE)
  target_include_directories(libxstream INTERFACE ${LIBXSTREAM_PUBLIC_INCLUDE_DIRS})
  target_compile_definitions(libxstream INTERFACE
    ${LIBXSTREAM_PUBLIC_COMPILE_DEFINITIONS}
    LIBXSTREAM_SOURCE=1)
  target_link_libraries(libxstream INTERFACE ${LIBXSTREAM_LINK_LIBRARIES})
  add_library(libxstream::libxstream ALIAS libxstream)
else()
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
  target_compile_definitions(libxstream PUBLIC ${LIBXSTREAM_PUBLIC_COMPILE_DEFINITIONS})

  target_include_directories(libxstream
    PUBLIC
      ${LIBXSTREAM_PUBLIC_INCLUDE_DIRS}
    PRIVATE
      ${LIBXSTREAM_ROOT_DIR}/src)

  target_link_libraries(libxstream PUBLIC ${LIBXSTREAM_LINK_LIBRARIES})
endif()

# Pkg-config file
if(NOT LIBXSTREAM_HEADER_ONLY)
  set(PREFIX "${CMAKE_INSTALL_PREFIX}")
  set(INCLUDEDIR "\${prefix}/${CMAKE_INSTALL_INCLUDEDIR}")
  set(LIBDIR "\${prefix}/${CMAKE_INSTALL_LIBDIR}")
  set(PROJECT "libxstream")
  set(DESCRIPTION "Specialized tensor operations")
  set(URL "https://github.com/hfp/libxstream/")
  set(VERSION "${PROJECT_VERSION}")
  set(LINKNAME "xstream")
  set(LIBS_PRIVATE "")
  set(REQUIRES_PRIVATE "Requires.private: libxs")
  if(NOT LIBXSTREAM_SHARED)
    set(LIBS_PRIVATE "Libs.private: -lpthread -ldl")
  endif()

  configure_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/scripts/libxstream.pc.in"
    "${CMAKE_CURRENT_BINARY_DIR}/libxstream.pc"
    @ONLY
  )
endif()

configure_package_config_file(
  "${CMAKE_CURRENT_SOURCE_DIR}/scripts/libxstreamConfig.cmake.in"
  "${CMAKE_CURRENT_BINARY_DIR}/libxstreamConfig.cmake"
  INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/libxstream
  PATH_VARS CMAKE_INSTALL_INCLUDEDIR CMAKE_INSTALL_LIBDIR)

write_basic_package_version_file(
  "${CMAKE_CURRENT_BINARY_DIR}/libxstreamConfigVersion.cmake"
  VERSION ${PROJECT_VERSION}
  COMPATIBILITY SameMajorVersion)

# Installation rules
if(LIBXSTREAM_HEADER_ONLY)
  install(TARGETS libxstream
    EXPORT libxstreamTargets)
else()
  install(TARGETS libxstream
    EXPORT libxstreamTargets
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
endif()

install(DIRECTORY ${LIBXSTREAM_INCLUDE_DIR}/
  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/libxstream
  FILES_MATCHING PATTERN "*.h")

if(LIBXSTREAM_HEADER_ONLY)
  file(READ "${LIBXSTREAM_INCLUDE_DIR}/libxstream_source.h" LIBXSTREAM_SOURCE_HEADER)
  string(REPLACE "../src/" "src/" LIBXSTREAM_SOURCE_HEADER "${LIBXSTREAM_SOURCE_HEADER}")
  file(WRITE "${CMAKE_CURRENT_BINARY_DIR}/libxstream_source.h" "${LIBXSTREAM_SOURCE_HEADER}")

  install(FILES "${CMAKE_CURRENT_BINARY_DIR}/libxstream_source.h"
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/libxstream)
  install(DIRECTORY ${LIBXSTREAM_ROOT_DIR}/src/
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/libxstream/src
    FILES_MATCHING
      PATTERN "*.c"
      PATTERN "*.h")
endif()

if(NOT LIBXSTREAM_HEADER_ONLY)
  install(FILES
    "${CMAKE_CURRENT_BINARY_DIR}/libxstream.pc"
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/pkgconfig
  )
endif()

install(EXPORT libxstreamTargets
  FILE libxstreamTargets.cmake
  NAMESPACE libxstream::
  DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/libxstream)

install(FILES
  "${CMAKE_CURRENT_BINARY_DIR}/libxstreamConfig.cmake"
  "${CMAKE_CURRENT_BINARY_DIR}/libxstreamConfigVersion.cmake"
  DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/libxstream)
