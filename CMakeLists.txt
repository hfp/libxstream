cmake_minimum_required(VERSION 3.13 FATAL_ERROR)
project(libxstream
  VERSION 0.9.1
  LANGUAGES C)

option(LIBXSTREAM_HEADER_ONLY "Header-only usage (no library target)" OFF)
option(LIBXSTREAM_SHARED "Build shared library instead of static" OFF)
set(LIBXSROOT "${PROJECT_SOURCE_DIR}/../libxs" CACHE PATH
  "Root of the LIBXS library (sibling directory by default)")

if(NOT IS_DIRECTORY "${LIBXSROOT}")
  message(FATAL_ERROR
    "LIBXS not found at ${LIBXSROOT}\n"
    "Set -DLIBXSROOT=<path> to the LIBXS source/install tree.")
endif()
set(LIBXS_INCLUDE_DIR "${LIBXSROOT}/include")

if(NOT LIBXSTREAM_HEADER_ONLY)
  find_package(OpenCL REQUIRED)
endif()

message(STATUS "******** Build Summary ********")
message(STATUS "  CMake version         : ${CMAKE_VERSION}")
message(STATUS "  System                : ${CMAKE_SYSTEM_NAME}")
message(STATUS "  LIBXSTREAM_HEADER_ONLY: ${LIBXSTREAM_HEADER_ONLY}")
message(STATUS "  LIBXSTREAM_SHARED     : ${LIBXSTREAM_SHARED}")
message(STATUS "  LIBXSROOT             : ${LIBXSROOT}")
if(NOT LIBXSTREAM_HEADER_ONLY)
  message(STATUS "  OpenCL                : ${OpenCL_VERSION_STRING}")
endif()

set(LIBXSTREAM_ROOT_DIR ${PROJECT_SOURCE_DIR})
set(LIBXSTREAM_INCLUDE_DIR ${LIBXSTREAM_ROOT_DIR}/include)

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

if(LIBXSTREAM_HEADER_ONLY)
  add_library(libxstream INTERFACE)
  target_include_directories(libxstream INTERFACE
    $<BUILD_INTERFACE:${LIBXSTREAM_INCLUDE_DIR}>
    $<BUILD_INTERFACE:${LIBXS_INCLUDE_DIR}>
    $<INSTALL_INTERFACE:include>)
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
  set(REQUIRES_PRIVATE "")
  if(NOT LIBXSTREAM_SHARED)
    set(LIBS_PRIVATE "Libs.private: -lpthread -ldl")
    if(LIBXSTREAM_HAS_LIBM)
      string(APPEND LIBS_PRIVATE " -lm")
    endif()
    if(LIBXSTREAM_HAS_LIBRT)
      string(APPEND LIBS_PRIVATE " -lrt")
    endif()
    if(LIBXS_LIBRARY)
      set(REQUIRES_PRIVATE "Requires.private: libxs-static")
    endif()
  else()
    if(LIBXS_LIBRARY)
      set(REQUIRES_PRIVATE "Requires.private: libxs")
    endif()
  endif()

  configure_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/scripts/libxstream.pc.in"
    "${CMAKE_CURRENT_BINARY_DIR}/libxstream.pc"
    @ONLY
  )
endif()

# Installation rules
include(GNUInstallDirs)

if(LIBXSTREAM_HEADER_ONLY)
  install(TARGETS libxstream
    EXPORT libxstreamTargets
    INCLUDES DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
else()
  install(TARGETS libxstream
    EXPORT libxstreamTargets
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
endif()

install(DIRECTORY ${LIBXSTREAM_INCLUDE_DIR}/
  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
  FILES_MATCHING PATTERN "*.h")

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
