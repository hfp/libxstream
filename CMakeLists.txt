cmake_minimum_required(VERSION 3.13 FATAL_ERROR)
project(libxstream
  VERSION 0.9.1
  LANGUAGES C)

include(GNUInstallDirs)
include(CMakePackageConfigHelpers)

set(LIBXS_GIT_REPOSITORY "https://github.com/hfp/libxs.git")
set(LIBXS_GIT_TAG "main")

find_package(libxs CONFIG QUIET
  HINTS "${PROJECT_SOURCE_DIR}/../libxs")
if(NOT TARGET libxs::libxs)
  message(STATUS "Cannot find libxs locally; downloading through FetchContent...")
  include(FetchContent)
  FetchContent_Declare(libxs
    GIT_REPOSITORY ${LIBXS_GIT_REPOSITORY}
    GIT_TAG ${LIBXS_GIT_TAG})
  FetchContent_MakeAvailable(libxs)
endif()
if(NOT TARGET libxs::libxs)
  message(FATAL_ERROR "libxs target libxs::libxs was not found or created")
endif()

find_package(OpenCL REQUIRED)
find_package(OpenMP REQUIRED COMPONENTS C)

set(LIBXSTREAM_LINK_LIBRARIES
  OpenCL::OpenCL
  OpenMP::OpenMP_C
  libxs::libxs)

message(STATUS "******** Build Summary ********")
message(STATUS "  CMake version         : ${CMAKE_VERSION}")
message(STATUS "  System                : ${CMAKE_SYSTEM_NAME}")
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

set(LIBXSTREAM_PUBLIC_COMPILE_DEFINITIONS __OPENCL __ACC __LIBXS)

file(GLOB LIBXSTREAM_SRCS LIST_DIRECTORIES false CONFIGURE_DEPENDS
  ${LIBXSTREAM_ROOT_DIR}/src/*.c)

add_library(libxstream ${LIBXSTREAM_SRCS})
add_library(libxstream::libxstream ALIAS libxstream)
set_target_properties(libxstream PROPERTIES
  OUTPUT_NAME xstream
  POSITION_INDEPENDENT_CODE ON)

target_compile_definitions(libxstream PRIVATE LIBXSTREAM_BUILD=1)
target_compile_definitions(libxstream PUBLIC ${LIBXSTREAM_PUBLIC_COMPILE_DEFINITIONS})

target_include_directories(libxstream
  PUBLIC
    ${LIBXSTREAM_PUBLIC_INCLUDE_DIRS}
  PRIVATE
    ${LIBXSTREAM_ROOT_DIR}/src)

target_link_libraries(libxstream PUBLIC ${LIBXSTREAM_LINK_LIBRARIES})

# Pkg-config file
set(PREFIX "${CMAKE_INSTALL_PREFIX}")
set(INCLUDEDIR "\${prefix}/${CMAKE_INSTALL_INCLUDEDIR}")
set(LIBDIR "\${prefix}/${CMAKE_INSTALL_LIBDIR}")
set(PROJECT "libxstream")
set(DESCRIPTION "Specialized tensor operations")
set(URL "https://github.com/hfp/libxstream/")
set(VERSION "${PROJECT_VERSION}")
set(REQUIRES_PRIVATE "Requires.private: libxs")
if(BUILD_SHARED_LIBS)
  set(LIBS "-L\${libdir} -lxstream")
  set(LIBS_PRIVATE "")
else()
  set(LIBS "\${libdir}/libxstream.a")
  set(LIBS_PRIVATE "Libs.private: -lpthread -ldl")
  string(REPLACE "libxs" "libxs-static" REQUIRES_PRIVATE "${REQUIRES_PRIVATE}")
endif()

configure_file(
  "${CMAKE_CURRENT_SOURCE_DIR}/scripts/libxstream.pc.in"
  "${CMAKE_CURRENT_BINARY_DIR}/libxstream.pc"
  @ONLY
)

configure_package_config_file(
  "${CMAKE_CURRENT_SOURCE_DIR}/scripts/libxstreamConfig.cmake.in"
  "${CMAKE_CURRENT_BINARY_DIR}/libxstreamConfig.cmake"
  INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/libxstream
  PATH_VARS CMAKE_INSTALL_INCLUDEDIR CMAKE_INSTALL_LIBDIR CMAKE_INSTALL_DATADIR)

write_basic_package_version_file(
  "${CMAKE_CURRENT_BINARY_DIR}/libxstreamConfigVersion.cmake"
  VERSION ${PROJECT_VERSION}
  COMPATIBILITY SameMajorVersion)

# Installation rules
install(TARGETS libxstream
  EXPORT libxstreamTargets
  ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
  RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})

install(DIRECTORY ${LIBXSTREAM_INCLUDE_DIR}/
  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/libxstream
  FILES_MATCHING PATTERN "*.h")

# Install header-only source files (always available)
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

# Install pkg-config files
install(FILES
  "${CMAKE_CURRENT_BINARY_DIR}/libxstream.pc"
  DESTINATION ${CMAKE_INSTALL_LIBDIR}/pkgconfig
)

install(PROGRAMS
  "${CMAKE_CURRENT_SOURCE_DIR}/scripts/tool_opencl.sh"
  DESTINATION ${CMAKE_INSTALL_DATADIR}/libxstream/scripts)

set(LIBXSTREAM_SMM_SRC
  "${CMAKE_CURRENT_SOURCE_DIR}/samples/smm/smm_acc.c"
  "${CMAKE_CURRENT_SOURCE_DIR}/samples/smm/smm_kernel.c"
  "${CMAKE_CURRENT_SOURCE_DIR}/samples/smm/smm_params.c"
  "${CMAKE_CURRENT_SOURCE_DIR}/samples/smm/smm_trans.c"
  "${CMAKE_CURRENT_SOURCE_DIR}/samples/smm/smm_acc_opencl.h"
  "${CMAKE_CURRENT_SOURCE_DIR}/samples/smm/acc.h"
  "${CMAKE_CURRENT_SOURCE_DIR}/samples/smm/acc_bench.h"
  "${CMAKE_CURRENT_SOURCE_DIR}/samples/smm/acc_libsmm.h")
install(FILES ${LIBXSTREAM_SMM_SRC}
  DESTINATION ${CMAKE_INSTALL_DATADIR}/libxstream/samples/smm)

install(DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/samples/smm/kernels/"
  DESTINATION ${CMAKE_INSTALL_DATADIR}/libxstream/samples/smm/kernels
  FILES_MATCHING PATTERN "*.cl")

install(FILES
  "${CMAKE_CURRENT_SOURCE_DIR}/samples/smm/tune_multiply.py"
  "${CMAKE_CURRENT_SOURCE_DIR}/samples/smm/tune_multiply.sh"
  DESTINATION ${CMAKE_INSTALL_DATADIR}/libxstream/samples/smm)

file(GLOB LIBXSTREAM_SMM_PARAMS LIST_DIRECTORIES false
  "${CMAKE_CURRENT_SOURCE_DIR}/samples/smm/params/*.bin"
  "${CMAKE_CURRENT_SOURCE_DIR}/samples/smm/params/*.csv")
install(FILES ${LIBXSTREAM_SMM_PARAMS}
  DESTINATION ${CMAKE_INSTALL_DATADIR}/libxstream/samples/smm/params)

file(GLOB LIBXSTREAM_SMM_CL LIST_DIRECTORIES false
  "${CMAKE_CURRENT_SOURCE_DIR}/samples/smm/kernels/*.cl")
file(GLOB LIBXSTREAM_SMM_CSV LIST_DIRECTORIES false
  "${CMAKE_CURRENT_SOURCE_DIR}/samples/smm/params/*.csv")
install(CODE "
  set(_out \"\${CMAKE_INSTALL_PREFIX}/${CMAKE_INSTALL_DATADIR}/libxstream/samples/smm/smm_kernels.h\")
  execute_process(
    COMMAND \"${CMAKE_CURRENT_SOURCE_DIR}/scripts/tool_opencl.sh\"
      ${LIBXSTREAM_SMM_CL} ${LIBXSTREAM_SMM_CSV} \${_out}
    RESULT_VARIABLE _rc)
  if(NOT _rc EQUAL 0)
    message(FATAL_ERROR \"tool_opencl.sh failed with code \${_rc}\")
  endif()
")

install(EXPORT libxstreamTargets
  FILE libxstreamTargets.cmake
  NAMESPACE libxstream::
  DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/libxstream)

install(FILES
  "${CMAKE_CURRENT_BINARY_DIR}/libxstreamConfig.cmake"
  "${CMAKE_CURRENT_BINARY_DIR}/libxstreamConfigVersion.cmake"
  DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/libxstream)

# Install documentation into share/libxstream
set(LIBXSTREAM_SHARE_DIR ${CMAKE_INSTALL_DATADIR}/libxstream)

file(GLOB LIBXSTREAM_DOC_MDS LIST_DIRECTORIES false
  "${CMAKE_CURRENT_SOURCE_DIR}/documentation/libxstream_*.md")
list(REMOVE_ITEM LIBXSTREAM_DOC_MDS
  "${CMAKE_CURRENT_SOURCE_DIR}/documentation/libxstream_samples.md"
  "${CMAKE_CURRENT_SOURCE_DIR}/documentation/libxstream_scripts.md")

set(LIBXSTREAM_DOC_EXTRA
  "${CMAKE_CURRENT_SOURCE_DIR}/documentation/index.md"
  "${CMAKE_CURRENT_SOURCE_DIR}/documentation/LICENSE.md")

file(GLOB LIBXSTREAM_DOC_PDFS LIST_DIRECTORIES false
  "${CMAKE_CURRENT_SOURCE_DIR}/documentation/*.pdf")

install(FILES ${LIBXSTREAM_DOC_MDS} ${LIBXSTREAM_DOC_EXTRA} ${LIBXSTREAM_DOC_PDFS}
  DESTINATION ${LIBXSTREAM_SHARE_DIR})
