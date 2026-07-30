cmake_minimum_required(VERSION 3.13 FATAL_ERROR)

set(LIBXSTREAM_VERSION_FILE "${CMAKE_CURRENT_SOURCE_DIR}/version.txt")
if(NOT EXISTS "${LIBXSTREAM_VERSION_FILE}")
  message(FATAL_ERROR "Missing ${LIBXSTREAM_VERSION_FILE}")
endif()

file(READ "${LIBXSTREAM_VERSION_FILE}" LIBXSTREAM_VERSION)
string(STRIP "${LIBXSTREAM_VERSION}" LIBXSTREAM_VERSION)
if(NOT LIBXSTREAM_VERSION MATCHES "^[0-9]+\\.[0-9]+\\.[0-9]+$")
  message(FATAL_ERROR
    "Invalid version '${LIBXSTREAM_VERSION}' in ${LIBXSTREAM_VERSION_FILE}; expected MAJOR.MINOR.PATCH")
endif()

project(libxstream
  VERSION "${LIBXSTREAM_VERSION}"
  LANGUAGES C)

include(GNUInstallDirs)
include(CMakePackageConfigHelpers)

option(LIBXSTREAM_INSTALL_HEADER_ONLY "Install the header-only implementation sources" OFF)

# Matches the Makefile default (OMP=1). OpenMP is not an optimization here: it
# sizes per-thread resources (streams, memory pointers) for threaded consumers
# such as DBCSR. Without it the library assumes a single thread.
# An explicit request is an error if unavailable, whereas the default degrades
# (AppleClang ships no OpenMP); this matches how the Makefile treats OMP.
# Latched on the first configure: afterwards the cache entry always exists.
if(NOT DEFINED LIBXSTREAM_OMP_EXPLICIT)
  if(DEFINED LIBXSTREAM_OMP)
    set(LIBXSTREAM_OMP_EXPLICIT ON CACHE INTERNAL "LIBXSTREAM_OMP was requested")
  else()
    set(LIBXSTREAM_OMP_EXPLICIT OFF CACHE INTERNAL "LIBXSTREAM_OMP was requested")
  endif()
endif()
option(LIBXSTREAM_OMP "Use OpenMP for per-thread resource sizing" ON)

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
if(LIBXSTREAM_OMP)
  # AppleClang needs a separate libomp. FindOpenMP knows the flag but searches
  # only the implicit link directories, which exclude Homebrew; seed its search
  # with the brew prefix like the Makefile build does. Homebrew keeps libomp
  # keg-only (never symlinked into the main prefix), so the keg path is the one
  # that matters and the plain prefix is only a fallback for other layouts.
  if(APPLE AND NOT OpenMP_libomp_LIBRARY)
    find_program(LIBXSTREAM_BREW NAMES brew)
    if(LIBXSTREAM_BREW)
      execute_process(COMMAND "${LIBXSTREAM_BREW}" --prefix libomp
        OUTPUT_VARIABLE LIBXSTREAM_BREW_LIBOMP
        OUTPUT_STRIP_TRAILING_WHITESPACE
        ERROR_QUIET)
      if(NOT IS_DIRECTORY "${LIBXSTREAM_BREW_LIBOMP}")
        execute_process(COMMAND "${LIBXSTREAM_BREW}" --prefix
          OUTPUT_VARIABLE LIBXSTREAM_BREW_LIBOMP
          OUTPUT_STRIP_TRAILING_WHITESPACE
          ERROR_QUIET)
      endif()
      if(IS_DIRECTORY "${LIBXSTREAM_BREW_LIBOMP}")
        list(APPEND CMAKE_PREFIX_PATH "${LIBXSTREAM_BREW_LIBOMP}")
      endif()
    endif()
  endif()
  find_package(OpenMP COMPONENTS C)
  if(OpenMP_libomp_LIBRARY)
    message(STATUS "Using libomp: ${OpenMP_libomp_LIBRARY}")
  endif()
  if(NOT OpenMP_C_FOUND)
    if(APPLE)
      set(LIBXSTREAM_OMP_HINT
        " AppleClang needs a separate libomp (brew install libomp).")
    endif()
    if(LIBXSTREAM_OMP_EXPLICIT)
      message(FATAL_ERROR
        "LIBXSTREAM_OMP was requested but no OpenMP C support was found."
        "${LIBXSTREAM_OMP_HINT}")
    endif()
    message(WARNING
      "No OpenMP C support found; per-thread resources are sized for a single "
      "thread. Threaded consumers (e.g. DBCSR) require an OpenMP-capable "
      "compiler.${LIBXSTREAM_OMP_HINT}")
    set(LIBXSTREAM_OMP OFF)
  endif()
endif()

set(LIBXSTREAM_LINK_LIBRARIES
  OpenCL::OpenCL
  libxs::libxs)
if(LIBXSTREAM_OMP)
  list(APPEND LIBXSTREAM_LINK_LIBRARIES OpenMP::OpenMP_C)
endif()

message(STATUS "******** Build Summary ********")
message(STATUS "  CMake version         : ${CMAKE_VERSION}")
message(STATUS "  System                : ${CMAKE_SYSTEM_NAME}")
message(STATUS "  libxs target          : libxs::libxs")
message(STATUS "  OpenCL                : ${OpenCL_VERSION_STRING}")
message(STATUS "  OpenMP                : ${OpenMP_C_VERSION}")

set(LIBXSTREAM_ROOT_DIR "${PROJECT_SOURCE_DIR}")
set(LIBXSTREAM_INCLUDE_DIR "${LIBXSTREAM_ROOT_DIR}/libxstream")

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Generate include/libxstream_version.h
set(LIBXSTREAM_VERSION_HEADER "${LIBXSTREAM_INCLUDE_DIR}/libxstream_version.h")
set(LIBXSTREAM_VERSION_SCRIPT "${CMAKE_CURRENT_SOURCE_DIR}/scripts/tool_version.sh")

if(NOT EXISTS "${LIBXSTREAM_VERSION_SCRIPT}")
  message(FATAL_ERROR "Missing ${LIBXSTREAM_VERSION_SCRIPT}")
endif()

find_program(SH_EXECUTABLE NAMES bash sh REQUIRED)

execute_process(
  COMMAND "${SH_EXECUTABLE}" "${LIBXSTREAM_VERSION_SCRIPT}" libxstream -1
  OUTPUT_FILE "${LIBXSTREAM_VERSION_HEADER}"
  RESULT_VARIABLE LIBXSTREAM_VERSION_RC
  WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
)

if(NOT LIBXSTREAM_VERSION_RC EQUAL 0)
  message(FATAL_ERROR "tool_version.sh failed with code ${LIBXSTREAM_VERSION_RC}")
endif()

# Re-run on reconfigure whenever version metadata changes.
set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS
  "${LIBXSTREAM_VERSION_FILE}"
  "${LIBXSTREAM_VERSION_SCRIPT}")

set(LIBXSTREAM_PUBLIC_INCLUDE_DIRS
  $<BUILD_INTERFACE:${LIBXSTREAM_ROOT_DIR}>
  $<BUILD_INTERFACE:${LIBXSTREAM_INCLUDE_DIR}>
  $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
  $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}/libxstream>)

set(LIBXSTREAM_PUBLIC_COMPILE_DEFINITIONS __OPENCL __ACC __LIBXS)

file(GLOB LIBXSTREAM_SRCS LIST_DIRECTORIES false CONFIGURE_DEPENDS
  "${LIBXSTREAM_ROOT_DIR}/src/*.c")

add_library(libxstream ${LIBXSTREAM_SRCS})
add_library(libxstream::libxstream ALIAS libxstream)
set_target_properties(libxstream PROPERTIES
  OUTPUT_NAME xstream
  VERSION ${PROJECT_VERSION}
  SOVERSION ${PROJECT_VERSION_MAJOR}
  POSITION_INDEPENDENT_CODE ON)

target_compile_definitions(libxstream PRIVATE LIBXSTREAM_BUILD=1)
target_compile_definitions(libxstream PUBLIC ${LIBXSTREAM_PUBLIC_COMPILE_DEFINITIONS})

target_include_directories(libxstream
  PUBLIC
    "${LIBXSTREAM_PUBLIC_INCLUDE_DIRS}"
  PRIVATE
    "${LIBXSTREAM_ROOT_DIR}/src")

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
# Consumers linking statically must resolve the OpenMP runtime themselves.
set(PRIVLIBS "")
if(LIBXSTREAM_OMP)
  set(PRIVLIBS "${OpenMP_C_FLAGS}")
endif()
if(BUILD_SHARED_LIBS)
  set(LIBS "-L\${libdir} -lxstream")
else()
  set(LIBS "\${libdir}/libxstream.a")
  string(STRIP "${PRIVLIBS} -lpthread -ldl" PRIVLIBS)
  string(REPLACE "libxs" "libxs-static" REQUIRES_PRIVATE "${REQUIRES_PRIVATE}")
endif()
if(PRIVLIBS)
  set(LIBS_PRIVATE "Libs.private: ${PRIVLIBS}")
else()
  set(LIBS_PRIVATE "")
endif()

configure_file(
  "${CMAKE_CURRENT_SOURCE_DIR}/scripts/libxstream.pc.in"
  "${CMAKE_CURRENT_BINARY_DIR}/libxstream.pc"
  @ONLY
)

configure_package_config_file(
  "${CMAKE_CURRENT_SOURCE_DIR}/scripts/libxstreamConfig.cmake.in"
  "${CMAKE_CURRENT_BINARY_DIR}/libxstreamConfig.cmake"
  INSTALL_DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake/libxstream"
  PATH_VARS CMAKE_INSTALL_INCLUDEDIR CMAKE_INSTALL_LIBDIR CMAKE_INSTALL_DATADIR)

write_basic_package_version_file(
  "${CMAKE_CURRENT_BINARY_DIR}/libxstreamConfigVersion.cmake"
  VERSION ${PROJECT_VERSION}
  COMPATIBILITY SameMajorVersion)

# Installation rules
install(TARGETS libxstream
  EXPORT libxstreamTargets
  ARCHIVE DESTINATION "${CMAKE_INSTALL_LIBDIR}"
  LIBRARY DESTINATION "${CMAKE_INSTALL_LIBDIR}"
  RUNTIME DESTINATION "${CMAKE_INSTALL_BINDIR}")

install(DIRECTORY "${LIBXSTREAM_INCLUDE_DIR}/"
  DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}/libxstream"
  FILES_MATCHING
    PATTERN "*.h"
    PATTERN "libxstream_source.h" EXCLUDE)

# Install the optional header-only implementation.
if(LIBXSTREAM_INSTALL_HEADER_ONLY)
  file(READ "${LIBXSTREAM_INCLUDE_DIR}/libxstream_source.h" LIBXSTREAM_SOURCE_HEADER)
  string(REPLACE "../src/" "src/" LIBXSTREAM_SOURCE_HEADER "${LIBXSTREAM_SOURCE_HEADER}")
  file(WRITE "${CMAKE_CURRENT_BINARY_DIR}/libxstream_source.h" "${LIBXSTREAM_SOURCE_HEADER}")

  install(FILES "${CMAKE_CURRENT_BINARY_DIR}/libxstream_source.h"
    DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}/libxstream")
  install(DIRECTORY "${LIBXSTREAM_ROOT_DIR}/src/"
    DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}/libxstream/src"
    FILES_MATCHING
      PATTERN "*.c"
      PATTERN "*.h")
endif()

# Install pkg-config files
install(FILES
  "${CMAKE_CURRENT_BINARY_DIR}/libxstream.pc"
  DESTINATION "${CMAKE_INSTALL_LIBDIR}/pkgconfig"
)

install(PROGRAMS
  "${CMAKE_CURRENT_SOURCE_DIR}/scripts/tool_opencl.sh"
  DESTINATION "${CMAKE_INSTALL_DATADIR}/libxstream/scripts")

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
  DESTINATION "${CMAKE_INSTALL_DATADIR}/libxstream/samples/smm")

install(DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/samples/smm/kernels/"
  DESTINATION "${CMAKE_INSTALL_DATADIR}/libxstream/samples/smm/kernels"
  FILES_MATCHING PATTERN "*.cl")

install(FILES
  "${CMAKE_CURRENT_SOURCE_DIR}/samples/smm/tune_multiply.py"
  "${CMAKE_CURRENT_SOURCE_DIR}/samples/smm/tune_multiply.sh"
  DESTINATION "${CMAKE_INSTALL_DATADIR}/libxstream/samples/smm")

file(GLOB LIBXSTREAM_SMM_PARAMS LIST_DIRECTORIES false
  "${CMAKE_CURRENT_SOURCE_DIR}/samples/smm/params/*.bin"
  "${CMAKE_CURRENT_SOURCE_DIR}/samples/smm/params/*.csv")
install(FILES ${LIBXSTREAM_SMM_PARAMS}
  DESTINATION "${CMAKE_INSTALL_DATADIR}/libxstream/samples/smm/params")

file(GLOB LIBXSTREAM_SMM_CL LIST_DIRECTORIES false CONFIGURE_DEPENDS
  "${CMAKE_CURRENT_SOURCE_DIR}/samples/smm/kernels/*.cl")
file(GLOB LIBXSTREAM_SMM_CSV LIST_DIRECTORIES false CONFIGURE_DEPENDS
  "${CMAKE_CURRENT_SOURCE_DIR}/samples/smm/params/*.csv")
file(GLOB LIBXSTREAM_SMM_BIN LIST_DIRECTORIES false CONFIGURE_DEPENDS
  "${CMAKE_CURRENT_SOURCE_DIR}/samples/smm/params/*.bin")

set(LIBXSTREAM_SMM_KERNELS_H "${CMAKE_CURRENT_BINARY_DIR}/smm_kernels.h")

add_custom_command(
  OUTPUT "${LIBXSTREAM_SMM_KERNELS_H}"
  COMMAND "${CMAKE_CURRENT_SOURCE_DIR}/scripts/tool_opencl.sh"
    ${LIBXSTREAM_SMM_CL} ${LIBXSTREAM_SMM_CSV} "${LIBXSTREAM_SMM_KERNELS_H}"
  DEPENDS ${LIBXSTREAM_SMM_CL} ${LIBXSTREAM_SMM_CSV} ${LIBXSTREAM_SMM_BIN}
    "${CMAKE_CURRENT_SOURCE_DIR}/scripts/tool_opencl.sh"
  WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/samples/smm/kernels"
  COMMENT "Generating smm_kernels.h from .cl and .csv sources"
  VERBATIM)

add_custom_target(smm_kernels_h ALL DEPENDS "${LIBXSTREAM_SMM_KERNELS_H}")

install(FILES "${LIBXSTREAM_SMM_KERNELS_H}"
  DESTINATION "${CMAKE_INSTALL_DATADIR}/libxstream/samples/smm")

# Stencil sample: generate stencil_kernels.h at build time and install sources
set(LIBXSTREAM_STENCIL_DIR "${CMAKE_CURRENT_SOURCE_DIR}/samples/stencil")
set(LIBXSTREAM_STENCIL_KERNELS_H "${CMAKE_CURRENT_BINARY_DIR}/stencil_kernels.h")

set(LIBXSTREAM_STENCIL_CL
  "${LIBXSTREAM_STENCIL_DIR}/kernels/stencil_bf16.cl"
  "${LIBXSTREAM_STENCIL_DIR}/kernels/stencil_fp32.cl"
  "${LIBXSTREAM_STENCIL_DIR}/kernels/stencil_int8.cl")

file(GLOB LIBXSTREAM_STENCIL_CL_ALL LIST_DIRECTORIES false CONFIGURE_DEPENDS
  "${LIBXSTREAM_STENCIL_DIR}/kernels/*.cl")
file(GLOB LIBXSTREAM_STENCIL_OCL_H LIST_DIRECTORIES false CONFIGURE_DEPENDS
  "${LIBXSTREAM_ROOT_DIR}/libxstream/opencl/*.h")

add_custom_command(
  OUTPUT "${LIBXSTREAM_STENCIL_KERNELS_H}"
  COMMAND "${CMAKE_CURRENT_SOURCE_DIR}/scripts/tool_opencl.sh"
    -p "" ${LIBXSTREAM_STENCIL_CL} "${LIBXSTREAM_STENCIL_KERNELS_H}"
  DEPENDS ${LIBXSTREAM_STENCIL_CL_ALL} ${LIBXSTREAM_STENCIL_OCL_H}
    "${CMAKE_CURRENT_SOURCE_DIR}/scripts/tool_opencl.sh"
  WORKING_DIRECTORY "${LIBXSTREAM_STENCIL_DIR}/kernels"
  COMMENT "Generating stencil_kernels.h from .cl sources"
  VERBATIM)

add_custom_target(stencil_kernels_h ALL DEPENDS "${LIBXSTREAM_STENCIL_KERNELS_H}")

set(LIBXSTREAM_STENCIL_SRC
  "${LIBXSTREAM_STENCIL_DIR}/stencil_opencl.c"
  "${LIBXSTREAM_STENCIL_DIR}/stencil_opencl.h")

install(FILES ${LIBXSTREAM_STENCIL_SRC} "${LIBXSTREAM_STENCIL_KERNELS_H}"
  DESTINATION "${CMAKE_INSTALL_DATADIR}/libxstream/samples/stencil")

install(DIRECTORY "${LIBXSTREAM_STENCIL_DIR}/kernels/"
  DESTINATION "${CMAKE_INSTALL_DATADIR}/libxstream/samples/stencil/kernels"
  FILES_MATCHING PATTERN "*.cl")

install(EXPORT libxstreamTargets
  FILE libxstreamTargets.cmake
  NAMESPACE libxstream::
  DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake/libxstream")

install(FILES
  "${CMAKE_CURRENT_BINARY_DIR}/libxstreamConfig.cmake"
  "${CMAKE_CURRENT_BINARY_DIR}/libxstreamConfigVersion.cmake"
  DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake/libxstream")

# Install documentation.
install(
  DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/documentation/"
  DESTINATION "${CMAKE_INSTALL_DOCDIR}"
  FILES_MATCHING
    PATTERN "*.md"
    PATTERN "*.pdf"
    PATTERN "*.css"
    PATTERN "*.png"
    PATTERN "tests" EXCLUDE
    PATTERN "libxstream_present.md" EXCLUDE
    PATTERN "libxstream_samples.md" EXCLUDE
    PATTERN "libxstream_scripts.md" EXCLUDE
)
