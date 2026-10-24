# lib/cmake/libxstream or lib/<target>/cmake/libxstream
get_filename_component(_libxstream_libdir "${CMAKE_CURRENT_LIST_DIR}/../.."
                       ABSOLUTE)
get_filename_component(_libxstream_prefix "${_libxstream_libdir}/.." ABSOLUTE)
if(NOT EXISTS "${_libxstream_prefix}/include/libxstream/libxstream.h"
   AND NOT EXISTS "${_libxstream_prefix}/libxstream/libxstream.h")
  get_filename_component(_libxstream_prefix "${_libxstream_libdir}/../.."
                         ABSOLUTE)
endif()

# Substituted by the Makefile to match the OMP setting of this build; the
# unsubstituted default matches the Makefile default (OMP=1).
set(LIBXSTREAM_OMP @LIBXSTREAM_OMP@)
if(NOT LIBXSTREAM_OMP MATCHES "^(ON|OFF)$")
  set(LIBXSTREAM_OMP ON)
endif()

set(_libxstream_suffixes_save "${CMAKE_FIND_LIBRARY_SUFFIXES}")
if(DEFINED BUILD_SHARED_LIBS AND NOT BUILD_SHARED_LIBS)
  set(CMAKE_FIND_LIBRARY_SUFFIXES .a)
endif()
find_library(
  LIBXSTREAM_LIBRARY
  NAMES xstream
  HINTS "${_libxstream_libdir}"
  NO_DEFAULT_PATH)
if(NOT LIBXSTREAM_LIBRARY)
  set(CMAKE_FIND_LIBRARY_SUFFIXES "${_libxstream_suffixes_save}")
  find_library(
    LIBXSTREAM_LIBRARY
    NAMES xstream
    HINTS "${_libxstream_libdir}"
    NO_DEFAULT_PATH)
endif()
set(CMAKE_FIND_LIBRARY_SUFFIXES "${_libxstream_suffixes_save}")
unset(_libxstream_suffixes_save)
# pin "latest" (lib/ symlinks) to the folder of the actual build (lib/<target>)
if(LIBXSTREAM_LIBRARY)
  get_filename_component(_libxstream_real "${LIBXSTREAM_LIBRARY}" REALPATH)
  get_filename_component(_libxstream_real "${_libxstream_real}" DIRECTORY)
  get_filename_component(_libxstream_name "${LIBXSTREAM_LIBRARY}" NAME)
  if(EXISTS "${_libxstream_real}/${_libxstream_name}")
    set(_libxstream_libdir "${_libxstream_real}")
    set(LIBXSTREAM_LIBRARY
        "${_libxstream_real}/${_libxstream_name}"
        CACHE FILEPATH "Path to a library." FORCE)
  endif()
  unset(_libxstream_name)
  unset(_libxstream_real)
endif()
find_path(
  LIBXSTREAM_INCLUDE_DIR
  NAMES libxstream/libxstream.h
  HINTS "${_libxstream_prefix}/include" "${_libxstream_prefix}"
  NO_DEFAULT_PATH)

if(LIBXSTREAM_LIBRARY AND LIBXSTREAM_INCLUDE_DIR)
  if(NOT TARGET libxstream::libxstream)
    find_package(libxs CONFIG QUIET HINTS "${_libxstream_prefix}")
    find_package(OpenCL QUIET)
    if(LIBXSTREAM_OMP)
      find_package(OpenMP REQUIRED COMPONENTS C)
    endif()
    add_library(libxstream::libxstream UNKNOWN IMPORTED)
    # the same PUBLIC definitions as the target a CMake build exports
    set_target_properties(
      libxstream::libxstream
      PROPERTIES IMPORTED_LOCATION "${LIBXSTREAM_LIBRARY}"
                 INTERFACE_INCLUDE_DIRECTORIES "${LIBXSTREAM_INCLUDE_DIR}"
                 INTERFACE_COMPILE_DEFINITIONS "__OPENCL;__ACC;__LIBXS")
    if(TARGET libxs::libxs)
      set_property(
        TARGET libxstream::libxstream
        APPEND
        PROPERTY INTERFACE_LINK_LIBRARIES libxs::libxs)
    endif()
    if(TARGET OpenCL::OpenCL)
      set_property(
        TARGET libxstream::libxstream
        APPEND
        PROPERTY INTERFACE_LINK_LIBRARIES OpenCL::OpenCL)
    endif()
    if(TARGET OpenMP::OpenMP_C)
      set_property(
        TARGET libxstream::libxstream
        APPEND
        PROPERTY INTERFACE_LINK_LIBRARIES OpenMP::OpenMP_C)
    endif()
  else()
    set_property(TARGET libxstream::libxstream PROPERTY IMPORTED_LOCATION
                                                        "${LIBXSTREAM_LIBRARY}")
  endif()
endif()

if(EXISTS "${_libxstream_prefix}/include/libxstream/opencl")
  set(LIBXSTREAM_INCLUDE_DIRS "${_libxstream_prefix}/include/libxstream")
elseif(EXISTS "${_libxstream_prefix}/libxstream/opencl")
  set(LIBXSTREAM_INCLUDE_DIRS "${_libxstream_prefix}/libxstream")
else()
  set(LIBXSTREAM_INCLUDE_DIRS "${_libxstream_prefix}/include")
endif()
if(EXISTS "${_libxstream_prefix}/share/libxstream/samples/smm/smm_acc.c")
  set(LIBXSTREAM_OPENCL_SCRIPT
      "${_libxstream_prefix}/share/libxstream/scripts/tool_opencl.sh")
  set(LIBXSTREAM_SMM_DIR "${_libxstream_prefix}/share/libxstream/samples/smm")
  set(LIBXSTREAM_DBM_DIR "${_libxstream_prefix}/share/libxstream/samples/dbm")
else()
  set(LIBXSTREAM_OPENCL_SCRIPT "${_libxstream_prefix}/scripts/tool_opencl.sh")
  set(LIBXSTREAM_SMM_DIR "${_libxstream_prefix}/samples/smm")
  set(LIBXSTREAM_DBM_DIR "${_libxstream_prefix}/samples/dbm")
endif()
set(LIBXSTREAM_SMM_ACC_SOURCE "${LIBXSTREAM_SMM_DIR}/smm_acc.c")

if(NOT LIBXSTREAM_LIBRARY)
  find_library(
    LIBXSTREAM_LIBRARY
    NAMES xstream
    HINTS "${_libxstream_libdir}"
    NO_DEFAULT_PATH)
endif()
include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(libxstream DEFAULT_MSG LIBXSTREAM_LIBRARY)
unset(_libxstream_prefix)
unset(_libxstream_libdir)
include("${CMAKE_CURRENT_LIST_DIR}/libxstreamSamples.cmake")
