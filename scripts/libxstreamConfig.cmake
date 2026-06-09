if(TARGET libxstream::libxstream)
  return()
endif()

get_filename_component(_prefix "${CMAKE_CURRENT_LIST_DIR}/../../.." ABSOLUTE)

find_library(LIBXSTREAM_LIBRARY NAMES xstream HINTS "${_prefix}/lib" NO_DEFAULT_PATH)
find_path(LIBXSTREAM_INCLUDE_DIR NAMES libxstream/libxstream.h HINTS "${_prefix}/include" "${_prefix}" NO_DEFAULT_PATH)

if(LIBXSTREAM_LIBRARY AND LIBXSTREAM_INCLUDE_DIR)
  find_package(libxs CONFIG QUIET
    HINTS "${_prefix}")
  find_package(OpenCL QUIET)
  find_package(OpenMP QUIET COMPONENTS C)
  add_library(libxstream::libxstream UNKNOWN IMPORTED)
  set_target_properties(libxstream::libxstream PROPERTIES
    IMPORTED_LOCATION "${LIBXSTREAM_LIBRARY}"
    INTERFACE_INCLUDE_DIRECTORIES "${LIBXSTREAM_INCLUDE_DIR}")
  if(TARGET libxs::libxs)
    set_property(TARGET libxstream::libxstream APPEND PROPERTY
      INTERFACE_LINK_LIBRARIES libxs::libxs)
  endif()
  if(TARGET OpenCL::OpenCL)
    set_property(TARGET libxstream::libxstream APPEND PROPERTY
      INTERFACE_LINK_LIBRARIES OpenCL::OpenCL)
  endif()
  if(TARGET OpenMP::OpenMP_C)
    set_property(TARGET libxstream::libxstream APPEND PROPERTY
      INTERFACE_LINK_LIBRARIES OpenMP::OpenMP_C)
  endif()
  set(LIBXSTREAM_INCLUDE_DIRS "${LIBXSTREAM_INCLUDE_DIR}")
  if(EXISTS "${_prefix}/share/libxstream/samples/smm/smm_acc.c")
    set(LIBXSTREAM_OPENCL_SCRIPT "${_prefix}/share/libxstream/scripts/tool_opencl.sh")
    set(LIBXSTREAM_SMM_DIR "${_prefix}/share/libxstream/samples/smm")
  else()
    set(LIBXSTREAM_OPENCL_SCRIPT "${_prefix}/scripts/tool_opencl.sh")
    set(LIBXSTREAM_SMM_DIR "${_prefix}/samples/smm")
  endif()
  set(LIBXSTREAM_SMM_ACC_SOURCE "${LIBXSTREAM_SMM_DIR}/smm_acc.c")
endif()

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(libxstream DEFAULT_MSG LIBXSTREAM_LIBRARY LIBXSTREAM_INCLUDE_DIR)
unset(_prefix)
