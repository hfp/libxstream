# Builds the OpenCL machinery of a sample into a consumer's own target:
#
# libxstream_add_dbm(<target>)  DBM backend (samples/dbm), e.g., CP2K
# libxstream_add_grid(<target> <grid>)  GRID backend (samples/grid), e.g., CP2K
# libxstream_add_smm(<target>)  LIBSMM (samples/smm), e.g., DBCSR
#
# Sources, kernels, and parameter files are enumerated here rather than by the
# consumer, hence renaming them is private to LIBXSTREAM. The embedded kernel
# header is generated into the target's build tree, whose directory precedes the
# sample's own on the include path: a header installed next to the sources is
# not picked up instead. The consumer links libxstream::libxstream itself.

function(libxstream_add_kernels_ TARGET DIR HEADER)
  cmake_parse_arguments(ARG "" "" "OPTIONS;INPUTS;DEPENDS" ${ARGN})
  list(GET LIBXSTREAM_INCLUDE_DIRS 0 incdir)
  if(EXISTS "${incdir}/libxstream/opencl")
    set(incdir "${incdir}/libxstream")
  endif()
  set(gendir "${CMAKE_CURRENT_BINARY_DIR}/${TARGET}_libxstream")
  file(
    GLOB common
    LIST_DIRECTORIES false
    CONFIGURE_DEPENDS "${incdir}/opencl/*.h")
  add_custom_command(
    OUTPUT "${gendir}/${HEADER}"
    COMMAND "${CMAKE_COMMAND}" -E make_directory "${gendir}"
    COMMAND "${LIBXSTREAM_OPENCL_SCRIPT}" ${ARG_OPTIONS} -I "${incdir}"
            ${ARG_INPUTS} "${gendir}/${HEADER}"
    DEPENDS ${ARG_INPUTS} ${ARG_DEPENDS} ${common} "${LIBXSTREAM_OPENCL_SCRIPT}"
    COMMENT "Generating ${HEADER} for ${TARGET}"
    VERBATIM)
  target_sources(${TARGET} PRIVATE "${gendir}/${HEADER}")
  target_include_directories(${TARGET} BEFORE PRIVATE "${gendir}" "${DIR}")
endfunction()

function(libxstream_add_dbm TARGET)
  file(
    GLOB kernels
    LIST_DIRECTORIES false
    CONFIGURE_DEPENDS "${LIBXSTREAM_DBM_DIR}/kernels/*.cl")
  # the sample's own parameter directory (none yet); an empty -p "" does not
  # survive CMake
  libxstream_add_kernels_(
    ${TARGET}
    "${LIBXSTREAM_DBM_DIR}"
    dbm_kernels.h
    OPTIONS
    -p
    "${LIBXSTREAM_DBM_DIR}/params"
    INPUTS
    ${kernels})
  target_sources(${TARGET} PRIVATE "${LIBXSTREAM_DBM_DIR}/dbm_opencl.c")
endfunction()

# the kernels inline the consumer's grid/common headers (-k keeps include guards
# since headers repeat), hence <grid> is the consumer's grid directory
function(libxstream_add_grid TARGET GRID)
  file(
    GLOB kernels
    LIST_DIRECTORIES false
    CONFIGURE_DEPENDS "${LIBXSTREAM_GRID_DIR}/kernels/*.cl")
  file(
    GLOB headers
    LIST_DIRECTORIES false
    CONFIGURE_DEPENDS "${LIBXSTREAM_GRID_DIR}/kernels/*.h"
                      "${GRID}/common/*.h")
  libxstream_add_kernels_(
    ${TARGET}
    "${LIBXSTREAM_GRID_DIR}"
    grid_kernels.h
    OPTIONS
    -k
    -I
    "${GRID}/common"
    INPUTS
    ${kernels}
    DEPENDS
    "${LIBXSTREAM_GRID_DIR}/grid_opencl.h"
    ${headers})
  target_sources(${TARGET} PRIVATE "${LIBXSTREAM_GRID_DIR}/grid_opencl.c")
endfunction()

function(libxstream_add_smm TARGET)
  file(
    GLOB kernels
    LIST_DIRECTORIES false
    CONFIGURE_DEPENDS "${LIBXSTREAM_SMM_DIR}/kernels/*.cl")
  file(
    GLOB params
    LIST_DIRECTORIES false
    CONFIGURE_DEPENDS "${LIBXSTREAM_SMM_DIR}/params/*.csv")
  file(
    GLOB models
    LIST_DIRECTORIES false
    CONFIGURE_DEPENDS "${LIBXSTREAM_SMM_DIR}/params/*.bin")
  # a prediction model (.bin) is embedded along with the CSV file of its name
  libxstream_add_kernels_(
    ${TARGET}
    "${LIBXSTREAM_SMM_DIR}"
    smm_kernels.h
    INPUTS
    ${kernels}
    ${params}
    DEPENDS
    ${models})
  target_sources(
    ${TARGET}
    PRIVATE "${LIBXSTREAM_SMM_DIR}/smm_acc.c"
            "${LIBXSTREAM_SMM_DIR}/smm_kernel.c"
            "${LIBXSTREAM_SMM_DIR}/smm_params.c"
            "${LIBXSTREAM_SMM_DIR}/smm_trans.c")
  libxstream_add_smm_sycl_(${TARGET})
endfunction()

# oneMKL's SYCL interface takes stacks too large for the SMM kernels, compiled by
# the C++ compiler at hand (no SYCL compiler): on if the consumer uses MKL and the
# SYCL runtime is in reach (CMPLR_ROOT, icx, or next to MKL), unless
# LIBXSTREAM_SMM_SYCL is OFF. The runtime loads its OpenCL backend by name, hence
# the oneAPI libraries must be on the library path when the program runs.
function(libxstream_add_smm_sycl_ TARGET)
  if(DEFINED LIBXSTREAM_SMM_SYCL AND NOT LIBXSTREAM_SMM_SYCL)
    return()
  endif()
  get_property(languages GLOBAL PROPERTY ENABLED_LANGUAGES)
  get_target_property(definitions ${TARGET} COMPILE_DEFINITIONS)
  set(blas "${BLAS_LIBRARIES};${LAPACK_LIBRARIES}")
  if(NOT "CXX" IN_LIST languages OR NOT ("${blas}" MATCHES "mkl_"
                                         OR "${definitions}" MATCHES "__MKL"))
    return()
  endif()
  set(mkldirs "$ENV{MKLROOT}/lib/intel64" "$ENV{MKLROOT}/lib")
  foreach(lib IN LISTS blas)
    if(lib MATCHES "mkl_" AND EXISTS "${lib}")
      get_filename_component(dir "${lib}" DIRECTORY)
      list(INSERT mkldirs 0 "${dir}")
    endif()
  endforeach()
  find_library(
    LIBXSTREAM_MKL_SYCL_BLAS
    NAMES mkl_sycl_blas
    HINTS ${mkldirs})
  find_program(LIBXSTREAM_ICX icx)
  set(roots "$ENV{CMPLR_ROOT}")
  if(LIBXSTREAM_ICX)
    get_filename_component(dir "${LIBXSTREAM_ICX}" DIRECTORY)
    list(APPEND roots "${dir}/..")
  endif()
  if(LIBXSTREAM_MKL_SYCL_BLAS)
    # lib or lib/intel64 of mkl/<version>, next to compiler/latest
    get_filename_component(dir "${LIBXSTREAM_MKL_SYCL_BLAS}" DIRECTORY)
    list(APPEND roots "${dir}/../../compiler/latest" "${dir}/../../../compiler/latest")
  endif()
  find_path(
    LIBXSTREAM_SYCL_INCLUDE_DIR
    NAMES sycl/sycl.hpp
    HINTS ${roots}
    PATH_SUFFIXES include
    NO_DEFAULT_PATH)
  find_library(
    LIBXSTREAM_SYCL_LIBRARY
    NAMES sycl
    HINTS ${roots}
    PATH_SUFFIXES lib
    NO_DEFAULT_PATH)
  if(NOT LIBXSTREAM_MKL_SYCL_BLAS
     OR NOT LIBXSTREAM_SYCL_INCLUDE_DIR
     OR NOT LIBXSTREAM_SYCL_LIBRARY)
    return()
  endif()
  set(libraries "${LIBXSTREAM_MKL_SYCL_BLAS}" "${LIBXSTREAM_SYCL_LIBRARY}")
  # MKL's GNU Fortran interface lacks the *_64 CBLAS names oneMKL calls: the Intel
  # interface follows the consumer's, hence supplies only what nothing else defines
  if("${blas}" MATCHES "mkl_gf_(i?lp64)")
    set(bits "${CMAKE_MATCH_1}") # before the next MATCHES replaces it
    if("${blas}" MATCHES "mkl_[^;]*\\.a(;|$)")
      message(STATUS "LIBXSTREAM: oneMKL (SYCL) needs MKL's shared libraries")
      return()
    endif()
    find_library(
      LIBXSTREAM_MKL_INTEL_IFACE
      NAMES mkl_intel_${bits}
      HINTS ${mkldirs})
    if(NOT LIBXSTREAM_MKL_INTEL_IFACE)
      return()
    endif()
    list(APPEND libraries "${LIBXSTREAM_MKL_INTEL_IFACE}")
  endif()
  # the consumer's MKL again: a linker dropping libraries nothing has asked for yet
  # (--as-needed) would otherwise leave the above unresolved
  foreach(lib IN LISTS blas)
    if(lib MATCHES "mkl_(intel_i?lp64|core|sequential|[a-z]+_thread)")
      list(APPEND libraries "${lib}")
    endif()
  endforeach()
  # the SYCL headers as system headers of this file only: they bring CL headers
  set(source "${LIBXSTREAM_SMM_DIR}/smm_sycl.cpp")
  target_sources(${TARGET} PRIVATE "${source}")
  set_source_files_properties(
    "${source}" PROPERTIES COMPILE_OPTIONS
                           "-isystem;${LIBXSTREAM_SYCL_INCLUDE_DIR};-std=c++17")
  target_compile_definitions(${TARGET} PRIVATE OPENCL_LIBSMM_SYCL)
  target_link_libraries(${TARGET} PRIVATE ${libraries})
  message(STATUS "LIBXSTREAM: oneMKL (SYCL) for SMMs too large for the kernels")
endfunction()
