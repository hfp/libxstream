# Builds the OpenCL machinery of a sample into a consumer's own target:
#
#   libxstream_add_dbm(<target>)  DBM backend (samples/dbm), e.g., CP2K
#   libxstream_add_smm(<target>)  LIBSMM (samples/smm), e.g., DBCSR
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
  file(GLOB common LIST_DIRECTORIES false CONFIGURE_DEPENDS "${incdir}/opencl/*.h")
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
  file(GLOB kernels LIST_DIRECTORIES false CONFIGURE_DEPENDS
    "${LIBXSTREAM_DBM_DIR}/kernels/*.cl")
  # the sample's own parameter directory (none yet); an empty -p "" does not survive CMake
  libxstream_add_kernels_(${TARGET} "${LIBXSTREAM_DBM_DIR}" dbm_kernels.h
    OPTIONS -p "${LIBXSTREAM_DBM_DIR}/params" INPUTS ${kernels})
  target_sources(${TARGET} PRIVATE "${LIBXSTREAM_DBM_DIR}/dbm_opencl.c")
endfunction()

function(libxstream_add_smm TARGET)
  file(GLOB kernels LIST_DIRECTORIES false CONFIGURE_DEPENDS
    "${LIBXSTREAM_SMM_DIR}/kernels/*.cl")
  file(GLOB params LIST_DIRECTORIES false CONFIGURE_DEPENDS
    "${LIBXSTREAM_SMM_DIR}/params/*.csv")
  file(GLOB models LIST_DIRECTORIES false CONFIGURE_DEPENDS
    "${LIBXSTREAM_SMM_DIR}/params/*.bin")
  # a prediction model (.bin) is embedded along with the CSV file of its name
  libxstream_add_kernels_(${TARGET} "${LIBXSTREAM_SMM_DIR}" smm_kernels.h
    INPUTS ${kernels} ${params} DEPENDS ${models})
  target_sources(${TARGET} PRIVATE
    "${LIBXSTREAM_SMM_DIR}/smm_acc.c" "${LIBXSTREAM_SMM_DIR}/smm_kernel.c"
    "${LIBXSTREAM_SMM_DIR}/smm_params.c" "${LIBXSTREAM_SMM_DIR}/smm_trans.c")
endfunction()
