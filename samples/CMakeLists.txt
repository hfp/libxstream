find_package(BLAS REQUIRED)

function(libxstream_sample target directory output_name)
  add_executable(${target} ${ARGN})
  target_link_libraries(${target} PRIVATE libxstream)
  target_include_directories(${target} PRIVATE "${PROJECT_BINARY_DIR}")
  set_target_properties(
    ${target}
    PROPERTIES RUNTIME_OUTPUT_DIRECTORY
               "${CMAKE_CURRENT_BINARY_DIR}/${directory}"
               OUTPUT_NAME "${output_name}"
               SUFFIX ".x")
endfunction()

# Reuse the kernel headers already generated for installation.
set(smm_sources smm/smm_acc.c smm/smm_kernel.c smm/smm_params.c smm/smm_trans.c)
libxstream_sample(libxstream_sample_smm smm acc_bench smm/acc_bench.c
                  ${smm_sources})
add_dependencies(libxstream_sample_smm smm_kernels_h)

libxstream_sample(libxstream_sample_dbm dbm dbm_bench smm/acc_bench.c
                  ${smm_sources} dbm/dbm_opencl.c)
add_dependencies(libxstream_sample_dbm smm_kernels_h dbm_kernels_h)

set(ozaki_dir "${CMAKE_CURRENT_SOURCE_DIR}/ozaki")
set(ozaki_header "${CMAKE_CURRENT_BINARY_DIR}/ozaki_kernels.h")
file(GLOB ozaki_kernels CONFIGURE_DEPENDS "${ozaki_dir}/kernels/*.cl")
add_custom_command(
  OUTPUT "${ozaki_header}"
  COMMAND
    "${PROJECT_SOURCE_DIR}/scripts/tool_opencl.sh" -p ""
    "${ozaki_dir}/kernels/ozaki1.cl" "${ozaki_dir}/kernels/ozaki2.cl"
    "${ozaki_dir}/kernels/gemm3m.cl" "${ozaki_header}"
  DEPENDS ${ozaki_kernels} ${LIBXSTREAM_STENCIL_OCL_H}
          "${PROJECT_SOURCE_DIR}/scripts/tool_opencl.sh"
  COMMENT "Generating ozaki_kernels.h from .cl sources"
  VERBATIM)
libxstream_sample(
  libxstream_sample_ozaki
  ozaki
  ozaki
  ozaki/ozaki_bench.c
  ozaki/ozaki_opencl.c
  ozaki/ozaki_gemm.c
  ozaki/ozaki_gemm3m.c
  "${ozaki_header}")
target_include_directories(libxstream_sample_ozaki
                           PRIVATE "${CMAKE_CURRENT_BINARY_DIR}")

libxstream_sample(
  libxstream_sample_stencil
  stencil
  stencil
  stencil/stencil.c
  stencil/stencil_opencl.c
  stencil/stencil_cpu.c
  stencil/stencil_pack.c
  stencil/stencil_pml.c
  stencil/stencil_weights.c)
add_dependencies(libxstream_sample_stencil stencil_kernels_h)

foreach(sample smm dbm ozaki)
  target_compile_definitions(libxstream_sample_${sample} PRIVATE __BLAS)
  target_link_libraries(libxstream_sample_${sample} PRIVATE ${BLAS_LIBRARIES})
endforeach()

libxstream_sample(libxstream_sample_match smm acc_match smm/acc_match.c)
add_dependencies(libxstream_sample_match smm_kernels_h)
if(BUILD_TESTING)
  add_test(
    NAME libxstream-match
    COMMAND
      "${CMAKE_COMMAND}" -E env
      "LIBXSTREAM_ACC_MATCH=$<TARGET_FILE:libxstream_sample_match>"
      "${LIBXSTREAM_TEST_SHELL}" "${PROJECT_SOURCE_DIR}/tests/match.sh")
  set_tests_properties(libxstream-match PROPERTIES LABELS libxstream)
endif()
