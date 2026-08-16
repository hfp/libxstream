# Native CTest coverage for the GNU Make LIBXSTREAM test suite.
#
# The test runner itself is the existing tests/test.sh script, but all
# executables it runs are built by CMake.  This keeps the CTest path
# independent from the Makefile path while preserving the same test list and
# checks used by `make test`.

find_program(LIBXSTREAM_TEST_SHELL NAMES bash sh REQUIRED)

set(LIBXSTREAM_TEST_DIR "${PROJECT_BINARY_DIR}/tests")
file(MAKE_DIRECTORY "${LIBXSTREAM_TEST_DIR}")
file(COPY "${CMAKE_CURRENT_SOURCE_DIR}/test.sh"
  DESTINATION "${LIBXSTREAM_TEST_DIR}")

# test.sh discovers its tests as the *.c files defining a main function, hence
# the same discovery here: a new test needs no entry in this file.
file(GLOB LIBXSTREAM_TEST_SOURCES CONFIGURE_DEPENDS
  "${CMAKE_CURRENT_SOURCE_DIR}/*.c")

foreach(test_source IN LISTS LIBXSTREAM_TEST_SOURCES)
  file(STRINGS "${test_source}" LIBXSTREAM_TEST_MAIN REGEX "main[ \t]*\\(")
  if(LIBXSTREAM_TEST_MAIN)
    get_filename_component(test_name "${test_source}" NAME_WE)
    set(test_target "libxstream_test_${test_name}")
    add_executable(${test_target} "${test_source}")
    # libxstream links libxs publicly, i.e. tests including <libxs/...> as well
    target_link_libraries(${test_target} PRIVATE libxstream)
    set_target_properties(${test_target} PROPERTIES
      RUNTIME_OUTPUT_DIRECTORY "${LIBXSTREAM_TEST_DIR}"
      OUTPUT_NAME "${test_name}"
      SUFFIX ".x")
    add_test(NAME "libxstream-${test_name}"
      COMMAND "${LIBXSTREAM_TEST_SHELL}"
        "${LIBXSTREAM_TEST_DIR}/test.sh" "${test_name}.c")
    set_tests_properties("libxstream-${test_name}" PROPERTIES
      WORKING_DIRECTORY "${LIBXSTREAM_TEST_DIR}"
      LABELS libxstream)
  endif()
endforeach()
