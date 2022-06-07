# Function for adding a test.
function(add_canvas_test TEST_NAME TEST_ENTRY_PATH)
    add_executable(${TEST_NAME} ${TEST_ENTRY_PATH})
    get_property(CANVAS_ALL_LIBS GLOBAL PROPERTY CANVAS_ALL_LIBS_PROPERTY)
    target_link_libraries(${TEST_NAME} gtest gtest_main pthread ${CANVAS_ALL_LIBS})
    add_test(${TEST_NAME} ${TEST_NAME})
endfunction()

# Sub-directories.
add_subdirectory(cpp)
