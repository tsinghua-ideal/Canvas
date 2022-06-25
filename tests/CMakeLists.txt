# Function for adding a test.
function(add_canvas_test TEST_NAME TEST_ENTRY_PATH TEST_DEPENDENCIES)
    add_executable(${TEST_NAME} ${TEST_ENTRY_PATH})
    get_property(CANVAS_ALL_LIBS GLOBAL PROPERTY CANVAS_ALL_LIBS_PROPERTY)
    target_link_libraries(${TEST_NAME} ${TEST_DEPENDENCIES} ${CANVAS_3RD_PARTY_LIBS} ${GTEST_BOTH_LIBRARIES} pthread)
    add_test(${TEST_NAME} ${TEST_NAME})
endfunction()

# Sub-directories.
add_subdirectory(cpp)
