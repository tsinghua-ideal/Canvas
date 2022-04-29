# Add PyBind API
pybind11_add_module(cpp_canvas pybind_api.cpp)
target_link_libraries(cpp_canvas PRIVATE ${CANVAS_ALL_LIBS})
