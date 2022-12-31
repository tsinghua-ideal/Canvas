# Add PyBind API
pybind11_add_module(cpp_canvas PyBindAPI.cpp)
target_link_libraries(cpp_canvas PRIVATE ${CANVAS_LIBS} ${CANVAS_3RD_PARTY_LIBS})
