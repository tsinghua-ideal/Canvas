# Sub-directory of the C++ library
add_subdirectory(Algorithms)
add_subdirectory(Core)
add_subdirectory(CodeGen)
add_subdirectory(Kernels)
add_subdirectory(Primitives)
add_subdirectory(Search)
add_subdirectory(Utils)

# Link flags
set_property(GLOBAL PROPERTY CANVAS_ALL_LIBS_PROPERTY
        SearchLib CodeGenLib KernelsLib AlgorithmsLib CoreLib PrimitiveLib UtilsLib ${Boost_LIBRARIES} -lstdc++ -lm)
get_property(CANVAS_ALL_LIBS GLOBAL PROPERTY CANVAS_ALL_LIBS_PROPERTY)

# Add PyBind API
pybind11_add_module(cpp_canvas pybind_api.cpp)
target_link_libraries(cpp_canvas PRIVATE ${CANVAS_ALL_LIBS})
