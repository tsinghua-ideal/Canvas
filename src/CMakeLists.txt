# Sub-directory of the C++ library
add_subdirectory(Core)
add_subdirectory(CodeGen)
add_subdirectory(Primitives)
add_subdirectory(Search)
add_subdirectory(Utils)

# Link flags
set_property(GLOBAL PROPERTY CANVAS_ALL_LIBS_PROPERTY
        SearchLib CodeGenLib CoreLib PrimitiveLib UtilsLib ${Boost_LIBRARIES} -lstdc++ -lm)
get_property(CANVAS_ALL_LIBS GLOBAL PROPERTY CANVAS_ALL_LIBS_PROPERTY)

# Packed executables/libraries
add_subdirectory(PyBind)
