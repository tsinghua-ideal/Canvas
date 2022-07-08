# Sub-directory of the C++ library
add_subdirectory(CodeGen)
add_subdirectory(Core)
add_subdirectory(Impls)
add_subdirectory(Passes)
add_subdirectory(Primitives)
add_subdirectory(Search)
add_subdirectory(Utils)

# Link flags
set_property(GLOBAL PROPERTY CANVAS_3RD_PARTY_LIBS_PROPERTY
        ${Boost_LIBRARIES} -lstdc++ -lm)
get_property(CANVAS_3RD_PARTY_LIBS GLOBAL PROPERTY CANVAS_3RD_PARTY_LIBS_PROPERTY)

set_property(GLOBAL PROPERTY CANVAS_LIBS_PROPERTY
        SearchLib CodeGenLib PassLib ImplLib CoreLib PrimitiveLib UtilsLib)
get_property(CANVAS_LIBS GLOBAL PROPERTY CANVAS_LIBS_PROPERTY)

# Packed executables/libraries
add_subdirectory(PyBind)
