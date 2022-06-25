add_canvas_test(core_tests Core.cpp "CoreLib")
add_canvas_test(primitive_tests Primitives.cpp "CoreLib;PrimitiveLib;UtilsLib")
add_canvas_test(search_tests Search.cpp "SearchLib;CodeGenLib;CoreLib;PrimitiveLib;UtilsLib")
add_canvas_test(utils_tests Utils.cpp "UtilsLib")
