add_library(CodeGenLib STATIC CodeGen.cpp
        DotCodeGen.cpp IRCodeGen.cpp PyTorchCodeGen.cpp TVMCodeGen.cpp VarsCodeGen.cpp)
