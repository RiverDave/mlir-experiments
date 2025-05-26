set(LLVM_TARGET_DEFINITIONS Passes.td)
mlir_tablegen(Passes.h.inc --gen-pass-decls -name Arith)
#mlir_tablegen(Passes AffinePassses ./ -gen-pass-doc)

# Create the TableGen target
add_public_tablegen_target(MulToAddPasses)

add_mlir_library(MulToAdd
    MulToAdd.cpp
    ${PROJECT_SOURCE_DIR}/lib/Transform/Arith/
    ADDITIONAL_HEADER_DIRS

    DEPENDS
    MulToAddPasses  # Match the target name above

    LINK_LIBS PUBLIC
)

# Ensure the build directory is included
target_include_directories(MulToAdd PRIVATE ${CMAKE_CURRENT_BINARY_DIR})