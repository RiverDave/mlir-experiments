set(LLVM_TARGET_DEFINITIONS Passes.td)
mlir_tablegen(Passes.h.inc --gen-pass-decls)
#mlir_tablegen(Passes AffinePassses ./ -gen-pass-doc)

# Create the TableGen target
add_public_tablegen_target(AffineFullUnrollPasses)

add_mlir_library(AffineFullUnroll
    AffineFullUnroll.cpp
    AffineFullUnrollPatternRewrite.cpp
    ${PROJECT_SOURCE_DIR}/lib/Transform/Affine/
    ADDITIONAL_HEADER_DIRS

    DEPENDS
    AffineFullUnrollPasses  # Match the target name above

    LINK_LIBS PUBLIC
)

# Ensure the build directory is included
target_include_directories(AffineFullUnroll PRIVATE ${CMAKE_CURRENT_BINARY_DIR})