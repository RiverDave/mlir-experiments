set(LLVM_TARGET_DEFINITIONS PolyDialect.td)
mlir_tablegen(PolyDialect.h.inc -gen-dialect-decls -dialect=poly)
mlir_tablegen(PolyDialect.cpp.inc -gen-dialect-defs -dialect=poly)
add_public_tablegen_target(MLIRPolyDialectIncGen)

# Define TableGen source for types
set(LLVM_TARGET_DEFINITIONS PolyTypes.td)
mlir_tablegen(PolyTypes.h.inc -gen-typedef-decls -typedefs-dialect=poly)
mlir_tablegen(PolyTypes.cpp.inc -gen-typedef-defs -typedefs-dialect=poly)
add_public_tablegen_target(MLIRPolyTypesIncGen)

# Create the dialect library
add_mlir_dialect_library(Poly
    PolyDialect.cpp

    ADDITIONAL_HEADER_DIRS
    ${PROJECT_SOURCE_DIR}/lib/Dialect/Poly

    DEPENDS
    MLIRPolyDialectIncGen
    MLIRPolyTypesIncGen

    LINK_LIBS PUBLIC
    MLIRIR
    MLIRSupport
)