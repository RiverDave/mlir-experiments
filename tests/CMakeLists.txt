set(MLIR_TEST_DEPENDS mlir-opt FileCheck mlir-runner) # Add other tools if needed
set(PYTHON_EXECUTABLE python3)

add_custom_target(check-mlir
  COMMAND ${PYTHON_EXECUTABLE} -m lit -v ${CMAKE_CURRENT_SOURCE_DIR}
  DEPENDS ${MLIR_TEST_DEPENDS}
  WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
  COMMENT "Running MLIR lit tests"
)
