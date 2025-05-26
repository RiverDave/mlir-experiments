#include <lib/Transform/Arith/MulToAdd.h>
#include "lib/Transform/Affine/Passes.h"
#include "mlir/include/mlir/InitAllDialects.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/include/mlir/Tools/mlir-opt/MlirOptMain.h"

int main(int argc, char **argv) {
    mlir::DialectRegistry registry;
    mlir::registerAllDialects(registry);

    mlir::PassRegistration<mlir::tutorial::MulToAddPass>();
    mlir::tutorial::registerPasses();

    return mlir::asMainReturnCode(
        mlir::MlirOptMain(argc, argv, "Tutorial Pass Driver", registry));
}
