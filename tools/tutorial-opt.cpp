#include "mlir/include/mlir/InitAllDialects.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/include/mlir/Tools/mlir-opt/MlirOptMain.h"

#include "lib/Transform/Affine/Passes.h"
#include "lib/Transform/Arith/Passes.h"


int main(int argc, char **argv) {
    mlir::DialectRegistry registry;
    mlir::registerAllDialects(registry);

    mlir::tutorial::registerAffinePasses();
    mlir::tutorial::registerArithPasses();


    return mlir::asMainReturnCode(
        mlir::MlirOptMain(argc, argv, "Tutorial Pass Driver", registry));
}
