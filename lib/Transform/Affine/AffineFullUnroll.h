#ifndef LIB_TRANSFORM_AFFINE_AFFINEFULLUNROLL_H_
#define LIB_TRANSFORM_AFFINE_AFFINEFULLUNROLL_H_

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/include/mlir/Pass/Pass.h"

namespace mlir {
namespace tutorial {

class AffineFullUnrollPass
    : public PassWrapper<AffineFullUnrollPass, //CRTP => Concept, see https://en.wikipedia.org/wiki/Curiously_recurring_template_pattern
                         OperationPass<mlir::func::FuncOp>> { //OperationPass, basically anchors ourself to a specific operation
private:
  void runOnOperation() override;

  //These are functions required for the opt tool to work and register our pass.
  StringRef getArgument() const final { return "affine-full-unroll"; }

  StringRef getDescription() const final {
    return "Fully unroll all affine loops";
  }
};

class AffineFullUnrollPassAsPatternRewrite
  : public PassWrapper<AffineFullUnrollPassAsPatternRewrite,
                       OperationPass<mlir::func::FuncOp>> {
private:
  void runOnOperation() override;

  StringRef getArgument() const final { return "affine-full-unroll-rewrite"; }

  StringRef getDescription() const final {
    return "Fully unroll all affine loops using pattern rewrite engine";
  }
};

} // namespace tutorial
} // namespace mlir

#endif // LIB_TRANSFORM_AFFINE_AFFINEFULLUNROLL_H_
