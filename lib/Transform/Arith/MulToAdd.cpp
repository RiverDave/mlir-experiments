#include "lib/Transform/Arith/MulToAdd.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/include/mlir/Pass/Pass.h"

namespace mlir {
    namespace tutorial {

        using arith::AddIOp;
        using arith::ConstantOp;
        using arith::MulIOp;

        // Replace y = C*x with y = C/2*x + C/2*x,where C represents a constant.
        // when C is a power of 2, otherwise do nothing.
        //Cannonical optimization constants are always meant to be on the rhs where valid
        // So in effect this would be translated to: C * x => x * C/2
        struct PowerOfTwoExpand :
          public OpRewritePattern<MulIOp> {
            PowerOfTwoExpand(mlir::MLIRContext *context)
                : OpRewritePattern<MulIOp>(context, /*benefit=*/2) {}

            LogicalResult matchAndRewrite(MulIOp op,
                                          PatternRewriter &rewriter) const override {

                Value lhs = op.getOperand(0);
                Value rhs = op.getOperand(1); // constant value

                // Verify that rhs is an integer
                arith::ConstantIntOp rhsDefiningOp = rhs.getDefiningOp<arith::ConstantIntOp>();
                if (!rhsDefiningOp) {
                    return failure();
                }

                int64_t value = rhsDefiningOp.value();
                //shift bits and sustract 1
                //Po2's usually have only 1 bitset hence => x - 1 = 0, therefore it is a power of two.
                bool is_power_of_two = (value & (value -1)) == 0;

                if (!is_power_of_two) {
                    return failure();
                }

                // Define new operation
                ConstantOp newConstant = rewriter.create<ConstantOp>(
        rhsDefiningOp.getLoc(), rewriter.getIntegerAttr(rhs.getType(), value / 2));

                MulIOp newMul = rewriter.create<MulIOp>(op.getLoc(), lhs, newConstant);
                AddIOp newAdd = rewriter.create<AddIOp>(op.getLoc(), newMul, newMul);

                // So the result would be
                // newMul = lhs * (C/2)
                // newAdd (C/2) + (C/2)

                /*
                 * 2 * 8
                 * 2 * 8/2, hence
                 * = (2 * 4) + (2 * 4) => This is where we can keep greedily executing this pass
                 * = ((2 * 2) + (2 * 2)) + ((2 * 2) + (2 * 2))
                 * = ((2 * 1) + (2 * 1) + (2 * 1) + (2 * 1)) + ((2 * 1) + (2 * 1) + (2 * 1) + (2 * 1))
                 * = ((2) + (2) + (2) + (2)) + ((2) + (2) + (2) + (2))
                 * (Okay this is beautiful)
                 */

                rewriter.replaceOp(op, newAdd);
                rewriter.eraseOp(rhsDefiningOp);

                return success();
            }
        };

        // Replace y = 9*x with y = 8*x + x
        struct PeelFromMul :
          public OpRewritePattern<MulIOp> {
            PeelFromMul(mlir::MLIRContext *context)
                : OpRewritePattern<MulIOp>(context, /*benefit=*/1) {}

            LogicalResult matchAndRewrite(MulIOp op,
                                          PatternRewriter &rewriter) const override {
                return success();
            }
        };

        void MulToAddPass::runOnOperation() {
            mlir::RewritePatternSet patterns(&getContext());
            patterns.add<PowerOfTwoExpand>(&getContext());
            patterns.add<PeelFromMul>(&getContext());

            (void)applyPatternsGreedily(getOperation(), std::move(patterns));
        }

    } // namespace tutorial
} // namespace mlir