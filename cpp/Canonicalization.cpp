#include "Option.hpp"
#include "OptionOps.hpp"
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/IR/PatternMatch.h>

using namespace mlir;
using namespace mlir::option;

struct FoldIdentityAndThen : public OpRewritePattern<AndThenOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(AndThenOp op,
                                PatternRewriter& rewriter) const override {
    // check if input is some
    auto inputSomeOp = op.getInput().getDefiningOp<SomeOp>();
    if (not inputSomeOp)
      return failure();

    // check if yield's operand comes from a some op
    auto yieldSomeOp = op.thenYield().getResult().getDefiningOp<SomeOp>();
    if (not yieldSomeOp)
      return failure();

    // check if we are simply yielding the wrapped value of the block argument
    if (yieldSomeOp.getValue() != op.thenBlock()->getArgument(0))
      return failure();

    // this is just an identity transformation - replace with the input
    rewriter.replaceOp(op, inputSomeOp.getResult());
    return success();
  }
};

struct FoldUnwrapOrSome : public OpRewritePattern<SomeOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(SomeOp op,
                                PatternRewriter& rewriter) const override {
    // check if the input comes from unwrap_or
    auto unwrapOp = op.getValue().getDefiningOp<UnwrapOrOp>();
    if (not unwrapOp)
      return failure();

    // replace with the original option
    rewriter.replaceOp(op, unwrapOp.getOption());
    return success();
  }
};

void OptionDialect::getCanonicalizationPatterns(RewritePatternSet& patterns) const {
  patterns.add<
    FoldIdentityAndThen,
    FoldUnwrapOrSome
  >(patterns.getContext());
}
