#include "ConvertToSum.hpp"
#include "Option.hpp"
#include "OptionOps.hpp"
#include "OptionTypes.hpp"
#include "SumOps.hpp"
#include "SumTypeInterface.hpp"
#include <mlir/IR/BuiltinTypes.h>
#include <mlir/Transforms/GreedyPatternRewriteDriver.h>

namespace mlir::option {

// option.none → sum.make 0 (no payload) : !option.option<T>
struct NoneOpLowering : OpRewritePattern<NoneOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult
  matchAndRewrite(NoneOp op, PatternRewriter &rewriter) const override {
    // Variant 0 = None (nullary)
    rewriter.replaceOpWithNewOp<sum::MakeOp>(op, op.getResult().getType(), 0);
    return success();
  }
};

// option.some %v → sum.make 1 %v : !option.option<T>
struct SomeOpLowering : OpRewritePattern<SomeOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult
  matchAndRewrite(SomeOp op, PatternRewriter &rewriter) const override {
    auto indexAttr = rewriter.getIndexAttr(1);
    rewriter.replaceOpWithNewOp<sum::MakeOp>(
        op, op.getResult().getType(), indexAttr, op.getValue());
    return success();
  }
};

// option.is_some %x → sum.is_variant %x, 1
struct IsSomeOpLowering : OpRewritePattern<IsSomeOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult
  matchAndRewrite(IsSomeOp op, PatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<sum::IsVariantOp>(
        op, op.getInput(), /*index=*/1);
    return success();
  }
};

// option.unwrap_or %x, %def →
//   sum.match %x : !option.option<T> -> T
//   case 0 { yield %def }
//   case 1 (%v: T) { yield %v }
struct UnwrapOrOpLowering : OpRewritePattern<UnwrapOrOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult
  matchAndRewrite(UnwrapOrOp op, PatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    Type resultTy = op.getResult().getType();

    auto matchOp = rewriter.create<sum::MatchOp>(
        loc, TypeRange{resultTy}, op.getOption(), /*casesCount=*/2u);

    // Case 0: None → yield default
    {
      Region &caseRegion = matchOp.getCases()[0];
      Block *block = rewriter.createBlock(&caseRegion);
      rewriter.setInsertionPointToEnd(block);
      rewriter.create<sum::YieldOp>(loc, ValueRange{op.getDefaultValue()});
    }

    // Case 1: Some(%v) → yield %v
    {
      Region &caseRegion = matchOp.getCases()[1];
      Block *block = rewriter.createBlock(&caseRegion, {}, {resultTy}, {loc});
      rewriter.setInsertionPointToEnd(block);
      rewriter.create<sum::YieldOp>(loc, ValueRange{block->getArgument(0)});
    }

    rewriter.replaceOp(op, matchOp.getResults());
    return success();
  }
};

// option.and_then %x : !option.option<T> -> !option.option<U> { body } →
//   sum.match %x : !option.option<T> -> !option.option<U>
//   case 0 { %none = sum.make 0 : result_ty; yield %none }
//   case 1 (%v: T) { <body with option.yield → sum.yield> }
struct AndThenOpLowering : OpRewritePattern<AndThenOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult
  matchAndRewrite(AndThenOp op, PatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    Type resultTy = op.getResult().getType();

    auto matchOp = rewriter.create<sum::MatchOp>(
        loc, TypeRange{resultTy}, op.getInput(), /*casesCount=*/2u);

    // Case 0: None → yield sum.make 0 (none of result type)
    {
      Region &caseRegion = matchOp.getCases()[0];
      Block *block = rewriter.createBlock(&caseRegion);
      rewriter.setInsertionPointToEnd(block);
      auto noneVal = rewriter.create<sum::MakeOp>(loc, resultTy, 0);
      rewriter.create<sum::YieldOp>(loc, ValueRange{noneVal});
    }

    // Case 1: Some(%v) → inline body, replacing option.yield with sum.yield
    {
      Region &caseRegion = matchOp.getCases()[1];

      // Move the body from and_then into case 1
      rewriter.inlineRegionBefore(op.getThenRegion(), caseRegion, caseRegion.end());

      // Replace option.yield terminators with sum.yield
      for (Block &block : caseRegion) {
        if (auto yield = dyn_cast<YieldOp>(block.getTerminator())) {
          rewriter.setInsertionPoint(yield);
          rewriter.replaceOpWithNewOp<sum::YieldOp>(yield, ValueRange{yield.getResult()});
        }
      }
    }

    rewriter.replaceOp(op, matchOp.getResults());
    return success();
  }
};

void populateOptionToSumConversionPatterns(RewritePatternSet &patterns) {
  patterns.add<
    NoneOpLowering,
    SomeOpLowering,
    IsSomeOpLowering,
    UnwrapOrOpLowering,
    AndThenOpLowering
  >(patterns.getContext());
}

void ConvertOptionToSumPass::runOnOperation() {
  RewritePatternSet patterns(&getContext());
  populateOptionToSumConversionPatterns(patterns);
  if (failed(applyPatternsGreedily(getOperation(), std::move(patterns))))
    signalPassFailure();
}

} // end mlir::option
