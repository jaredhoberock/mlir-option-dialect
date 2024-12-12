#include "Dialect.hpp"
#include "Lowering.hpp"
#include "Ops.hpp"
#include <mlir/Conversion/LLVMCommon/TypeConverter.h>
#include <mlir/Conversion/SCFToControlFlow/SCFToControlFlow.h>
#include <mlir/Dialect/LLVMIR/LLVMDialect.h>
#include <mlir/Dialect/LLVMIR/LLVMTypes.h>
#include <mlir/Dialect/SCF/IR/SCF.h>
#include <mlir/Transforms/DialectConversion.h>


using namespace mlir;


namespace option {


static Value createLLVMStructFromValues(OpBuilder& rewriter, Location loc, Type structTy, ArrayRef<Value> values) {
  // begin with an undefined struct
  Value result = rewriter.create<LLVM::UndefOp>(loc, structTy);

  // insert each value into the struct
  for (unsigned i = 0; i < values.size(); ++i) {
    result = rewriter.create<LLVM::InsertValueOp>(loc, structTy, result, values[i], i);
  }

  return result;
}


struct IsSomeOpLowering : public OpConversionPattern<IsSomeOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(IsSomeOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter& rewriter) const override {
    // in principle, we could lower option.is_some to other primitive option operations 
    // but it complicates the lowering process
    //
    // so just lower option.is_some to an operation on the lowered option's LLVM struct

    auto structTy = cast<LLVM::LLVMStructType>(adaptor.getInput().getType());

    // extract the first field of the struct, which is the "presence flag"
    rewriter.replaceOpWithNewOp<LLVM::ExtractValueOp>(
      op, structTy.getBody()[0], adaptor.getInput(), 0
    );

    return success();
  }
};


struct NoneOpLowering : public OpConversionPattern<NoneOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(NoneOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter& rewriter) const override {
    // convert the result type to the target type
    auto structTy = dyn_cast_or_null<LLVM::LLVMStructType>(getTypeConverter()->convertType(op.getType()));
    if (not structTy)
      return rewriter.notifyMatchFailure(op, "type conversion failed");

    // Extract element types
    auto flagTy = structTy.getBody()[0];  // i1
    auto valueTy = structTy.getBody()[1]; // T

    Location loc = op.getLoc();

    // Create the constant presence flag (false)
    Value flagVal = rewriter.create<LLVM::ConstantOp>(loc, flagTy, rewriter.getBoolAttr(false));

    // Create the undefined value
    Value valueVal = rewriter.create<LLVM::UndefOp>(loc, valueTy);

    // Create the final value
    Value noneVal = createLLVMStructFromValues(rewriter, loc, structTy, {flagVal, valueVal});

    // Replace the original operation
    rewriter.replaceOp(op, noneVal);

    return success();
  }
};


struct SomeOpLowering : public OpConversionPattern<SomeOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(SomeOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter& rewriter) const override {
    // convert the result type to the target type
    auto structTy = dyn_cast_or_null<LLVM::LLVMStructType>(getTypeConverter()->convertType(op.getType()));
    if (not structTy)
      return rewriter.notifyMatchFailure(op, "type conversion failed");

    // Extract element types
    auto flagTy = structTy.getBody()[0];
    auto valueTy = structTy.getBody()[1];

    Location loc = op.getLoc();

    // Create the constant presence flag (true)
    Value flagVal = rewriter.create<LLVM::ConstantOp>(loc, flagTy, rewriter.getBoolAttr(true));

    // Create the value
    assert(adaptor.getOperands().size() == 1);
    Value valueVal = adaptor.getOperands()[0];

    // Create the final value
    Value someVal = createLLVMStructFromValues(rewriter, loc, structTy, {flagVal, valueVal});

    // Replace the original operation
    rewriter.replaceOp(op, someVal);
    return success();
  }
};


struct AndThenOpLowering : public OpConversionPattern<AndThenOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(AndThenOp andThenOp, OpAdaptor adaptor,
                                ConversionPatternRewriter& rewriter) const override {
    // Check types
    auto inputStructTy = dyn_cast_or_null<LLVM::LLVMStructType>(adaptor.getInput().getType());
    auto resultOptionTy = dyn_cast_or_null<OptionType>(andThenOp.getResult().getType());
    if (!inputStructTy or !resultOptionTy)
      return rewriter.notifyMatchFailure(andThenOp, "type check failed");

    Location loc = andThenOp.getLoc();

    // Extract the presence flag from input
    Value inputFlag = rewriter.create<LLVM::ExtractValueOp>(
      loc,
      rewriter.getI1Type(),
      adaptor.getInput(),
      0
    );

    // Create an scf.if operation
    auto ifOp = rewriter.create<scf::IfOp>(
      loc,
      resultOptionTy,
      inputFlag,
      /*withElseRegion=*/true
    );

    // fill in the then region
    {
      OpBuilder::InsertionGuard guard(rewriter);

      // extract the value from the option
      rewriter.setInsertionPointToStart(ifOp.thenBlock());
      Value inputVal = rewriter.create<LLVM::ExtractValueOp>(
        loc,
        inputStructTy.getBody()[1],
        adaptor.getInput(),
        1
      );

      // inline the original and_then region
      // XXX note that this inlineRegionBefore can result in multiple blocks in the ifOp's then region
      //     is it our responsibility to ensure that the ifOp's then region has a single block?
      rewriter.inlineRegionBefore(andThenOp.getThenRegion(), ifOp.getThenRegion(), ifOp.getThenRegion().end());

      // merge the first two blocks in ifOp's then region, replacing the
      // 2nd block argument with our extracted value
      auto firstBlockInThenRegion = ifOp.getThenRegion().begin();
      auto secondBlockInThenRegion = std::next(firstBlockInThenRegion);
      rewriter.mergeBlocks(&*secondBlockInThenRegion, &*firstBlockInThenRegion, {inputVal});

      // convert region types
      if (failed(rewriter.convertRegionTypes(&ifOp.getThenRegion(), *getTypeConverter()))) {
        return rewriter.notifyMatchFailure(andThenOp,
                                           "region types converion failed");
      }

      // replace the and_then's terminator with scf.yield
      auto oldYieldOp = dyn_cast<YieldOp>(ifOp.getThenRegion().back().getTerminator());
      if (not oldYieldOp)
        return rewriter.notifyMatchFailure(andThenOp, "failed to find option.yield");

      rewriter.replaceOpWithNewOp<scf::YieldOp>(oldYieldOp, oldYieldOp.getOperand());
    }

    // fill in the else region
    {
      OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPointToStart(ifOp.elseBlock());
      Value noneVal = rewriter.create<NoneOp>(loc, resultOptionTy);
      rewriter.create<scf::YieldOp>(loc, noneVal);
    }

    if (failed(ifOp.verify())) {
      assert(false);
    }

    rewriter.replaceOp(andThenOp, ifOp);
    return success();
  }
};


struct UnwrapOrOpLowering : public OpConversionPattern<UnwrapOrOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(UnwrapOrOp oldOp, OpAdaptor adaptor,
                                ConversionPatternRewriter& rewriter) const override {
    Location loc = oldOp.getLoc();
    Type resultTy = getTypeConverter()->convertType(oldOp.getType());

    // Extract the presence flag from input option
    Value flagVal = rewriter.create<LLVM::ExtractValueOp>(
      loc,
      rewriter.getI1Type(),
      adaptor.getOption(),
      0
    );

    // create if-else structure
    auto ifOp = rewriter.create<scf::IfOp>(
      loc,
      resultTy,
      flagVal,
      /*withElseRegion=*/true
    );

    // Some case - extract and yield the value from the struct
    {
      OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPointToStart(ifOp.thenBlock());

      Value unwrappedVal = rewriter.create<LLVM::ExtractValueOp>(
        loc,
        resultTy,
        adaptor.getOption(),
        1
      );

      rewriter.create<scf::YieldOp>(
        loc,
        unwrappedVal
      );
    }

    // None case - yield the default value
    {
      OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPointToStart(ifOp.elseBlock());

      rewriter.create<scf::YieldOp>(
        loc,
        adaptor.getDefaultValue()
      );
    }

    rewriter.replaceOp(oldOp, ifOp);
    return success();
  }
};


void populateOptionToLLVMConversionPatterns(LLVMTypeConverter& typeConverter, RewritePatternSet& patterns) {
  // add a type conversion for OptionType
  typeConverter.addConversion([&](OptionType type) -> Type {
    auto innerTy = typeConverter.convertType(type.getInnerType());
    if (not innerTy)
      return Type();

    auto flagTy = IntegerType::get(&typeConverter.getContext(), 1);
    return LLVM::LLVMStructType::getLiteral(&typeConverter.getContext(), {flagTy, innerTy});
  });

  patterns.add<
    AndThenOpLowering,
    IsSomeOpLowering,
    NoneOpLowering,
    SomeOpLowering,
    UnwrapOrOpLowering
  >(typeConverter, patterns.getContext());

  populateSCFToControlFlowConversionPatterns(patterns);
}


} // end option
