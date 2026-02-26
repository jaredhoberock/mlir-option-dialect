#include "Option.hpp"
#include "OptionOps.hpp"
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Interfaces/FunctionImplementation.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/BuiltinTypes.h>
#include <mlir/IR/OpImplementation.h>

#define GET_OP_CLASSES
#include "OptionOps.cpp.inc"

using namespace mlir;
using namespace option;

// AndThenOp

Block* AndThenOp::thenBlock() { return &getThenRegion().back(); }
YieldOp AndThenOp::thenYield() { return cast<YieldOp>(thenBlock()->back()); }


void AndThenOp::build(OpBuilder& builder, OperationState& result,
                      Value input,
                      llvm::function_ref<void(OpBuilder&, Location, Value)> thenBuilder) {
  // get input type
  auto inputTy = cast<OptionType>(input.getType());

  // create the body/region
  Region* thenRegion = result.addRegion();
  Block& thenBlock = thenRegion->emplaceBlock();
  thenBlock.addArgument(inputTy.getInnerType(), result.location);

  // set insertion point to start of block and call the provided builder
  OpBuilder::InsertionGuard guard(builder);
  builder.setInsertionPointToStart(&thenBlock);
  thenBuilder(builder, result.location, thenBlock.getArgument(0));

  // build the main operation
  build(builder, result, inputTy, input);
}


LogicalResult AndThenOp::verify() {
  Block& block = *thenBlock();

  // Verify block argument matches inner input type
  if (block.getNumArguments() != 1)
    return emitOpError("expected block to have exactly one argument");

  auto blockArg = block.getArguments()[0];

  if (blockArg.getType() != getInput().getType().getInnerType())
    return emitOpError("block argument type ")
      << blockArg.getType()
      << " does not match inner input type "
      << getInput().getType().getInnerType();

  // Verify the yield op's type
  auto yieldOp = cast<YieldOp>(block.getTerminator());
  if (yieldOp.getResult().getType() != getResult().getType())
    return emitOpError("yield type ")
      << yieldOp.getResult().getType()
      << " does not match result type "
      << getResult().getType();

  return success();
}


OpFoldResult AndThenOp::fold(FoldAdaptor adaptor) {
  if (auto noneOp = getInput().getDefiningOp<NoneOp>()) {
    OpBuilder builder(getContext());
    builder.setInsertionPoint(getOperation());

    // Create a new NoneOp with the result type
    return builder.create<NoneOp>(getLoc(), getResult().getType()).getResult();
  }

  return {};
}


// UnwrapOrOp

OpFoldResult UnwrapOrOp::fold(FoldAdaptor adaptor) {
  // If input is none, return default
  if (getOption().getDefiningOp<NoneOp>())
    return getDefaultValue();

  // If input is some, return wrapped value
  if (auto someOp = getOption().getDefiningOp<SomeOp>())
    return someOp.getValue();

  return {};
}


// IsSomeOp

OpFoldResult IsSomeOp::fold(FoldAdaptor adaptor) {
  // try to fold some -> true
  if (auto someOp = getInput().getDefiningOp<SomeOp>()) {
    OpBuilder builder(getContext());
    builder.setInsertionPoint(getOperation());
    return builder.create<arith::ConstantOp>(getLoc(), builder.getBoolAttr(true)).getResult();
  }
    
  // try to fold none -> false
  if (auto noneOp = getInput().getDefiningOp<NoneOp>()) {
    OpBuilder builder(getContext());
    builder.setInsertionPoint(getOperation());
    return builder.create<arith::ConstantOp>(getLoc(), builder.getBoolAttr(false)).getResult();
  }
    
  return {};
}
