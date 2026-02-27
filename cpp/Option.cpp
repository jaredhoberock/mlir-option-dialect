#include "Option.hpp"
#include "OptionOps.hpp"
#include "ConvertToSum.hpp"
#include "OptionTypes.hpp"
#include "Sum.hpp"
#include <mlir/Conversion/ConvertToLLVM/ToLLVMInterface.h>
#include <mlir/Conversion/LLVMCommon/TypeConverter.h>
#include <mlir/Dialect/ControlFlow/IR/ControlFlow.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/SCF/IR/SCF.h>
#include <mlir/Dialect/LLVMIR/LLVMDialect.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/OpImplementation.h>

using namespace mlir;
using namespace mlir::option;

#include "Option.cpp.inc"

struct ConvertToLLVMInterface : public mlir::ConvertToLLVMPatternInterface {
  using mlir::ConvertToLLVMPatternInterface::ConvertToLLVMPatternInterface;

  void populateConvertToLLVMConversionPatterns(ConversionTarget& target,
                                               LLVMTypeConverter& typeConverter,
                                               RewritePatternSet& patterns) const override final {
    // Register OptionType → LLVM type conversion matching sum's layout
    typeConverter.addConversion([&](option::OptionType optTy) -> std::optional<Type> {
      Type innerTy = typeConverter.convertType(optTy.getInnerType());
      if (!innerTy)
        return std::nullopt;

      DataLayout layout;
      size_t innerSize = layout.getTypeSize(innerTy).getFixedValue();

      auto *ctx = optTy.getContext();
      auto tagTy = IntegerType::get(ctx, 8);
      auto i8Ty = IntegerType::get(ctx, 8);
      auto payloadTy = LLVM::LLVMArrayType::get(i8Ty, innerSize);
      return LLVM::LLVMStructType::getLiteral(ctx, {tagTy, payloadTy});
    });
    // No op lowering patterns — sum ops handle that after convert-option-to-sum
  }
};

void OptionDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "OptionOps.cpp.inc"
  >();

  registerTypes();

  addInterfaces<
    ConvertToLLVMInterface
  >();
}
