#include "Option.hpp"
#include "OptionTypes.hpp"
#include "SumTypeInterface.hpp"
#include <llvm/ADT/TypeSwitch.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/BuiltinTypes.h>
#include <mlir/IR/DialectImplementation.h>

using namespace mlir;
using namespace mlir::option;

#define GET_TYPEDEF_CLASSES
#include "OptionTypes.cpp.inc"

// SumTypeInterface implementation for OptionType
// Variant 0 = None (NoneType), Variant 1 = Some(T)
size_t OptionType::getNumVariants() const { return 2; }

Type OptionType::getVariantType(size_t index) const {
  if (index == 0)
    return NoneType::get(getContext());
  return getInnerType();
}

void OptionDialect::registerTypes() {
  addTypes<
#define GET_TYPEDEF_LIST
#include "OptionTypes.cpp.inc"
  >();
}
