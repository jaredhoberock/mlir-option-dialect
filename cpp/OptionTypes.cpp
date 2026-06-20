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
#include <OptionTypes.cpp.inc>

// SumTypeInterface: OptionType has two variants — None and Some(T).
// The default getNumVariants/getVariantType call getVariants().
SmallVector<Type, 2> OptionType::getVariants() const {
  return {NoneType::get(getContext()), getInnerType()};
}

void OptionDialect::registerTypes() {
  addTypes<
#define GET_TYPEDEF_LIST
#include <OptionTypes.cpp.inc>
  >();
}
