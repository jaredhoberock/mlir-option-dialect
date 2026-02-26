#include "Option.hpp"
#include "OptionTypes.hpp"
#include <llvm/ADT/TypeSwitch.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/DialectImplementation.h>

using namespace mlir;
using namespace mlir::option;

#define GET_TYPEDEF_CLASSES
#include "OptionTypes.cpp.inc"

void OptionDialect::registerTypes() {
  addTypes<
#define GET_TYPEDEF_LIST
#include "OptionTypes.cpp.inc"
  >();
}
