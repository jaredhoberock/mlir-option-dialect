#include "option_c.h"
#include "Dialect.hpp"
#include "Ops.hpp"
#include <mlir/CAPI/IR.h>
#include <mlir/CAPI/Pass.h>
#include <mlir/IR/Builders.h>

extern "C" {

void optionRegisterDialect(MlirContext context) {
  mlir::MLIRContext *ctx = unwrap(context);
  ctx->loadDialect<option::OptionDialect>();
}

} // end extern "C"
