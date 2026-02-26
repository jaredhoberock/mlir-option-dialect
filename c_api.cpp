#include "c_api.h"
#include "Option.hpp"
#include "OptionOps.hpp"
#include <mlir/CAPI/IR.h>
#include <mlir/CAPI/Pass.h>
#include <mlir/IR/Builders.h>

extern "C" {

void optionRegisterDialect(MlirContext context) {
  mlir::MLIRContext *ctx = unwrap(context);
  ctx->loadDialect<mlir::option::OptionDialect>();
}

} // end extern "C"
