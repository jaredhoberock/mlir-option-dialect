#pragma once

#include <mlir-c/IR.h>
#include <mlir-c/Pass.h>

#ifdef __cplusplus
extern "C" {
#endif

// Register the dialect with an MLIR context
MLIR_CAPI_EXPORTED void optionRegisterDialect(MlirContext context);

#ifdef __cplusplus
}
#endif
