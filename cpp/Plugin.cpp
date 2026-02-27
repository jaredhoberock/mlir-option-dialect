#include "ConvertToSCF.hpp"
#include "ConvertToSum.hpp"
#include "Option.hpp"
#include "Sum.hpp"
#include <mlir/Tools/Plugins/DialectPlugin.h>

static void registerPlugin(mlir::DialectRegistry* registry) {
  registry->insert<mlir::option::OptionDialect>();
  registry->insert<mlir::sum::SumDialect>();
  ::mlir::PassRegistration<::mlir::option::ConvertOptionToSumPass>();
  ::mlir::PassRegistration<::mlir::sum::ConvertSumToSCFPass>();
}

extern "C" ::mlir::DialectPluginLibraryInfo LLVM_ATTRIBUTE_WEAK
mlirGetDialectPluginInfo() {
  return {
    MLIR_PLUGIN_API_VERSION, "OptionDialectPlugin", "v0.1", registerPlugin
  };
}
