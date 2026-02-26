#include "Option.hpp"
#include <mlir/Tools/Plugins/DialectPlugin.h>

static void registerPlugin(mlir::DialectRegistry* registry) {
  registry->insert<option::OptionDialect>();
}

extern "C" ::mlir::DialectPluginLibraryInfo LLVM_ATTRIBUTE_WEAK
mlirGetDialectPluginInfo() {
  return {
    MLIR_PLUGIN_API_VERSION, "OptionDialectPlugin", "v0.1", registerPlugin
  };
}
