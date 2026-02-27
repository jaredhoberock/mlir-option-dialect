#pragma once

#include <mlir/Pass/Pass.h>

namespace mlir {

class Pass;
class RewritePatternSet;

namespace option {

struct ConvertOptionToSumPass : public PassWrapper<ConvertOptionToSumPass, OperationPass<>> {
  inline StringRef getArgument() const override { return "convert-option-to-sum"; }
  inline StringRef getDescription() const override {
    return "Convert option dialect ops to sum dialect ops";
  }

  void runOnOperation() override;
};

void populateOptionToSumConversionPatterns(RewritePatternSet& patterns);

inline std::unique_ptr<Pass> createConvertOptionToSumPass() {
  return std::make_unique<ConvertOptionToSumPass>();
}

}
}
