#pragma once

namespace mlir {

class LLVMTypeConverter;
class RewritePatternSet;

}

namespace mlir::option {

void populateOptionToLLVMConversionPatterns(LLVMTypeConverter& typeConverter,
                                            RewritePatternSet& patterns);
}
