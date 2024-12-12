#pragma once

namespace mlir {

class LLVMTypeConverter;
class RewritePatternSet;

}

namespace option {

void populateOptionToLLVMConversionPatterns(mlir::LLVMTypeConverter& typeConverter,
                                            mlir::RewritePatternSet& patterns);
}
