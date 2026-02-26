#ifndef OPTION_DIALECT_TD
#define OPTION_DIALECT_TD

include "mlir/IR/OpBase.td"

def OptionDialect : Dialect {
  let name = "option";
  let summary = "A dialect for monadic option<T> types";
  let cppNamespace = "::option";
  
  let useDefaultTypePrinterParser = 1;

  let extraClassDeclaration = [{
    void registerTypes();
  }];

  let dependentDialects = [
    "mlir::scf::SCFDialect"
  ];

  let hasCanonicalizer = 1;
}

class OptionOp<string mnemonic, list<Trait> traits = []> :
  Op<OptionDialect, mnemonic, traits>;


#endif // OPTION_DIALECT_TD
