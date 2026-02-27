#ifndef OPTION_TYPES_TD
#define OPTION_TYPES_TD

include "mlir/IR/AttrTypeBase.td"
include "Option.td"
include "SumTypeInterface.td"

def OptionType : TypeDef<OptionDialect, "Option", [
  DeclareTypeInterfaceMethods<SumTypeInterface>
]> {
  let mnemonic = "option";
  let summary = "A container for at most one value";
  let parameters = (ins
    "Type":$innerType
  );

  let assemblyFormat = "`<` $innerType `>`";
}

#endif // OPTION_TYPES_TD
