#ifndef OPTION_OPS_TD
#define OPTION_OPS_TD

include "mlir/Interfaces/ControlFlowInterfaces.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/IR/BuiltinTypes.td"
include "mlir/IR/OpAsmInterface.td"
include "mlir/IR/OpBase.td"
include "Option.td"
include "OptionTypes.td"

def NoneOp : OptionOp<"none", [Pure]> {
  let summary = "Creates an empty optional value";
  let description = [{
    The none operation creates an optional value containing no value (None)
    of a specified type.

    Example:
    ```mlir
    %0 = option.none : !option.option<i32>
    ```
  }];

  let results = (outs OptionType:$result);
  let assemblyFormat = "attr-dict `:` qualified(type($result))";
}

def SomeOp : OptionOp<"some", [
  Pure,
  TypesMatchWith<"value type matches inner type of result",
                 "result", "value",
                 "mlir::cast<OptionType>($_self).getInnerType()">
]> {
  let summary = "Creates an optional value containing a value";
  let description = [{
    The some operation wraps a value into an optional.

    Example:
    ```mlir
    %0 = arith.constant 42 : i32
    %1 = option.some %0 : !option.option<i32>
    ```
  }];

  let arguments = (ins AnyType:$value);
  let results = (outs OptionType:$result);
  let assemblyFormat = "$value attr-dict `:` qualified(type($result))";

  let extraClassDeclaration = [{
    // build a SomeOp from the specified input value
    static void build(::mlir::OpBuilder& builder, ::mlir::OperationState& result,
                      ::mlir::Value value) {
      auto optionTy = OptionType::get(builder.getContext(), value.getType());
      build(builder, result, optionTy, value);
    }
  }];
}

def AndThenOp : OptionOp<"and_then"> {
  let summary = "Maps an `option.option` value to another `option.option` using the provided region";
  let description = [{
    The `option.and_then` operation maps an `!option.option<T>` to `!option.option<U>` using the provided
    region code. If the input is `option.none`, the result is `option.none`. Within the region, references
    to the input value refer to its unwrapped value of type `T`.

    Example:
    ```mlir
    %val = option.some %x : !option.option<i32>
    %res = option.and_then %val : !option.option<i32> -> !option.option<f32> {
    ^bb0(%unwrapped : i32):
      %1 = arith.sitofp %unwrapped : i32 to f32
      %2 = option.some %1 : !option.option<f32>
      option.yield %2 : !option.option<f32>
    }
    ```
  }];

  let arguments = (ins OptionType:$input);
  let results = (outs OptionType:$result);
  let regions = (region SizedRegion<1>:$thenRegion);

  let extraClassDeclaration = [{
    mlir::Block* thenBlock();
    YieldOp thenYield();

    // Build an AndThenOp with the specified input and result type
    static void build(::mlir::OpBuilder& builder, ::mlir::OperationState& result,
                      ::mlir::Value input,
                      ::llvm::function_ref<void(::mlir::OpBuilder&, ::mlir::Location, ::mlir::Value)> thenBuilder);
  }];

  let assemblyFormat = "$input `:` qualified(type($input)) `->` qualified(type($result)) $thenRegion attr-dict";
  let hasVerifier = 1;
  let hasFolder = 1;
}

def IsSomeOp : OptionOp<"is_some", [
  Pure
]> {
  let summary = "Returns whether an `option.option` contains a value";
  let description = [{
    The `option.is_some` operation returns an `i1` indicating whether an `!option.option<T>` originated from
    an `option.some` operation. In other words, `option.is_some` returns whether or not the given `!option.option<T>`
    contains a value.

    Example:
    ```mlir
    %some_value = option.some %s : !option.option<i32>
    %cond = option.is_some %some_value : !option.option<i32>
    %result = scf.if %cond {
      %c0 = arith.constant 13 : i32
      scf.yield %c0 : i32 
    } else {
      %c1 = arith.constant 42 : i32
      scf.yield %c1 : i32
    }
    // %result is 13
    ```
  }];

  let arguments = (ins OptionType:$input);
  let results = (outs I1:$result);

  let assemblyFormat = "$input attr-dict `:` qualified(type($input))";
  let hasFolder = 1;

  let extraClassDeclaration = [{
    // build an is_some op with the result type inferred
    static void build(::mlir::OpBuilder& builder, ::mlir::OperationState& result,
                      ::mlir::Value option) {
      build(builder, result, builder.getI1Type(), option);
    }
  }];
}

def YieldOp : OptionOp<"yield", [
  HasParent<"AndThenOp">,
  Pure,
  ReturnLike,
  Terminator
]> {
  let summary = "Yield operation for option.and_then";
  let description = [{
    `option.yield` yields SSA values from the `option.and_then` region and terminates the region.
    `option.yield`'s argument type must match the type expected by the `option.and_then`'s result.
  }];

  // Allow any number of types for operands
  let arguments = (ins OptionType:$result);

  // Define the assembly format for the `yield` operation
  let assemblyFormat = "attr-dict $result `:` qualified(type($result))";
}

def UnwrapOrOp : OptionOp<"unwrap_or", [
  Pure,
  TypesMatchWith<"default_value matches type of result",
                 "result", "defaultValue",
                 "$_self">,
  TypesMatchWith<"option matches option of result type",
                 "result", "option",
                 "OptionType::get($_self.getContext(), $_self)">
]> {
  let summary = "Unwraps an `option<T>` or returns the provided default value";
  let description = [{
    Returns the contained value if the `!option.option` is `option.some`, otherwise returns the provided default.

    Example:
    ```mlir
    %none = option.none : !option.option<i32>
    %default = arith.constant 42 : i32
    %1 = option.unwrap_or %none, %default : i32
    // %1 is 42

    %some = option.some %val : !option.option<i32>
    %2 = option.unwrap_or %some, %default : i32
    // %2 is %val
  }];

  let arguments = (ins OptionType:$option, AnyType:$defaultValue);
  let results = (outs AnyType:$result);
  let hasFolder = 1;

  let assemblyFormat = "$option `,` $defaultValue attr-dict `:` qualified(type($result))";

  let extraClassDeclaration = [{
    // build an unwrap_or op with the type inferred from inputs
    static void build(::mlir::OpBuilder& builder, ::mlir::OperationState& result,
                      ::mlir::Value option, ::mlir::Value defaultValue) {
      build(builder, result, defaultValue.getType(), option, defaultValue);
    }
  }];
}

#endif // OPTION_OPS_TD
