// RUN: mlir-opt --pass-pipeline="builtin.module(convert-option-to-sum,convert-sum-to-scf,convert-scf-to-cf,convert-to-llvm)" %s | FileCheck %s

// CHECK-LABEL: llvm.func @and_then_and_then
// CHECK-NOT: option.
// CHECK-NOT: sum.
// CHECK-NOT: builtin.unrealized_conversion_cast
// CHECK: llvm.return
func.func @and_then_and_then(%arg0 : i32, %arg1 : i32) -> !option.option<i32> {
  %o0 = option.some %arg0 : !option.option<i32>
  %o1 = option.some %arg1 : !option.option<i32>

  %result0 = option.and_then %o0 : !option.option<i32> -> !option.option<i32> {
  ^bb0(%unwrapped0 : i32):
    %result1 = option.and_then %o1 : !option.option<i32> -> !option.option<i32> {
    ^bb1(%unwrapped1 : i32):
      %sum = arith.addi %unwrapped0, %unwrapped1 : i32
      %result2 = option.some %sum : !option.option<i32>
      option.yield %result2 : !option.option<i32>
    }
    option.yield %result1 : !option.option<i32>
  }
  return %result0 : !option.option<i32>
}
