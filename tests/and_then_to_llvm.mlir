// RUN: opt %s -canonicalize --convert-to-llvm | FileCheck %s

// CHECK-LABEL: llvm.func @and_then_and_then
// CHECK-SAME: (%[[ARG0:.*]]: i32, %[[ARG1:.*]]: i32) -> !llvm.struct<(i[[FLAGWIDTH:.*]], i32)>
func.func @and_then_and_then(%arg0 : i32, %arg1 : i32) -> !option.option<i32> {
  // CHECK: %[[SOME0:.*]] = llvm.mlir.constant(1 : i[[FLAGWIDTH]]) : i[[FLAGWIDTH]]
  // CHECK: %[[STRUCT0:.*]] = llvm.mlir.undef : !llvm.struct<(i[[FLAGWIDTH]], i32)>
  // CHECK: %[[STRUCT0_1:.*]] = llvm.insertvalue %[[SOME0]], %[[STRUCT0]][0]
  // CHECK: %[[STRUCT0_2:.*]] = llvm.insertvalue %[[ARG0]], %[[STRUCT0_1]][1]
  %o0 = option.some %arg0 : !option.option<i32>

  // CHECK: %[[SOME1:.*]] = llvm.mlir.constant(1 : i[[FLAGWIDTH]]) : i[[FLAGWIDTH]]
  // CHECK: %[[STRUCT1:.*]] = llvm.mlir.undef : !llvm.struct<(i[[FLAGWIDTH]], i32)>
  // CHECK: %[[STRUCT1_1:.*]] = llvm.insertvalue %[[SOME1]], %[[STRUCT1]][0]
  // CHECK: %[[STRUCT1_2:.*]] = llvm.insertvalue %[[ARG1]], %[[STRUCT1_1]][1]
  %o1 = option.some %arg1 : !option.option<i32>

  // CHECK: %[[FLAG0:.*]] = llvm.extractvalue %[[STRUCT0_2]][0]
  // CHECK: %[[COND0:.*]] = llvm.trunc %[[FLAG0]] : [[FLAG0TY:.*]] to i1
  // CHECK: llvm.cond_br %[[COND0]], ^[[THEN0:.*]], ^[[ELSE0:.*]]
  %result0 = option.and_then %o0 : !option.option<i32> -> !option.option<i32> {
  ^bb0(%unwrapped0 : i32):
    // CHECK: ^[[THEN0]]:
    // CHECK: %[[UNWRAPPED0:.*]] = llvm.extractvalue %[[STRUCT0_2]][1]
    // CHECK: %[[FLAG1:.*]] = llvm.extractvalue %[[STRUCT1_2]][0]
    // CHECK: %[[COND1:.*]] = llvm.trunc %[[FLAG1]] : [[FLAG0TY:.*]] to i1
    // CHECK: llvm.cond_br %[[COND1]], ^[[THEN1:.*]], ^[[ELSE1:.*]]
    %result1 = option.and_then %o1 : !option.option<i32> -> !option.option<i32> {
    ^bb1(%unwrapped1 : i32):
      // CHECK: ^[[THEN1]]:
      // CHECK: %[[UNWRAPPED1:.*]] = llvm.extractvalue %[[STRUCT1_2]][1]
      // CHECK: %[[SUM:.*]] = llvm.add %[[UNWRAPPED0]], %[[UNWRAPPED1]]
      %sum = arith.addi %unwrapped0, %unwrapped1 : i32
      // CHECK: %[[INNER_SOME:.*]] = llvm.mlir.constant(1 : i[[FLAGWIDTH]]) : i[[FLAGWIDTH]]
      // CHECK: %[[INNER_RESULT:.*]] = llvm.mlir.undef : !llvm.struct<(i[[FLAGWIDTH]], i32)>
      // CHECK: %[[INNER_STRUCT:.*]] = llvm.insertvalue %[[INNER_SOME]], %[[INNER_RESULT]][0]
      // CHECK: %[[INNER_STRUCT_2:.*]] = llvm.insertvalue %[[SUM]], %[[INNER_STRUCT]][1]
      %result2 = option.some %sum : !option.option<i32>
      option.yield %result2 : !option.option<i32>
    }
    option.yield %result1 : !option.option<i32>
  }
  return %result0 : !option.option<i32>
}
