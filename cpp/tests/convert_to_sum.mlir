// RUN: mlir-opt --pass-pipeline="builtin.module(convert-option-to-sum)" %s | FileCheck %s

// ---- Test: option.none → sum.make 0 (nullary) ----
// CHECK-LABEL: func @lower_none
// CHECK: sum.make 0 : !option.option<i32>
// CHECK-NOT: option.none
func.func @lower_none() -> !option.option<i32> {
  %0 = option.none : !option.option<i32>
  return %0 : !option.option<i32>
}

// ---- Test: option.some → sum.make 1 ----
// CHECK-LABEL: func @lower_some
// CHECK: sum.make 1 %arg0 : !option.option<i32>
// CHECK-NOT: option.some
func.func @lower_some(%arg0: i32) -> !option.option<i32> {
  %0 = option.some %arg0 : !option.option<i32>
  return %0 : !option.option<i32>
}

// ---- Test: option.is_some → sum.is_variant 1 ----
// CHECK-LABEL: func @lower_is_some
// CHECK: sum.is_variant %arg0, 1 : !option.option<i32>
// CHECK-NOT: option.is_some
func.func @lower_is_some(%arg0: !option.option<i32>) -> i1 {
  %0 = option.is_some %arg0 : !option.option<i32>
  return %0 : i1
}

// ---- Test: option.unwrap_or → sum.match ----
// CHECK-LABEL: func @lower_unwrap_or
// CHECK: sum.match %arg0 : !option.option<i32> -> i32
// CHECK: case 0
// CHECK: yield
// CHECK: case 1
// CHECK: yield
// CHECK-NOT: option.unwrap_or
func.func @lower_unwrap_or(%arg0: !option.option<i32>, %arg1: i32) -> i32 {
  %0 = option.unwrap_or %arg0, %arg1 : i32
  return %0 : i32
}

// ---- Test: option.and_then → sum.match ----
// CHECK-LABEL: func @lower_and_then
// CHECK: sum.match %arg0 : !option.option<i32> -> !option.option<f32>
// CHECK: case 0
// CHECK: make 0 : !option.option<f32>
// CHECK: case 1
// CHECK-NOT: option.and_then
func.func @lower_and_then(%arg0: !option.option<i32>) -> !option.option<f32> {
  %0 = option.and_then %arg0 : !option.option<i32> -> !option.option<f32> {
  ^bb0(%unwrapped: i32):
    %1 = arith.sitofp %unwrapped : i32 to f32
    %2 = option.some %1 : !option.option<f32>
    option.yield %2 : !option.option<f32>
  }
  return %0 : !option.option<f32>
}
