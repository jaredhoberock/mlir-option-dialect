// RUN: opt %s -canonicalize -split-input-file | FileCheck %s

// CHECK-LABEL: func @test_and_then_fold_none
func.func @test_and_then_fold_none() -> !option.option<i32> {
  // CHECK-NEXT: %[[NONE:.*]] = option.none : !option.option<i32>
  // CHECK-NEXT: return %[[NONE]]
  %0 = option.none : !option.option<f32>
  %1 = option.and_then %0 : !option.option<f32> -> !option.option<i32> {
  ^bb0(%arg0: f32):
    %2 = arith.fptosi %arg0 : f32 to i32
    %3 = option.some %2 : !option.option<i32>
    option.yield %3 : !option.option<i32>
  }
  return %1 : !option.option<i32>
}

// -----

// CHECK-LABEL: func @test_and_then_no_fold_some
func.func @test_and_then_no_fold_some() -> !option.option<i32> {
  %0 = arith.constant 1.0 : f32
  %1 = option.some %0 : !option.option<f32>
  // CHECK: option.and_then
  %2 = option.and_then %1 : !option.option<f32> -> !option.option<i32> {
  ^bb0(%arg0: f32):
    %3 = arith.fptosi %arg0 : f32 to i32
    %4 = option.some %3 : !option.option<i32>
    option.yield %4 : !option.option<i32>
  }
  return %2 : !option.option<i32>
}
