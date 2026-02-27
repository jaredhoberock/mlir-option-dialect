// RUN: mlir-opt %s | FileCheck %s

// CHECK-LABEL: func @test_and_then_none
// CHECK: option.and_then
// CHECK: option.none
// CHECK: option.yield

// CHECK-LABEL: func @test_and_then_some
// CHECK: option.and_then
// CHECK: option.some
// CHECK: option.yield

// CHECK-LABEL: func @test_and_then_some_f32
// CHECK: option.and_then
// CHECK: arith.sitofp
// CHECK: option.some
// CHECK: option.yield

func.func @test_and_then_none(%arg0 : !option.option<i32>) -> !option.option<i32> {
  %1 = option.and_then %arg0 : !option.option<i32> -> !option.option<i32> {
  ^bb0(%unwrapped : i32):
    %2 = option.none : !option.option<i32>
    option.yield %2 : !option.option<i32>
  }
  return %1 : !option.option<i32>
}

func.func @test_and_then_some(%arg0 : !option.option<i32>) -> !option.option<i32> {
  %1 = option.and_then %arg0 : !option.option<i32> -> !option.option<i32> {
  ^bb0(%unwrapped : i32):
    %2 = option.some %unwrapped : !option.option<i32>
    option.yield %2 : !option.option<i32>
  }
  return %1 : !option.option<i32>
}

func.func @test_and_then_some_f32(%arg0 : !option.option<i32>) -> !option.option<f32> {
  %1 = option.and_then %arg0 : !option.option<i32> -> !option.option<f32> {
  ^bb0(%unwrapped : i32):
    %1 = arith.sitofp %unwrapped : i32 to f32
    %2 = option.some %1 : !option.option<f32>
    option.yield %2 : !option.option<f32>
  }
  return %1 : !option.option<f32>
}
