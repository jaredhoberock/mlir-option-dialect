// RUN: mlir-opt %s | FileCheck %s

// CHECK-LABEL: func @test_unwrap_or_none
// CHECK: %[[NONE:.*]] = option.none
// CHECK: %[[RESULT:.*]] = option.unwrap_or %[[NONE]], %arg0
func.func @test_unwrap_or_none(%arg0 : i32) -> i32 {
  %0 = option.none : !option.option<i32>
  %1 = option.unwrap_or %0, %arg0 : i32
  return %1 : i32
}

// CHECK-LABEL: func @test_unwrap_or_some
// CHECK: %[[SOME:.*]] = option.some %arg0
// CHECK: %[[RESULT:.*]] = option.unwrap_or %[[SOME]], %arg0
func.func @test_unwrap_or_some(%arg0 : i32) -> i32 {
  %0 = option.some %arg0 : !option.option<i32>
  %1 = option.unwrap_or %0, %arg0 : i32
  return %1 : i32
}
