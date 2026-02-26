// RUN: mlir-opt %s | FileCheck %s

// CHECK-LABEL: func @test_none_i32
// CHECK: %{{.+}} = option.none
func.func @test_none_i32() -> !option.option<i32> {
  %0 = option.none : !option.option<i32>
  return %0 : !option.option<i32>
}

// CHECK-LABEL: func @test_none_f32
// CHECK: %{{.+}} = option.none
func.func @test_none_f32() -> !option.option<f32> {
  %0 = option.none : !option.option<f32>
  return %0 : !option.option<f32>
}
