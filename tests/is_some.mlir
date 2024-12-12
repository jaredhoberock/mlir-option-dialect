// RUN: opt %s | FileCheck %s

// CHECK-LABEL: func @test_is_some
// CHECK: %{{.+}} = option.is_some %arg0
func.func @test_is_some(%arg0: !option.option<i32>) -> i1 {
  %result = option.is_some %arg0 : !option.option<i32>
  return %result : i1
}
