// RUN: opt %s | FileCheck %s

// CHECK-LABEL: func @test_some
// CHECK: %{{.+}} = option.some %arg0
func.func @test_some(%arg0 : i32) -> !option.option<i32> {
  // Create a Some value
  %some = option.some %arg0 : !option.option<i32>
  return %some : !option.option<i32>
}
