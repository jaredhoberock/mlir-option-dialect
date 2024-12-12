// RUN: opt %s -canonicalize -split-input-file | FileCheck %s

// CHECK-LABEL: func @test_unwrap_or_some_fold
func.func @test_unwrap_or_some_fold(%opt: !option.option<i32>, %default: i32) -> !option.option<i32> {
 // CHECK-NOT: unwrap_or
 // CHECK-NOT: option.some
 // CHECK: return %arg0
 %unwrapped = option.unwrap_or %opt, %default : i32
 %result = option.some %unwrapped : !option.option<i32>
 return %result : !option.option<i32>
}

// -----

// CHECK-LABEL: func @test_unwrap_or_some_no_fold
func.func @test_unwrap_or_some_no_fold(%opt: !option.option<i32>, %default: i32) -> !option.option<i32> {
 %unwrapped = option.unwrap_or %opt, %default : i32
 // CHECK: %[[ADD:.*]] = arith.addi
 // CHECK: option.some %[[ADD]]
 %plus_one = arith.addi %unwrapped, %unwrapped : i32  // transforms the value
 %result = option.some %plus_one : !option.option<i32>
 return %result : !option.option<i32>
}
