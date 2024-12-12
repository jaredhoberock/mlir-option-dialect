// RUN: opt %s -canonicalize -split-input-file | FileCheck %s

// CHECK-LABEL: func @test_and_then_identity_fold
func.func @test_and_then_identity_fold() -> !option.option<i32> {
  // CHECK: %[[CST:.*]] = arith.constant 42 : i32
  %value = arith.constant 42 : i32
  // CHECK: %[[SOME:.*]] = option.some %[[CST]]
  %some = option.some %value : !option.option<i32>
  // CHECK-NOT: option.and_then
  // CHECK: return %[[SOME]]
  %result = option.and_then %some : !option.option<i32> -> !option.option<i32> {
  ^bb0(%arg0: i32):
    %wrapped = option.some %arg0 : !option.option<i32>
    option.yield %wrapped : !option.option<i32>
  }
  return %result : !option.option<i32>
}

// -----

// CHECK-LABEL: func @test_and_then_no_fold_transform
func.func @test_and_then_no_fold_transform() -> !option.option<i32> {
 %value = arith.constant 42 : i32
 %some = option.some %value : !option.option<i32>
 // CHECK: option.and_then
 %result = option.and_then %some : !option.option<i32> -> !option.option<i32> {
 ^bb0(%arg0: i32):
   %plus_one = arith.addi %arg0, %arg0 : i32  // actual transformation
   %wrapped = option.some %plus_one : !option.option<i32>
   option.yield %wrapped : !option.option<i32>
 }
 return %result : !option.option<i32>
}
