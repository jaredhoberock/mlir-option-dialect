// RUN: mlir-opt %s --canonicalize --split-input-file | FileCheck %s

// CHECK-LABEL: func @test_is_some_some_fold
func.func @test_is_some_some_fold(%val: i32) -> i1 {
  // CHECK-NOT: option.some
  // CHECK-NOT: option.is_some
  // CHECK: %true = arith.constant true
  // CHECK: return %true
  %some = option.some %val : !option.option<i32>
  %result = option.is_some %some : !option.option<i32>
  return %result : i1
}

// -----

// CHECK-LABEL: func @test_is_some_none_fold
func.func @test_is_some_none_fold() -> i1 {
  // CHECK-NOT: option.none
  // CHECK-NOT: option.is_some
  // CHECK: %false = arith.constant false
  // CHECK: return %false
  %none = option.none : !option.option<i32>
  %result = option.is_some %none : !option.option<i32>
  return %result : i1
}
