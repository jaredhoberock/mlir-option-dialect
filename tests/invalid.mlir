// RUN: opt %s -split-input-file -verify-diagnostics

// expected-note@+1 {{prior use here}}
func.func @some_mismatch(%arg0 : i32) {
  // expected-error@+1 {{use of value '%arg0' expects different type than prior uses: 'f32' vs 'i32'}}
  %0 = option.some %arg0 : !option.option<f32>
  return
}

// -----

// expected-note@+1 {{prior use here}}
func.func @and_then_incompatible_result(%arg0 : !option.option<i32>) {
  %0 = option.and_then %arg0 : !option.option<i32> -> !option.option<f32> {
    // expected-error@+1 {{use of value '%arg0' expects different type than prior uses: 'i32' vs '!option.option<i32>'}}
    %1 = option.some %arg0 : !option.option<i32>
    option.yield %1 : !option.option<i32>
  }
  return
}
