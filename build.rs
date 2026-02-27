use std::env;
use std::path::PathBuf;
use std::process::Command;

fn main() {
    // Get absolute path to the crate root
    let crate_dir = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap());

    // Path to cpp directory
    let cpp_dir = crate_dir.join("cpp");

    // Always run make in cpp/
    let status = Command::new("make")
        .arg("-j")
        .current_dir(&cpp_dir)
        .status()
        .expect("Failed to run make in cpp/");

    if !status.success() {
        panic!("C++ build failed");
    }

    // Link against the static library in cpp/
    println!("cargo:rustc-link-search=native={}", cpp_dir.display());
    println!("cargo:rustc-link-lib=static=option_dialect");

    // Also link the sum dialect (option dialect depends on it)
    let sum_dialect_dir = crate_dir.join("../mlir-sum-dialect/cpp");
    println!("cargo:rustc-link-search=native={}", sum_dialect_dir.display());
    println!("cargo:rustc-link-lib=static=sum_dialect");

    // Ensure rebuild if anything in cpp/ changes
    println!("cargo:rerun-if-changed=cpp/");
}
