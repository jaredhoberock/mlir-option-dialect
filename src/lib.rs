use melior::Context;
use mlir_sys::MlirContext;

#[link(name = "option_dialect")]
unsafe extern "C" {
    fn optionRegisterDialect(ctx: MlirContext);
}

pub fn register(context: &Context) {
    unsafe { optionRegisterDialect(context.to_raw()) }
}
