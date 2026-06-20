import os
import lit.formats

config.name = "Option Dialect Tests"
config.test_format = lit.formats.ShTest(True)
config.suffixes = ['.mlir']
config.test_source_root = os.path.dirname(__file__)

mlir_prefix = os.environ.get('MLIR_SYS_220_PREFIX', '/home/jhoberock/dev/git/llvm-project-22/install-release-asserts')
llvm_bin = os.path.join(mlir_prefix, 'bin')
fallback_llvm_bin = '/home/jhoberock/dev/git/llvm-project-22/build/bin'

def tool(name):
    installed = os.path.join(llvm_bin, name)
    if os.path.exists(installed):
        return installed
    return os.path.join(fallback_llvm_bin, name)

sum_plugin = os.environ.get(
    'SUM_DIALECT_PLUGIN',
    '/home/jhoberock/dev/git/mlir-sum-dialect/cpp/build/libsum_dialect.so',
)
option_plugin = os.environ.get(
    'OPTION_DIALECT_PLUGIN',
    os.path.join(os.path.dirname(__file__), '..', 'build', 'liboption_dialect.so'),
)

config.substitutions.append(('mlir-opt', f'{tool("mlir-opt")} --load-dialect-plugin={sum_plugin} --load-dialect-plugin={option_plugin}'))
config.substitutions.append(('FileCheck', tool('FileCheck')))
