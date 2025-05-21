import os
import lit.formats

config.test_format = lit.formats.ShTest(execute_external=True)

# Name of your test suite
config.name = 'MLIRTests'



# Test file suffixes
config.suffixes = ['.mlir']
#FIXME: This is wrong
# llvm_project_root_binaries = os.path.abspath(os.path.join(config.test_source_root, '/../externals/llvm-project/build/bin'))

# Substitutions for RUN lines
config.test_source_root = os.path.dirname(__file__)
#mlir_opt = os.path.abspath(os.path.join(config.test_source_root, '/../mlir-opt'))
# Not needed for now, Let's use the one in the PATH
#file_check = os.path.abspath(os.path.join(llvm_project_root_binaries, '/FileCheck'))
file_check = "FileCheck"
tutorial_opt = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'tutorial-opt'))
tutorial_opt_dir = os.path.dirname(tutorial_opt)


config.substitutions.append(('%mlir_opt', "mlir-opt"))
config.substitutions.append(('%FileCheck', file_check))
config.substitutions.append(('%tutorial-opt', tutorial_opt))

# Environment
config.environment['PATH'] = tutorial_opt_dir + os.pathsep + os.environ['PATH']