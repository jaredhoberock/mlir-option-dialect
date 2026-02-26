LLVM_BIN_PATH = /home/jhoberock/dev/git/llvm-project-20/install/bin

LLVM_CONFIG := $(LLVM_BIN_PATH)/llvm-config
TBLGEN := $(LLVM_BIN_PATH)/mlir-tblgen
OPT := $(LLVM_BIN_PATH)/mlir-opt

# Compiler flags
CXX := clang++
CXXFLAGS := -g -fPIC `$(LLVM_CONFIG) --cxxflags`

# LLVM/MLIR libraries
MLIR_INCLUDE = /home/jhoberock/dev/git/llvm-project-20/install/include

INCLUDES := -I $(MLIR_INCLUDE)

# Dialect library sources (everything except main)
DIALECT_SOURCES := c_api.cpp Canonicalization.cpp Option.cpp Lowering.cpp OptionOps.cpp OptionTypes.cpp
DIALECT_OBJECTS := $(DIALECT_SOURCES:.cpp=.o)

# Generated files
GENERATED := Option.hpp.inc Option.cpp.inc OptionOps.hpp.inc OptionOps.cpp.inc OptionTypes.hpp.inc OptionTypes.cpp.inc

.PHONY: all clean

all: liboption_dialect.so

# TableGen rules
Option.hpp.inc: Option.td
	$(TBLGEN) --gen-dialect-decls $(INCLUDES) $< -o $@

Option.cpp.inc: Option.td
	$(TBLGEN) --gen-dialect-defs $(INCLUDES) $< -o $@

OptionOps.hpp.inc: OptionOps.td
	$(TBLGEN) --gen-op-decls $(INCLUDES) $< -o $@

OptionOps.cpp.inc: OptionOps.td
	$(TBLGEN) --gen-op-defs $(INCLUDES) $< -o $@

OptionTypes.hpp.inc: OptionTypes.td
	$(TBLGEN) --gen-typedef-decls $(INCLUDES) $< -o $@

OptionTypes.cpp.inc: OptionTypes.td
	$(TBLGEN) --gen-typedef-defs $(INCLUDES) $< -o $@

# Object file rules
%.o: %.cpp $(GENERATED)
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@

PLUGIN_OBJECTS := $(DIALECT_OBJECTS) Plugin.o

liboption_dialect.so: $(PLUGIN_OBJECTS)
	$(CXX) -shared $^ -o $@

.PHONY: test
test: liboption_dialect.so
	@echo "Running option dialect tests..."
	./venv/bin/lit tests

clean:
	rm -f *.o *.inc *.a *.so
