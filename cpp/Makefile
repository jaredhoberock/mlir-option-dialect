LLVM_BIN_PATH = $(HOME)/dev/git/llvm-project-20/install/bin

LLVM_CONFIG := $(LLVM_BIN_PATH)/llvm-config
TBLGEN := $(LLVM_BIN_PATH)/mlir-tblgen
OPT := $(LLVM_BIN_PATH)/mlir-opt

# Compiler flags
CXX := clang++
CXXFLAGS := -g -fPIC `$(LLVM_CONFIG) --cxxflags`

# LLVM/MLIR libraries
MLIR_INCLUDE = /home/jhoberock/dev/git/llvm-project-20/install/include

SUM_DIALECT_INCLUDE = $(HOME)/dev/git/mlir-sum-dialect/cpp
SUM_DIALECT_LIB = $(HOME)/dev/git/mlir-sum-dialect/cpp

INCLUDES := -I $(MLIR_INCLUDE) -I $(SUM_DIALECT_INCLUDE)

# Dialect library sources (everything except main)
DIALECT_SOURCES := c_api.cpp Canonicalization.cpp ConvertToSum.cpp Option.cpp OptionOps.cpp OptionTypes.cpp
DIALECT_OBJECTS := $(DIALECT_SOURCES:.cpp=.o)

# Generated files
GENERATED := Option.hpp.inc Option.cpp.inc OptionOps.hpp.inc OptionOps.cpp.inc OptionTypes.hpp.inc OptionTypes.cpp.inc

.PHONY: all clean

all: liboption_dialect.a liboption_dialect.so

# TableGen rules
Option.hpp.inc: Option.td
	$(TBLGEN) --gen-dialect-decls $(INCLUDES) $< -o $@

Option.cpp.inc: Option.td
	$(TBLGEN) --gen-dialect-defs $(INCLUDES) $< -o $@

OptionOps.hpp.inc: OptionOps.td
	$(TBLGEN) --gen-op-decls $(INCLUDES) $< -o $@

OptionOps.cpp.inc: OptionOps.td
	$(TBLGEN) --gen-op-defs $(INCLUDES) $< -o $@

OptionTypes.hpp.inc: OptionTypes.td $(SUM_DIALECT_INCLUDE)/SumTypeInterface.td
	$(TBLGEN) --gen-typedef-decls $(INCLUDES) $< -o $@

OptionTypes.cpp.inc: OptionTypes.td $(SUM_DIALECT_INCLUDE)/SumTypeInterface.td
	$(TBLGEN) --gen-typedef-defs $(INCLUDES) $< -o $@

# Object file rules
%.o: %.cpp $(GENERATED)
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@

liboption_dialect.a: $(DIALECT_OBJECTS)
	ar rcs $@ $^

PLUGIN_OBJECTS := $(DIALECT_OBJECTS) Plugin.o

liboption_dialect.so: $(PLUGIN_OBJECTS) $(SUM_DIALECT_LIB)/libsum_dialect.a
	$(CXX) -shared $(PLUGIN_OBJECTS) $(SUM_DIALECT_LIB)/libsum_dialect.a -o $@

.PHONY: test
test: liboption_dialect.so
	@echo "Running option dialect tests..."
	lit tests

clean:
	rm -f *.o *.inc *.a *.so
