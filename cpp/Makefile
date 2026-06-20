ifndef MLIR_SYS_220_PREFIX
$(error MLIR_SYS_220_PREFIX must be set; point it at your LLVM/MLIR 22 install)
endif

BUILD_DIR := build
SUM_DIALECT_SOURCE_DIR := ../../mlir-sum-dialect/cpp
SUM_DIALECT_INCLUDE_DIR := ../../mlir-sum-dialect/cpp/build
LLVM_BIN_PATH := $(MLIR_SYS_220_PREFIX)/bin
LLVM_CONFIG := $(LLVM_BIN_PATH)/llvm-config
TBLGEN := $(LLVM_BIN_PATH)/mlir-tblgen

CXX := clang++
CXXFLAGS := -fPIC $(shell $(LLVM_CONFIG) --cxxflags)
INCLUDES := -I$(BUILD_DIR) -I. -I$(MLIR_SYS_220_PREFIX)/include -I$(SUM_DIALECT_SOURCE_DIR) -I$(SUM_DIALECT_INCLUDE_DIR)

DIALECT_SOURCES := c_api.cpp Canonicalization.cpp ConvertToSum.cpp Option.cpp OptionOps.cpp OptionTypes.cpp
DIALECT_OBJECTS := $(addprefix $(BUILD_DIR)/,$(DIALECT_SOURCES:.cpp=.o))
PLUGIN_OBJECTS := $(DIALECT_OBJECTS) $(BUILD_DIR)/Plugin.o
TD_FILES := $(wildcard *.td) $(SUM_DIALECT_SOURCE_DIR)/SumTypeInterface.td

GENERATED := \
  $(BUILD_DIR)/Option.hpp.inc \
  $(BUILD_DIR)/Option.cpp.inc \
  $(BUILD_DIR)/OptionOps.hpp.inc \
  $(BUILD_DIR)/OptionOps.cpp.inc \
  $(BUILD_DIR)/OptionTypes.hpp.inc \
  $(BUILD_DIR)/OptionTypes.cpp.inc

.PHONY: all clean test

all: $(BUILD_DIR)/liboption_dialect.a $(BUILD_DIR)/liboption_dialect.so

$(BUILD_DIR):
	mkdir -p $@

$(BUILD_DIR)/Option.hpp.inc: Option.td $(TD_FILES) | $(BUILD_DIR)
	$(TBLGEN) --gen-dialect-decls $(INCLUDES) $< -o $@

$(BUILD_DIR)/Option.cpp.inc: Option.td $(TD_FILES) | $(BUILD_DIR)
	$(TBLGEN) --gen-dialect-defs $(INCLUDES) $< -o $@

$(BUILD_DIR)/OptionOps.hpp.inc: OptionOps.td $(TD_FILES) | $(BUILD_DIR)
	$(TBLGEN) --gen-op-decls $(INCLUDES) $< -o $@

$(BUILD_DIR)/OptionOps.cpp.inc: OptionOps.td $(TD_FILES) | $(BUILD_DIR)
	$(TBLGEN) --gen-op-defs $(INCLUDES) $< -o $@

$(BUILD_DIR)/OptionTypes.hpp.inc: OptionTypes.td $(TD_FILES) | $(BUILD_DIR)
	$(TBLGEN) --gen-typedef-decls $(INCLUDES) $< -o $@

$(BUILD_DIR)/OptionTypes.cpp.inc: OptionTypes.td $(TD_FILES) | $(BUILD_DIR)
	$(TBLGEN) --gen-typedef-defs $(INCLUDES) $< -o $@

$(BUILD_DIR)/%.o: %.cpp $(GENERATED) | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@

$(BUILD_DIR)/liboption_dialect.a: $(DIALECT_OBJECTS)
	ar rcs $@ $^

$(BUILD_DIR)/liboption_dialect.so: $(PLUGIN_OBJECTS)
	$(CXX) -shared $^ -o $@

test: $(BUILD_DIR)/liboption_dialect.so
	OPTION_DIALECT_PLUGIN=$(CURDIR)/$(BUILD_DIR)/liboption_dialect.so lit tests

clean:
	rm -rf $(BUILD_DIR)
