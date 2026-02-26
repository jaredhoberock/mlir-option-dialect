#pragma once

#include "OptionTypes.hpp"
#include <mlir/Bytecode/BytecodeOpInterface.h>
#include <mlir/Interfaces/CallInterfaces.h>
#include <mlir/Interfaces/ControlFlowInterfaces.h>
#include <mlir/Interfaces/FunctionInterfaces.h>
#include <mlir/Interfaces/SideEffectInterfaces.h>
#include <mlir/IR/BuiltinTypes.h>
#include <mlir/IR/Dialect.h>
#include <mlir/IR/OpDefinition.h>

#define GET_OP_CLASSES
#include "OptionOps.hpp.inc"
