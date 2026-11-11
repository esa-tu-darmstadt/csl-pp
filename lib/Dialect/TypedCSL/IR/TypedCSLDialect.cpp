//===- TypedCSLDialect.cpp - TypedCSL Dialect ------------------*- C++ -*-===//
//
// Implementation of the TypedCSL dialect.
//
//===----------------------------------------------------------------------===//

#include "csl-mlir/Dialect/CSL/IR/CSL.h"
#include "csl-mlir/Dialect/TypedCSL/IR/TypedCSL.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/OpImplementation.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/ErrorHandling.h"

using namespace mlir;
using namespace mlir::typed_csl;

#define DEBUG_TYPE "typedcsl-dialect"

//===----------------------------------------------------------------------===//
// TypedCSL Dialect
//===----------------------------------------------------------------------===//

#include "csl-mlir/Dialect/TypedCSL/IR/TypedCSLDialect.cpp.inc"

// Note: Enum definitions (BinaryOpKind, FloatKind, Direction) are now in CSL
// dialect

void TypedCSLDialect::initialize() {
  registerTypes();
  registerAttributes();
  addOperations<
#define GET_OP_LIST
#include "csl-mlir/Dialect/TypedCSL/IR/TypedCSLOps.cpp.inc"
      >();
}

void TypedCSLDialect::registerTypes() {
  addTypes<
#define GET_TYPEDEF_LIST
#include "csl-mlir/Dialect/TypedCSL/IR/TypedCSLTypes.cpp.inc"
      >();
}

void TypedCSLDialect::registerAttributes() {
  addAttributes<
#define GET_ATTRDEF_LIST
#include "csl-mlir/Dialect/TypedCSL/IR/TypedCSLAttrs.cpp.inc"
      >();
}

// Type parsing and printing are generated automatically by tablegen

//===----------------------------------------------------------------------===//
// Constant Materializer
//===----------------------------------------------------------------------===//

Operation *TypedCSLDialect::materializeConstant(OpBuilder &builder,
                                                Attribute value, Type type,
                                                Location loc) {
  // Check if the attribute is a TypedAttr that can be used with ConstantOp
  if (auto typedAttr = mlir::dyn_cast<TypedAttr>(value)) {
    // Verify the attribute's type matches the requested type
    if (typedAttr.getType() != type) {
      emitError(loc) << "TypedCSLDialect::materializeConstant: attribute type "
                     << typedAttr.getType() << " does not match requested type "
                     << type;
      return nullptr;
    }
    return ConstantOp::create(builder, loc, typedAttr);
  }

  LLVM_DEBUG({
    llvm::dbgs() << "TypedCSLDialect::materializeConstant failed for value: "
                 << value << " and type: " << type << "\n";
  });

  return nullptr;
}

//===----------------------------------------------------------------------===//
// Type Definitions
//===----------------------------------------------------------------------===//

#define GET_TYPEDEF_CLASSES
#include "csl-mlir/Dialect/TypedCSL/IR/TypedCSLTypes.cpp.inc"

void TypedCSLDialect::printType(::mlir::Type type,
                                ::mlir::DialectAsmPrinter &printer) const {
  llvm_unreachable("TypedCSLDialect::printType should not be called");
}

::mlir::Type
TypedCSLDialect::parseType(::mlir::DialectAsmParser &parser) const {
  return {};
}

//===----------------------------------------------------------------------===//
// Attribute Definitions
//===----------------------------------------------------------------------===//

#define GET_ATTRDEF_CLASSES
#include "csl-mlir/Dialect/TypedCSL/IR/TypedCSLAttrs.cpp.inc"

void TypedCSLDialect::printAttribute(::mlir::Attribute attr,
                                     ::mlir::DialectAsmPrinter &printer) const {
  llvm_unreachable("TypedCSLDialect::printAttribute should not be called");
}

::mlir::Attribute
TypedCSLDialect::parseAttribute(::mlir::DialectAsmParser &parser,
                                ::mlir::Type type) const {
  return {};
}

//===----------------------------------------------------------------------===//
// Helper Functions
//===----------------------------------------------------------------------===//

namespace mlir {
namespace typed_csl {

// Type constraint implementations
#include "csl-mlir/Dialect/TypedCSL/IR/TypedCSLTypeConstraints.cpp.inc"

} // namespace typed_csl
} // namespace mlir
