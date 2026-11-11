//===- NominalTypeRegistry.cpp - CSL emitted type registry ---------------===//
//
// Implementation of generated type declarations for CSL emission.
//
//===----------------------------------------------------------------------===//

#include "csl-mlir/Dialect/CSL/IR/NominalTypeRegistry.h"
#include "csl-mlir/Dialect/CSL/IR/CSL.h"

using namespace mlir;
using namespace mlir::csl;

void NominalTypeRegistry::collectTypes(CompilationUnitOp unit) {
  unit.walk([&](Operation *op) {
    // Walk result types
    for (Type type : op->getResultTypes()) {
      collectFromType(type);
    }
    // Walk operand types
    for (Type type : op->getOperandTypes()) {
      collectFromType(type);
    }
    // Walk attribute types
    for (NamedAttribute attr : op->getAttrs()) {
      collectFromAttr(attr.getValue());
    }
  });
}

std::optional<StringRef> NominalTypeRegistry::getName(Type type) const {
  auto it = typeToName.find(type);
  if (it != typeToName.end())
    return StringRef(it->second);
  return std::nullopt;
}

void NominalTypeRegistry::emitDefinitions(raw_ostream &os) {
  for (Type type : typesInOrder) {
    os << "const ";
    if (auto enumType = mlir::dyn_cast<EnumType>(type)) {
      os << *getName(enumType);
      os << " = enum(";
      csl::printType(os, enumType.getUnderlyingType());
      os << ") { ";
      auto names = enumType.getElementNames();
      auto values = enumType.getElementValues();
      for (size_t i = 0; i < names.size(); ++i) {
        if (i > 0)
          os << ", ";
        os << names[i].getValue();
        int64_t expectedValue = (i == 0) ? 0 : values[i - 1] + 1;
        if (values[i] != expectedValue) {
          os << " = " << values[i];
        }
      }
      os << " }";
    } else if (auto structType = mlir::dyn_cast<StructType>(type)) {
      os << *getName(structType);
      os << " = struct { ";
      for (size_t i = 0; i < structType.getNumFields(); ++i) {
        if (i > 0)
          os << ", ";
        auto fieldName = structType.getFieldName(i);
        if (succeeded(fieldName)) {
          os << *fieldName << ": ";
        }
        csl::printType(os, structType.getFieldType(i), this);
      }
      os << " }";
    }
    os << ";\n";
  }
  if (!typesInOrder.empty()) {
    os << "\n";
  }
}

void NominalTypeRegistry::collectFromType(Type type) {
  SmallVector<Type> nestedTypes;
  type.walk([&](Type subType) { nestedTypes.push_back(subType); });

  // Type::walk visits children before their containers, so generated
  // declarations only refer to types which have already been declared.
  for (Type subType : nestedTypes) {
    if (mlir::isa<EnumType>(subType)) {
      registerType(subType);
    } else if (auto structType = mlir::dyn_cast<StructType>(subType)) {
      if (structType.getDefLocation() || !structType.isTuple())
        registerType(structType);
    }
  }
}

void NominalTypeRegistry::collectFromAttr(Attribute attr) {
  attr.walk([&](Type type) { collectFromType(type); });
}

void NominalTypeRegistry::registerType(Type type) {
  if (typeToName.contains(type))
    return;

  std::string name;
  if (mlir::isa<EnumType>(type)) {
    name = "__enum_" + std::to_string(enumCounter++);
  } else {
    name = "__struct_" + std::to_string(structCounter++);
  }

  typeToName[type] = name;
  typesInOrder.push_back(type);
}
