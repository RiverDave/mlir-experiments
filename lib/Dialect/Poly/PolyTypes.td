#ifndef LIB_DIALECT_POLY_POLYTYPES_TD_
#define LIB_DIALECT_POLY_POLYTYPES_TD_

include "PolyDialect.td"
include "mlir/IR/AttrTypeBase.td"

// A base class for all types in this dialect
// As a general rule it is recommended that all types in a Dialect are inherited
// From a Base class type
class Poly_Type<string name, string typeMnemonic> : TypeDef<Poly_Dialect, name> {
  let mnemonic = typeMnemonic;
}

def Polynomial : Poly_Type<"Polynomial", "poly"> {
  let summary = "A polynomial with u32 coefficients";

  let description = [{
    A type for polynomials with integer coefficients in a single-variable polynomial ring.
  }];
}

#endif  // LIB_DIALECT_POLY_POLYTYPES_TD_