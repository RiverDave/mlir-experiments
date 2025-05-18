func.func @select(%a: i32, %b: i32, %flag: i1) -> i32 {
  // Both targets are the same, operands differ
  cf.cond_br %flag, ^bb1(%a : i32), ^bb1(%b : i32)

^bb1(%x : i32) :
  return %x : i32
}
