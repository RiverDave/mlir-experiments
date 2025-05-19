//cf -> llvm dialect Transformation
module {
  func.func @select(%arg0: i32, %arg1: i32, %arg2: i1) -> i32 {
    llvm.cond_br %arg2, ^bb1(%arg0 : i32), ^bb1(%arg1 : i32)
  ^bb1(%0: i32):  // 2 preds: ^bb0, ^bb0
    return %0 : i32
  }
}

