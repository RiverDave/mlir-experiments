// RUN: tutorial-opt %s --mul-to-add > %t
// RUN: FileCheck %s < %t

func.func @just_power_of_two(%arg: i32) -> i32 {
  %0 = arith.constant 8 : i32
  %1 = arith.muli %arg, %0 : i32
  func.return %1 : i32
}

// CHECK-LABEL: func.func @just_power_of_two(
// CHECK-SAME:    %[[ARG:.*]]: i32
// CHECK-SAME:  ) -> i32 {
// CHECK:     %[[CST:.*]] = arith.constant 3 : i32
// CHECK:     %[[SHL:.*]] = arith.shli %arg0, %[[CST]] : i32
// CHECK:     return %[[SHL]] : i32

func.func @power_of_two_plus_one(%arg: i32) -> i32 {
  %0 = arith.constant 9 : i32
  %1 = arith.muli %arg, %0 : i32
  func.return %1 : i32
}

// CHECK-LABEL: func.func @power_of_two_plus_one(
// CHECK-SAME:    %[[ARG:.*]]: i32
// CHECK-SAME:  ) -> i32 {
// CHECK:     %[[CST:.*]] = arith.constant 3 : i32
// CHECK:     %[[SHL:.*]] = arith.shli %arg0, %[[CST]] : i32
// CHECK:     %[[ADDT:.*]] = arith.addi %[[SHL]], %arg0 : i32
// CHECK:   return %[[ADDT]] : i32
// CHECK: }
