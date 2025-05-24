// RUN: tutorial-opt %s --affine-full-unroll > %t
// RUN: FileCheck %s < %t

//The following op should intentionally fail:

func.func @test_single_nested_loop(%buffer: memref<4xi32>, %arg_1: index, %arg_2: index) -> (i32) {
  %sum_0 = arith.constant 0 : i32

  //Iterations will be done depending our two args
  %upper_bound = arith.maxui %arg_1 , %arg_2 : index

  // CHECK: affine.for
  %sum = affine.for %i = 0 to %upper_bound iter_args(%sum_iter = %sum_0) -> i32 {
    %t = affine.load %buffer[%i] : memref<4xi32>
    %sum_next = arith.addi %sum_iter, %t : i32

    affine.yield %sum_next : i32
  }
  return %sum : i32
}