/*
 The MIT License (MIT)

 Copyright (c) 2017 Iurii Zykov

 Permission is hereby granted, free of charge, to any person obtaining a copy
 of this software and associated documentation files (the "Software"), to deal
 in the Software without restriction, including without limitation the rights
 to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 copies of the Software, and to permit persons to whom the Software is
 furnished to do so, subject to the following conditions:

 The above copyright notice and this permission notice shall be included in all
 copies or substantial portions of the Software.

 THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
 THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
 IN THE SOFTWARE.
*/


#if !defined(MAIN_H)
#define MAIN_H


#include <mpi.h>

#include <stddef.h>


#define COLLAPSE(x) collapse(x)

#define INDEXTYPE unsigned int
#define BASETYPE double
#define MPI_BASETYPE MPI_DOUBLE

#define STR(x) #x
#define VARIANT(v) STR(variants/v##.h)

#define NDIMS 2
#define X 0
#define Y 1


typedef struct Context {
  // problem params (from <variant>.h)
  INDEXTYPE N1;
  INDEXTYPE N2;
  BASETYPE A1;
  BASETYPE A2;
  BASETYPE B1;
  BASETYPE B2;
  BASETYPE q;
  BASETYPE eps;
  
  // environment
  MPI_Comm comm;
  int rank;
  int size;
  int dims[NDIMS];
  int coords[NDIMS];
  int up, down, left, right;
  
  // block params, in global and local coords (latter used for effective 
  // handling of borders)
  INDEXTYPE blockSize[NDIMS];
  INDEXTYPE offsets[NDIMS];
  INDEXTYPE begin[NDIMS];
  INDEXTYPE end[NDIMS];
  
  // computation data
  BASETYPE* grid[NDIMS];
  BASETYPE* h[NDIMS];
  BASETYPE* mh[NDIMS];
  BASETYPE* shadow[NDIMS];  // storage for columns to send
  BASETYPE* shadow_copy[NDIMS];  // storage for columns to receive
  BASETYPE* prod_coeff;
  BASETYPE* f;  // F-values
  BASETYPE* p;  // current approximation (p-function in <variant>.h is guessed)
  BASETYPE* r;  // residual
  // temp values for conjugate gradient method (iteration interconnection)
  BASETYPE* g;  
  
  // GPU data
  // all pointers at host are device pointers
  struct Context* device_ctx;  // self-reference for device 
  INDEXTYPE blockDim, gridDim, threadDim;  // kernel params
  BASETYPE* shadows[NDIMS][NDIMS * NDIMS];  // mapped storages for send/receive
  double* reduction;  // temp storage, general purpose
  
  // algorithm control values
  INDEXTYPE iteration;
  double current_eps;
} Context;


#endif
