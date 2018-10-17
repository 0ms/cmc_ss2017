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


#if !defined(VARIANT_H)
#define VARIANT_H


#include "../main.h"

#include <math.h>


#if defined(__CUDA_ARCH__)
__device__
#endif
inline BASETYPE F(BASETYPE x, BASETYPE y) {
  return 2 * (x * x + y * y) * (1 - 2 * x * x * y * y) * exp(1 - x * x * y * y);
}

#if defined(__CUDA_ARCH__)
__device__
#endif
inline BASETYPE fi(BASETYPE x, BASETYPE y) {
  return exp(1 - x * x * y * y);
}

inline BASETYPE p(BASETYPE x, BASETYPE y) {
  return fi(x, y);
}

#if defined(__CUDA_ARCH__)
__device__
#endif
inline BASETYPE f(BASETYPE t, BASETYPE q) {
  return (pow(1 + t, q) - 1) / (pow((BASETYPE)2, q) - 1);
}

#if defined(__CUDA_ARCH__)
__device__
#endif
inline void initProblemParams(Context* ctx, int N1, int N2) {
  ctx->N1 = N1;
  ctx->N2 = N2;
  ctx->A1 = -2;
  ctx->A2 = 2;
  ctx->B1 = -2;
  ctx->B2 = 2;
  ctx->q = 1.0;
  ctx->eps = 1e-4;
  return;
}


#if defined(__CUDA_ARCH__)
__device__
#endif
inline void _norm(double* lhs, double rhs) {
  *lhs += rhs;
  return;
}

#if defined(__CUDA_ARCH__)
__device__
#endif
inline void norm(Context* ctx, double* lhs,
    INDEXTYPE i, INDEXTYPE j, double rhs) {
  _norm(lhs, ctx->mh[X][i] * ctx->mh[Y][j] * rhs);
  return;
}


#endif
