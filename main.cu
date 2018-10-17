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


#include "main.h"
#include VARIANT(14)

#include <cuda.h>
#include <cuda_runtime.h>


extern "C"
void print(Context* ctx, const char* format, ...);

extern "C"
void __checkMPIErrors(int err, const char* file, const char* func,
  const int line);
  
#define checkMPIErrors(err)  __checkMPIErrors(err, __FILE__, __func__, __LINE__)


// five-point stencil
// requires m of size (ctx->blockSize[X] + 2) * (ctx->blockSize[Y] + 2)
// (fictive borders for "general" computation scheme (without explicit ifs))
__device__ inline BASETYPE _cudaLaplOp(Context* ctx, BASETYPE* m,
    INDEXTYPE i, INDEXTYPE j) {
  return ((m[(i + 1) * (ctx->blockSize[Y] + 2) + (j + 1)]
      - m[i * (ctx->blockSize[Y] + 2) + (j + 1)]) * ctx->h[X][i]
      - (m[(i + 2) * (ctx->blockSize[Y] + 2) + (j + 1)]
      - m[(i + 1) * (ctx->blockSize[Y] + 2) + (j + 1)])
        * ctx->h[X][i + 1])
     * ctx->mh[X][i]
    + ((m[(i + 1) * (ctx->blockSize[Y] + 2) + (j + 1)]
        - m[(i + 1) * (ctx->blockSize[Y] + 2) + j]) * ctx->h[Y][j]
        - (m[(i + 1) * (ctx->blockSize[Y] + 2) + (j + 2)]
        - m[(i + 1) * (ctx->blockSize[Y] + 2) + (j + 1)])
          * ctx->h[Y][j + 1])
     * ctx->mh[Y][j];
}

void __checkCudaErrors(cudaError_t err, const char* file, const char* func,
    const int line) {
  if (cudaSuccess != err) {
    printf("%s error: %s (%s:%d)\n", func, cudaGetErrorString(err), file, line);
    exit(-1);
  }
}

#define checkCudaErrors(err) \
  __checkCudaErrors(err, __FILE__, __func__, __LINE__)


__global__ void _cudaInitGrid(Context* ctx) {
  INDEXTYPE i, j;
  INDEXTYPE hSize[NDIMS] = {
    ((ctx->coords[X] == ctx->dims[X] - 1
    ? ctx->blockSize[X] - 1 : ctx->blockSize[X]) + 1),
    ((ctx->coords[Y] == ctx->dims[Y] - 1
    ? ctx->blockSize[Y] - 1 : ctx->blockSize[Y]) + 1)};
  for (i = 0; i < ctx->blockSize[X]; ++i) {
    ctx->grid[X][i] = (ctx->A2 - ctx->A1)
      * f(((BASETYPE)i + ctx->offsets[X]) / ctx->N1, ctx->q) + ctx->A1;
  }
  for (j = 0; j < ctx->blockSize[Y]; ++j) {
    ctx->grid[Y][j] = (ctx->B2 - ctx->B1)
      * f(((BASETYPE)j + ctx->offsets[Y]) / ctx->N2, ctx->q) + ctx->B1;
  }
  for (i = 0; i < ctx->blockSize[X]; ++i) {
    for (j = 0; j < ctx->blockSize[Y]; ++j) {
      ctx->f[i * ctx->blockSize[Y] + j] = -F(ctx->grid[X][i], ctx->grid[Y][j]);
    }
  }
  ctx->h[X][0] = ctx->grid[X][0] - (ctx->A2 - ctx->A1)
    * f(((BASETYPE)ctx->offsets[X] - 1) / ctx->N1, ctx->q) + ctx->A1;
  for (i = 1; i < hSize[X] - 1; ++i) {
    ctx->h[X][i] = ctx->grid[X][i] - ctx->grid[X][i - 1];
  }
  ctx->h[X][hSize[X] - 1] = (ctx->A2 - ctx->A1)
    * f(((BASETYPE)hSize[X] - 1 + ctx->offsets[X]) / ctx->N1, ctx->q) + ctx->A1
    - ctx->grid[X][hSize[X] - 2];
  
  ctx->h[Y][0] = ctx->grid[Y][0] - (ctx->B2 - ctx->B1)
    * f(((BASETYPE)ctx->offsets[Y] - 1) / ctx->N2, ctx->q) + ctx->B1;
  for (j = 1; j < hSize[Y] - 1; ++j) {
    ctx->h[Y][j] = ctx->grid[Y][j] - ctx->grid[Y][j - 1];
  }
  ctx->h[Y][hSize[Y] - 1] = (ctx->B2 - ctx->B1)
    * f(((BASETYPE)hSize[Y] - 1 + ctx->offsets[Y]) / ctx->N2, ctx->q) + ctx->B1
    - ctx->grid[Y][hSize[Y] - 2];
  
  for (i = 0; i < hSize[X] - 1; ++i) {
    ctx->mh[X][i] = 0.5 * (ctx->h[X][i + 1] + ctx->h[X][i]);
  }
  for (j = 0; j < hSize[Y] - 1; ++j) {
    ctx->mh[Y][j] = 0.5 * (ctx->h[Y][j + 1] + ctx->h[Y][j]);
  }
  for (i = 0; i < hSize[X] - 1; ++i) {
    for (j = 0; j < hSize[Y] - 1; ++j) {
      ctx->prod_coeff[i * (hSize[Y] - 1) + j] = ctx->mh[X][i] * ctx->mh[Y][j];
    }
  }
  for (i = 0; i < hSize[X] - 1; ++i) {
    ctx->mh[X][i] = 1.0 / ctx->mh[X][i];
  }
  for (j = 0; j < hSize[Y] - 1; ++j) {
    ctx->mh[Y][j] = 1.0 / ctx->mh[Y][j];
  }
  for (i = 0; i < hSize[X]; ++i) {
    ctx->h[X][i] = 1.0 / ctx->h[X][i];
  }
  for (j = 0; j < hSize[Y]; ++j) {
    ctx->h[Y][j] = 1.0 / ctx->h[Y][j];
  }
  return;
}

extern "C"
void cudaInitGrid(Context* ctx) {
  INDEXTYPE hSize[NDIMS] = {
    ((ctx->coords[X] == ctx->dims[X] - 1
    ? ctx->blockSize[X] - 1 : ctx->blockSize[X]) + 1),
    ((ctx->coords[Y] == ctx->dims[Y] - 1
    ? ctx->blockSize[Y] - 1 : ctx->blockSize[Y]) + 1)};
  checkCudaErrors(cudaMalloc(&ctx->h[X], hSize[X] * sizeof(*ctx->h[X])));
  checkCudaErrors(cudaMalloc(&ctx->h[Y], hSize[Y] * sizeof(*ctx->h[Y])));
  checkCudaErrors(cudaMalloc(&ctx->mh[X],
    (hSize[X] - 1) * sizeof(*ctx->mh[X])));
  checkCudaErrors(cudaMalloc(&ctx->mh[Y],
    (hSize[Y] - 1) * sizeof(*ctx->mh[Y])));
  checkCudaErrors(cudaMalloc(&ctx->prod_coeff,
    (hSize[X] - 1) * (hSize[Y] - 1) * sizeof(*ctx->prod_coeff)));
  checkCudaErrors(cudaMalloc(&ctx->grid[X],
    ctx->blockSize[X] * sizeof(*ctx->grid[X])));
  checkCudaErrors(cudaMalloc(&ctx->grid[Y],
    ctx->blockSize[Y] * sizeof(*ctx->grid[Y])));
  checkCudaErrors(cudaMalloc(&ctx->f,
    ctx->blockSize[X] * ctx->blockSize[Y] * sizeof(*ctx->f)));
  checkCudaErrors(cudaMemcpy(ctx->device_ctx, ctx, sizeof(*ctx),
    cudaMemcpyHostToDevice));
  
  _cudaInitGrid<<<1, 1>>>(ctx->device_ctx);
  cudaStreamSynchronize(0);
  checkCudaErrors(cudaGetLastError());
  return;
}

void cudaReleaseGrid(Context* ctx) {
  checkCudaErrors(cudaFree(ctx->h[X]));
  checkCudaErrors(cudaFree(ctx->h[Y]));
  checkCudaErrors(cudaFree(ctx->mh[X]));
  checkCudaErrors(cudaFree(ctx->mh[Y]));
  checkCudaErrors(cudaFree(ctx->prod_coeff));
  checkCudaErrors(cudaFree(ctx->f));
  return;
}

__global__ void _cudaInitComputationData(Context* ctx) {
  INDEXTYPE i, j;
  for (i = 0; i < NDIMS * NDIMS; ++i) {
    for (j = 0; j < ctx->blockSize[X]; ++j) {
      ctx->shadows[X][i][j] = 0;
    }
  }
  for (i = 0; i < NDIMS * NDIMS; ++i) {
    for (j = 0; j < ctx->blockSize[Y]; ++j) {
      ctx->shadows[Y][i][j] = 0;
    }
  }
  for (i = 0; i < ctx->blockSize[X] + 2; ++i) {
    for (j = 0; j < ctx->blockSize[Y] + 2; ++j) {
      ctx->p[i * (ctx->blockSize[Y] + 2) + j] = 0;
    }
  }
  for (i = 0; i < ctx->blockSize[X] + 2; ++i) {
    for (j = 0; j < ctx->blockSize[Y] + 2; ++j) {
      ctx->r[i * (ctx->blockSize[Y] + 2) + j] = 0;
    }
  }
  for (i = 0; i < ctx->blockSize[X] + 2; ++i) {
    for (j = 0; j < ctx->blockSize[Y] + 2; ++j) {
      ctx->g[i * (ctx->blockSize[Y] + 2) + j] = 0;
    }
  }
  return;
}

extern "C"
void cudaInitComputationData(Context* ctx) {
  BASETYPE* h_shadows[NDIMS][NDIMS * NDIMS];
  BASETYPE* h_reduction;
  INDEXTYPE i, j;
  for (i = 0; i < NDIMS; ++i) {
    for (j = 0; j < NDIMS * NDIMS; ++j) {
      checkCudaErrors(cudaHostAlloc(&h_shadows[i][j],
        ctx->blockSize[i] * sizeof(*h_shadows[i][j]),
        cudaHostAllocMapped));
      checkCudaErrors(cudaHostGetDevicePointer(&ctx->shadows[i][j],
        h_shadows[i][j], 0));
    }
  }
  checkCudaErrors(cudaHostAlloc(&h_reduction,
    2 * ctx->gridDim * sizeof(*h_reduction), cudaHostAllocMapped));
  checkCudaErrors(cudaHostGetDevicePointer(&ctx->reduction, h_reduction,
    0));
    
  checkCudaErrors(cudaMalloc(&ctx->p,
    (ctx->blockSize[X] + 2) * (ctx->blockSize[Y] + 2) * sizeof(*ctx->p)));
  checkCudaErrors(cudaMalloc(&ctx->r,
    (ctx->blockSize[X] + 2) * (ctx->blockSize[Y] + 2) * sizeof(*ctx->r)));
  checkCudaErrors(cudaMalloc(&ctx->g,
    (ctx->blockSize[X] + 2) * (ctx->blockSize[Y] + 2) * sizeof(*ctx->g)));
  checkCudaErrors(cudaMemcpy(ctx->device_ctx, ctx, sizeof(*ctx),
    cudaMemcpyHostToDevice));
  for (i = 0; i < NDIMS; ++i) {
    for (j = 0; j < NDIMS * NDIMS; ++j) {
      ctx->shadows[i][j] = h_shadows[i][j];
    }
  }
  ctx->reduction = h_reduction;
  
  _cudaInitComputationData<<<1, 1>>>(ctx->device_ctx);
  cudaStreamSynchronize(0);
  checkCudaErrors(cudaGetLastError());
  return;
}

void cudaReleaseComputationData(Context* ctx) {
  INDEXTYPE i, j;
  for (i = 0; i < NDIMS; ++i) {
    for (j = 0; j < NDIMS * NDIMS; ++j) {
      checkCudaErrors(cudaFreeHost(ctx->shadows[i][j]));
    }
  }
  checkCudaErrors(cudaFreeHost(ctx->reduction));
  checkCudaErrors(cudaFree(ctx->r));
  checkCudaErrors(cudaFree(ctx->g));
  return;
}

extern "C"
void cudaConfigure(Context* ctx) {
  struct cudaDeviceProp deviceProp;
  checkCudaErrors(cudaSetDevice(ctx->rank % 2));
  checkCudaErrors(cudaSetDeviceFlags(cudaDeviceMapHost));
  checkCudaErrors(cudaDeviceSetCacheConfig(cudaFuncCachePreferL1));
  checkCudaErrors(cudaDeviceSetSharedMemConfig(cudaSharedMemBankSizeEightByte));
  checkCudaErrors(cudaGetDeviceProperties(&deviceProp, ctx->rank % 2));
  int maxThreads
    = deviceProp.multiProcessorCount * deviceProp.maxThreadsPerMultiProcessor;
  int nTasks = (ctx->end[X] - ctx->begin[X]) * (ctx->end[Y] - ctx->begin[Y]);
  int nTasksPerThread = nTasks / maxThreads
    + (nTasks % maxThreads != 0 ? 1 : 0);
  int nActiveThreads = nTasks / nTasksPerThread
    + (nTasks % nTasksPerThread != 0 ? 1 : 0);
  int nActiveThreadPerBlock
    = min(nActiveThreads, /*deviceProp.maxThreadsPerBlock*/512);
  ctx->blockDim = nActiveThreadPerBlock;
  int nActiveBlocks = nActiveThreads / nActiveThreadPerBlock
    + (nActiveThreads % nActiveThreadPerBlock != 0 ? 1 : 0);
  ctx->gridDim = nActiveBlocks;
  ctx->threadDim = nTasksPerThread;
  checkCudaErrors(cudaMalloc(&ctx->device_ctx, sizeof(*ctx->device_ctx)));
  checkCudaErrors(cudaMemcpy(ctx->device_ctx, ctx, sizeof(*ctx),
    cudaMemcpyHostToDevice));
  return;
}

__global__ void _cudaInitBorderXUp(Context* ctx) {
  INDEXTYPE idx = threadIdx.x + blockIdx.x * blockDim.x;
  if (idx < ctx->blockSize[Y])
    ctx->p[(0 + 1) * (ctx->blockSize[Y] + 2) + idx + 1]
        = fi(ctx->grid[X][0], ctx->grid[Y][idx]);
  return;
}

__global__ void _cudaInitBorderYLeft(Context* ctx) {
  INDEXTYPE idx = threadIdx.x + blockIdx.x * blockDim.x;
  if (idx < ctx->blockSize[X])
    ctx->p[(idx + 1) * (ctx->blockSize[Y] + 2) + 1]
        = fi(ctx->grid[X][idx], ctx->grid[Y][0]);
  return;
}

__global__ void _cudaInitBorderXDown(Context* ctx) {
  INDEXTYPE idx = threadIdx.x + blockIdx.x * blockDim.x;
  if (idx < ctx->blockSize[Y])
    ctx->p[ctx->blockSize[X] * (ctx->blockSize[Y] + 2) + idx + 1]
        = fi(ctx->grid[X][ctx->blockSize[X] - 1], ctx->grid[Y][idx]);
  return;
}

__global__ void _cudaInitBorderYRight(Context* ctx) {
  INDEXTYPE idx = threadIdx.x + blockIdx.x * blockDim.x;
  if (idx < ctx->blockSize[X])
    ctx->p[(idx + 1) * (ctx->blockSize[Y] + 2) + ctx->blockSize[Y]]
        = fi(ctx->grid[X][idx], ctx->grid[Y][ctx->blockSize[Y] - 1]);
  return;
}

extern "C"
void cudaStartup(Context* ctx) {
  INDEXTYPE gridDim;
  if (ctx->begin[X] != 0) {
    gridDim = ctx->blockSize[Y] / ctx->blockDim
      + (ctx->blockSize[Y] % ctx->blockDim != 0 ? 1 : 0);
    _cudaInitBorderXUp<<<gridDim, ctx->blockDim>>>(ctx->device_ctx);
  }
  if (ctx->begin[Y] != 0) {
    gridDim = ctx->blockSize[X] / ctx->blockDim
      + (ctx->blockSize[X] % ctx->blockDim != 0 ? 1 : 0);
    _cudaInitBorderYLeft<<<gridDim, ctx->blockDim>>>(ctx->device_ctx);
  }
  if (ctx->end[X] != ctx->blockSize[X]) {
    gridDim = ctx->blockSize[Y] / ctx->blockDim
      + (ctx->blockSize[Y] % ctx->blockDim != 0 ? 1 : 0);
    _cudaInitBorderXDown<<<gridDim, ctx->blockDim>>>(ctx->device_ctx);
  }
  if (ctx->end[Y] != ctx->blockSize[Y]) {
    gridDim = ctx->blockSize[X] / ctx->blockDim
      + (ctx->blockSize[X] % ctx->blockDim != 0 ? 1 : 0);
    _cudaInitBorderYRight<<<gridDim, ctx->blockDim>>>(ctx->device_ctx);
  }
  cudaStreamSynchronize(0);
  checkCudaErrors(cudaGetLastError());
  return;
}

__global__ void _cudaComputeR(Context* ctx) {
  INDEXTYPE i, j, k;
  for (k = threadIdx.x + blockIdx.x * blockDim.x;;
    k += gridDim.x * blockDim.x) {
    i = ctx->begin[X] + (k / (ctx->end[Y] - ctx->begin[Y]));
    j = ctx->begin[Y] + (k % (ctx->end[Y] - ctx->begin[Y]));
    if (i < ctx->end[X]) {
      ctx->r[(i + 1) * (ctx->blockSize[Y] + 2) + (j + 1)]
        = _cudaLaplOp(ctx, ctx->p, i, j) + ctx->f[i * ctx->blockSize[Y] + j];
    } else {
      break;
    }
  }
  return;
}

extern "C"
void cudaComputeR(Context* ctx) {
  _cudaComputeR<<<ctx->gridDim, ctx->blockDim>>>(ctx->device_ctx);
  cudaStreamSynchronize(0);
  checkCudaErrors(cudaGetLastError());
  return;
}

__global__ void copyVtoH(BASETYPE* dst, BASETYPE* src, INDEXTYPE pitch,
    INDEXTYPE size) {
  INDEXTYPE idx = threadIdx.x + blockIdx.x * blockDim.x;
  if (idx < size)
    dst[idx] = src[(idx + 1) * pitch];
  return;
}

__global__ void copyVtoD(BASETYPE* dst, BASETYPE* src, INDEXTYPE pitch,
    INDEXTYPE size) {
  INDEXTYPE idx = threadIdx.x + blockIdx.x * blockDim.x;
  if (idx < size)
    dst[(idx + 1) * pitch] = src[idx];
  return;
}

void cudaPrepareForUpdate(Context* ctx, BASETYPE* m) {
  INDEXTYPE gridDim = ctx->blockSize[X] / ctx->blockDim
    + (ctx->blockSize[X] % ctx->blockDim != 0 ? 1 : 0);
  // send last row to down
  if (ctx->coords[X] != ctx->dims[X] - 1) {
    cudaMemcpy(ctx->shadows[Y][1],
      &m[ctx->blockSize[X] * (ctx->blockSize[Y] + 2) + 1],
      ctx->blockSize[Y] * sizeof(*m), cudaMemcpyDeviceToDevice);
  }
  // send last column to right
  if (ctx->coords[Y] != ctx->dims[Y] - 1) {
    copyVtoH<<<gridDim, ctx->blockDim>>>(ctx->shadows[X][1],
      &m[(ctx->blockSize[Y] + 2) - 2], ctx->blockSize[Y] + 2,
      ctx->blockSize[X]);
  }
  // send first row to up
  if (ctx->coords[X] != 0) {
    cudaMemcpy(ctx->shadows[Y][2], &m[(ctx->blockSize[Y] + 2) + 1],
      ctx->blockSize[Y] * sizeof(*m), cudaMemcpyDeviceToDevice);
  }
  // send first column to left
  if (ctx->coords[Y] != 0) {
    copyVtoH<<<gridDim, ctx->blockDim>>>(ctx->shadows[X][2], &m[1],
      ctx->blockSize[Y] + 2, ctx->blockSize[X]);
  }
  return;
}

void cudaUpdateShadows(Context* ctx,
    MPI_Request (*requests)[NDIMS][NDIMS * NDIMS]) {
  // get shadow from up
  if (ctx->coords[X] != 0) {
    checkMPIErrors(MPI_Irecv(ctx->shadows[Y][0], ctx->blockSize[Y],
      MPI_BASETYPE, ctx->up, ctx->iteration, ctx->comm, &(*requests)[Y][0]));
  }
  // get shadow from left
  if (ctx->coords[Y] != 0) {
    checkMPIErrors(MPI_Irecv(ctx->shadows[X][0], ctx->blockSize[X],
      MPI_BASETYPE, ctx->left, ctx->iteration, ctx->comm, &(*requests)[X][0]));
  }
  // send last row to down
  if (ctx->coords[X] != ctx->dims[X] - 1) {
    checkMPIErrors(MPI_Isend(ctx->shadows[Y][1], ctx->blockSize[Y],
      MPI_BASETYPE, ctx->down, ctx->iteration, ctx->comm, &(*requests)[Y][1]));
  }
  // send last column to right
  if (ctx->coords[Y] != ctx->dims[Y] - 1) {
    checkMPIErrors(MPI_Isend(ctx->shadows[X][1], ctx->blockSize[X],
      MPI_BASETYPE, ctx->right, ctx->iteration, ctx->comm, &(*requests)[X][1]));
  }
  // send first row to up
  if (ctx->coords[X] != 0) {
    checkMPIErrors(MPI_Isend(ctx->shadows[Y][2], ctx->blockSize[Y],
      MPI_BASETYPE, ctx->up, ctx->iteration, ctx->comm, &(*requests)[Y][2]));
  }
  // send first column to left
  if (ctx->coords[Y] != 0) {
    checkMPIErrors(MPI_Isend(ctx->shadows[X][2], ctx->blockSize[X],
      MPI_BASETYPE, ctx->left, ctx->iteration, ctx->comm, &(*requests)[X][2]));
  }
  // get shadow from down
  if (ctx->coords[X] != ctx->dims[X] - 1) {
    checkMPIErrors(MPI_Irecv(ctx->shadows[Y][3], ctx->blockSize[Y],
      MPI_BASETYPE, ctx->down, ctx->iteration, ctx->comm, &(*requests)[Y][3]));
  }
  // get shadow from right
  if (ctx->coords[Y] != ctx->dims[Y] - 1) {
    checkMPIErrors(MPI_Irecv(ctx->shadows[X][3], ctx->blockSize[X],
      MPI_BASETYPE, ctx->right, ctx->iteration, ctx->comm, &(*requests)[X][3]));
  }
  return;
}

void cudaLoadShadows(Context* ctx, BASETYPE* m) {
  INDEXTYPE gridDim = ctx->blockSize[X] / ctx->blockDim
    + (ctx->blockSize[X] % ctx->blockDim != 0 ? 1 : 0);
  // get shadow from up
  if (ctx->coords[X] != 0) {
    cudaMemcpy(&m[1], ctx->shadows[Y][0],
      ctx->blockSize[Y] * sizeof(*ctx->shadows[Y][0]),
      cudaMemcpyDeviceToDevice);
  }
  // get shadow from left
  if (ctx->coords[Y] != 0) {
    copyVtoD<<<gridDim, ctx->blockDim>>>(m, ctx->shadows[X][0],
      ctx->blockSize[Y] + 2, ctx->blockSize[X]);
  }
  // get shadow from down
  if (ctx->coords[X] != ctx->dims[X] - 1) {
    cudaMemcpy(&m[(ctx->blockSize[X] + 1) * (ctx->blockSize[Y] + 2) + 1],
      ctx->shadows[Y][3], ctx->blockSize[Y] * sizeof(*ctx->shadows[Y][3]),
      cudaMemcpyDeviceToDevice);
  }
  // get shadow from right
  if (ctx->coords[Y] != ctx->dims[Y] - 1) {
    copyVtoD<<<gridDim, ctx->blockDim>>>(&m[(ctx->blockSize[Y] + 2) - 1],
      ctx->shadows[X][3], ctx->blockSize[Y] + 2, ctx->blockSize[X]);
  }
  return;
}

// main interconnection routine
extern "C"
void cudaActualizeShadows(Context* ctx, BASETYPE* m) {
  MPI_Request requests[NDIMS][NDIMS * NDIMS];
  INDEXTYPE i;
  for (i = 0; i < sizeof(requests) / sizeof(requests[0][0]); ++i) {
    ((MPI_Request*)requests)[i] = MPI_REQUEST_NULL;
  }
  cudaPrepareForUpdate(ctx, m);
  cudaStreamSynchronize(0);
  checkCudaErrors(cudaGetLastError());
  cudaUpdateShadows(ctx, &requests);
  checkMPIErrors(MPI_Waitall(sizeof(requests) / sizeof(requests[0][0]),
    (MPI_Request*)requests, MPI_STATUSES_IGNORE));
  cudaLoadShadows(ctx, m);
  cudaStreamSynchronize(0);
  checkCudaErrors(cudaGetLastError());
  return;
}

__global__ void _cudaComputeA(Context* ctx) {
  extern __shared__ BASETYPE s[];
  INDEXTYPE i, j, k;
  s[threadIdx.x] = 0;
  for (k = threadIdx.x + blockIdx.x * blockDim.x;;
    k += gridDim.x * blockDim.x) {
    i = ctx->begin[X] + (k / (ctx->end[Y] - ctx->begin[Y]));
    j = ctx->begin[Y] + (k % (ctx->end[Y] - ctx->begin[Y]));
    if (i < ctx->end[X]) {
      s[threadIdx.x]
        += ctx->prod_coeff[i * ctx->end[Y] + j] * _cudaLaplOp(ctx, ctx->r, i, j)
          * ctx->g[(i + 1) * (ctx->blockSize[Y] + 2) + (j + 1)];
    } else {
      break;
    }
  }
  
  __syncthreads();
  for (i = blockDim.x / 2; i > 0; i >>= 1) {
    if (threadIdx.x < i) {
      s[threadIdx.x] += s[threadIdx.x + i];
    }
    __syncthreads();
  }
  if (threadIdx.x == 0) {
    ctx->reduction[blockIdx.x] = s[0];
  }
  return;
}

__global__ void _cudaReduceA(Context* ctx) {
  INDEXTYPE i;
  for (i = 1; i < ctx->gridDim; ++i) {
    ctx->reduction[0] += ctx->reduction[i];
  }
  return;
}

__global__ void _cudaComputeG(Context* ctx, BASETYPE a) {
  INDEXTYPE i, j, k;
  for (k = threadIdx.x + blockIdx.x * blockDim.x;;
    k += gridDim.x * blockDim.x) {
    i = ctx->begin[X] + (k / (ctx->end[Y] - ctx->begin[Y]));
    j = ctx->begin[Y] + (k % (ctx->end[Y] - ctx->begin[Y]));
    if (i < ctx->end[X]) {
      ctx->g[(i + 1) * (ctx->blockSize[Y] + 2) + (j + 1)] *= a;
      ctx->g[(i + 1) * (ctx->blockSize[Y] + 2) + (j + 1)]
        += ctx->r[(i + 1) * (ctx->blockSize[Y] + 2) + (j + 1)];
    } else {
      break;
    }
  }
  return;
}

extern "C"
void cudaComputeG(Context* ctx, double dg_g) {
  _cudaComputeA
    <<<ctx->gridDim, ctx->blockDim, ctx->blockDim * sizeof(*ctx->reduction)>>>
    (ctx->device_ctx);
  _cudaReduceA<<<1, 1>>>(ctx->device_ctx);
  cudaStreamSynchronize(0);
  checkCudaErrors(cudaGetLastError());
  checkMPIErrors(MPI_Allreduce(&ctx->reduction[0], &ctx->reduction[1], 1,
    MPI_DOUBLE, MPI_SUM, ctx->comm));
  _cudaComputeG<<<ctx->gridDim, ctx->blockDim>>>(ctx->device_ctx,
    ctx->reduction[1] == 0 ? 0 : -ctx->reduction[1] / dg_g);
  cudaStreamSynchronize(0);
  checkCudaErrors(cudaGetLastError());
  return;
}

__global__ void _cudaComputeT(Context* ctx) {
  extern __shared__ BASETYPE s[];
  INDEXTYPE i, j, k;
  BASETYPE a;
  INDEXTYPE idx = 2 * threadIdx.x;
  s[idx] = 0;
  s[idx + 1] = 0;
  for (k = threadIdx.x + blockIdx.x * blockDim.x;;
    k += gridDim.x * blockDim.x) {
    i = ctx->begin[X] + (k / (ctx->end[Y] - ctx->begin[Y]));
    j = ctx->begin[Y] + (k % (ctx->end[Y] - ctx->begin[Y]));
    if (i < ctx->end[X]) {
      a = ctx->prod_coeff[i * ctx->end[Y] + j]
        * ctx->g[(i + 1) * (ctx->blockSize[Y] + 2) + (j + 1)];
      s[idx] += a * ctx->r[(i + 1) * (ctx->blockSize[Y] + 2) + (j + 1)];
      s[idx + 1] += a * _cudaLaplOp(ctx, ctx->g, i, j);
    } else {
      break;
    }
  }
  __syncthreads();
  for (i = blockDim.x / 2; i > 0; i >>= 1) {
    if (threadIdx.x < i) {
      s[idx] += s[idx + 2 * i];
      s[idx + 1] += s[idx + 2 * i + 1];
    }
    __syncthreads();
  }
  if (threadIdx.x == 0) {
    ctx->reduction[2 * blockIdx.x] = s[0];
    ctx->reduction[2 * blockIdx.x + 1] = s[1];
  }
  return;
}

__global__ void _cudaReduceT(Context* ctx) {
  INDEXTYPE i;
  for (i = 1; i < ctx->gridDim; ++i) {
    ctx->reduction[0] += ctx->reduction[2 * i];
    ctx->reduction[1] += ctx->reduction[2 * i + 1];
  }
  return;
}

__global__ void _cudaComputeP(Context* ctx, BASETYPE t) {
  extern __shared__ BASETYPE s[];
  INDEXTYPE i, j, k;
  s[threadIdx.x] = 0;
  for (k = threadIdx.x + blockIdx.x * blockDim.x;;
    k += gridDim.x * blockDim.x) {
    i = ctx->begin[X] + (k / (ctx->end[Y] - ctx->begin[Y]));
    j = ctx->begin[Y] + (k % (ctx->end[Y] - ctx->begin[Y]));
    if (i < ctx->end[X]) {
      ctx->p[(i + 1) * (ctx->blockSize[Y] + 2) + (j + 1)] += t * ctx->g[(i + 1)
        * (ctx->blockSize[Y] + 2) + (j + 1)];
      norm(ctx, &s[threadIdx.x], i, j,
        ctx->g[(i + 1) * (ctx->blockSize[Y] + 2) + (j + 1)]
          * ctx->g[(i + 1) * (ctx->blockSize[Y] + 2) + (j + 1)]);
    } else {
      break;
    }
  }
  __syncthreads();
  for (i = blockDim.x / 2; i > 0; i >>= 1) {
    if (threadIdx.x < i) {
      _norm(&s[threadIdx.x], s[threadIdx.x + i]);
    }
    __syncthreads();
  }
  if (threadIdx.x == 0) {
    ctx->reduction[blockIdx.x] = s[0];
  }
  return;
}

__global__ void _cudaReduceEps(Context* ctx) {
  INDEXTYPE i;
  for (i = 1; i < ctx->gridDim; ++i) {
    _norm(&ctx->reduction[0], ctx->reduction[i]);
  }
  return;
}

extern "C"
void cudaComputeP(Context* ctx, double* dg_g) {
  *dg_g = 0;
  double maxr = 0, t;
  _cudaComputeT
    <<<ctx->gridDim, ctx->blockDim,
      2 * ctx->blockDim * sizeof(*ctx->reduction)>>>
    (ctx->device_ctx);
  _cudaReduceT<<<1, 1>>>(ctx->device_ctx);
  cudaStreamSynchronize(0);
  checkCudaErrors(cudaGetLastError());
  checkMPIErrors(MPI_Allreduce(ctx->reduction, &ctx->reduction[2],
    2 * sizeof(*ctx->reduction), MPI_DOUBLE, MPI_SUM, ctx->comm));
  *dg_g = ctx->reduction[3];
  t = ctx->reduction[2] == 0 ? 0 : -ctx->reduction[2] / ctx->reduction[3];
  _cudaComputeP
    <<<ctx->gridDim, ctx->blockDim, ctx->blockDim * sizeof(*ctx->reduction)>>>
    (ctx->device_ctx, t);
  _cudaReduceEps<<<1, 1>>>(ctx->device_ctx);
  cudaStreamSynchronize(0);
  checkCudaErrors(cudaGetLastError());
  maxr = ctx->reduction[0];
  maxr = sqrt(maxr);
  maxr *= -t;
  checkMPIErrors(MPI_Allreduce(&maxr, &ctx->current_eps, 1, MPI_DOUBLE,
    MPI_MAX, ctx->comm));
  return;
}

extern "C"
void cudaFinalize(Context* ctx) {
  BASETYPE* p, * grid[NDIMS];
  checkMPIErrors(MPI_Alloc_mem(ctx->blockSize[X] * sizeof(*grid[X]),
    MPI_INFO_NULL, &grid[X]));
  checkMPIErrors(MPI_Alloc_mem(ctx->blockSize[Y] * sizeof(*grid[Y]),
    MPI_INFO_NULL, &grid[Y]));
  checkMPIErrors(MPI_Alloc_mem((ctx->blockSize[X] + 2) * (ctx->blockSize[Y] + 2)
    * sizeof(*p), MPI_INFO_NULL, &p));
  checkCudaErrors(cudaMemcpy(grid[X], ctx->grid[X],
    ctx->blockSize[X] * sizeof(*ctx->grid[X]), cudaMemcpyDeviceToHost));
  checkCudaErrors(cudaMemcpy(grid[Y], ctx->grid[Y],
    ctx->blockSize[Y] * sizeof(*ctx->grid[Y]), cudaMemcpyDeviceToHost));
  checkCudaErrors(cudaMemcpy(p, ctx->p,
    (ctx->blockSize[X] + 2) * (ctx->blockSize[Y] + 2) * sizeof(*ctx->p),
    cudaMemcpyDeviceToHost));
  checkCudaErrors(cudaFree(ctx->grid[X]));
  checkCudaErrors(cudaFree(ctx->grid[Y]));
  checkCudaErrors(cudaFree(ctx->p));
  ctx->grid[X] = grid[X];
  ctx->grid[Y] = grid[Y];
  ctx->p = p;
  return;
}

extern "C"
void cudaRelease(Context* ctx) {
  cudaDeviceSynchronize();
  cudaReleaseGrid(ctx);
  cudaReleaseComputationData(ctx);
  checkCudaErrors(cudaFree(ctx->device_ctx));
  cudaDeviceReset();
  return;
}
