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

#include <stdio.h>
#include <stdarg.h>


void __checkMPIErrors(int err, const char* file, const char* func,
    const int line) {
  if (MPI_SUCCESS != err) {
    char error[MPI_MAX_ERROR_STRING];
    int length = 0;
    MPI_Error_string(err, error, &length);
    error[length] = '\0';
    printf("%s error: %s (%s:%d)\n", func, error, file, line);
    MPI_Abort(MPI_COMM_WORLD, err);
  }
  return;
}

#define checkMPIErrors(err)  __checkMPIErrors(err, __FILE__, __func__, __LINE__)

void print(Context* ctx, const char* format, ...) {
  printf("%d(%d) @ (%d, %d): ", ctx->rank, ctx->size, ctx->coords[X],
    ctx->coords[Y]);
  va_list args;
  va_start(args, format);
  vprintf(format, args);
  va_end(args);
  return;
}

// requires N = 2 ^ n
// works only for NDIMS = 2
void partition(Context* ctx) {
  INDEXTYPE n1 = 0, n2 = 0, n = 0;
  int N = ctx->size;
  while (N != 1) {
    N >>= 1;
    ++n;
  }
  INDEXTYPE _N1 = ctx->N1;
  INDEXTYPE _N2 = ctx->N2;
  while (n1 + n2 != n) {
    if (_N1 >= _N2) {
      ++n1;
      _N1 >>= 1;
    } else {
      ++n2;
      _N2 >>= 1;
    }
  }
  ctx->dims[X] = 1 << n1;
  ctx->dims[Y] = 1 << n2;
  return;
}

void initEnvironment(Context* ctx) {
  checkMPIErrors(MPI_Comm_size(MPI_COMM_WORLD, &ctx->size));
  
  partition(ctx);
  
  int periods[NDIMS];
  INDEXTYPE i, j;
  for (i = 0; i < NDIMS; ++i) {
    periods[i] = 0;
  }
  
  checkMPIErrors(MPI_Cart_create(MPI_COMM_WORLD, NDIMS, ctx->dims, periods,
    /*reorder=*/1, &ctx->comm));
  checkMPIErrors(MPI_Comm_size(ctx->comm, &ctx->size));
  checkMPIErrors(MPI_Comm_rank(ctx->comm, &ctx->rank));
  checkMPIErrors(MPI_Cart_coords(ctx->comm, ctx->rank, NDIMS, ctx->coords));
  checkMPIErrors(MPI_Cart_shift(ctx->comm, X, 1, &ctx->up, &ctx->down));
  checkMPIErrors(MPI_Cart_shift(ctx->comm, Y, 1, &ctx->left, &ctx->right));
  return;
}

void initBlockParams(Context* ctx) {
  INDEXTYPE N1 = ctx->N1 + 1;
  INDEXTYPE N2 = ctx->N2 + 1;
  ctx->blockSize[X] = (0 <= ctx->coords[X]
    && ctx->coords[X] < N1 % ctx->dims[X])
    ? N1 / ctx->dims[X] + 1 : N1 / ctx->dims[X];
  ctx->blockSize[Y] = (0 <= ctx->coords[Y]
    && ctx->coords[Y] < N2 % ctx->dims[Y])
    ? N2 / ctx->dims[Y] + 1 : N2 / ctx->dims[Y];
  ctx->offsets[X] = 0;
  ctx->offsets[Y] = 0;
  INDEXTYPE i;
  for (i = 0; i < ctx->coords[X]; ++i) {
    ctx->offsets[X] += (0 <= i && i < N1 % ctx->dims[X])
      ? N1 / ctx->dims[X] + 1 : N1 / ctx->dims[X];
  }
  for (i = 0; i < ctx->coords[Y]; ++i) {
    ctx->offsets[Y] += (0 <= i && i < N2 % ctx->dims[Y])
      ? N2 / ctx->dims[Y] + 1 : N2 / ctx->dims[Y];
  }
  ctx->begin[X] = ctx->offsets[X] == 0 ? 1 : 0;
  ctx->begin[Y] = ctx->offsets[Y] == 0 ? 1 : 0;
  ctx->end[X] = ctx->offsets[X] + ctx->blockSize[X] - 1 == ctx->N1
      ? ctx->blockSize[X] - 1 : ctx->blockSize[X];
  ctx->end[Y] = ctx->offsets[Y] + ctx->blockSize[Y] - 1 == ctx->N2
      ? ctx->blockSize[Y] - 1 : ctx->blockSize[Y];
  return;
}

void initGrid(Context* ctx) {
  INDEXTYPE hSize[NDIMS] = {
    ((ctx->coords[X] == ctx->dims[X] - 1
    ? ctx->blockSize[X] - 1 : ctx->blockSize[X]) + 1),
    ((ctx->coords[Y] == ctx->dims[Y] - 1
    ? ctx->blockSize[Y] - 1 : ctx->blockSize[Y]) + 1)};
  checkMPIErrors(MPI_Alloc_mem(hSize[X] * sizeof(*ctx->h[X]),
    MPI_INFO_NULL, &ctx->h[X]));
  checkMPIErrors(MPI_Alloc_mem(hSize[Y] * sizeof(*ctx->h[Y]),
    MPI_INFO_NULL, &ctx->h[Y]));
  checkMPIErrors(MPI_Alloc_mem((hSize[X] - 1) * sizeof(*ctx->mh[X]),
    MPI_INFO_NULL, &ctx->mh[X]));
  checkMPIErrors(MPI_Alloc_mem((hSize[Y] - 1) * sizeof(*ctx->mh[Y]),
    MPI_INFO_NULL, &ctx->mh[Y]));
  checkMPIErrors(MPI_Alloc_mem((hSize[X] - 1) * (hSize[Y] - 1)
    * sizeof(*ctx->prod_coeff), MPI_INFO_NULL, &ctx->prod_coeff));
  checkMPIErrors(MPI_Alloc_mem(ctx->blockSize[X] * sizeof(*ctx->grid[X]),
    MPI_INFO_NULL, &ctx->grid[X]));
  checkMPIErrors(MPI_Alloc_mem(ctx->blockSize[Y] * sizeof(*ctx->grid[Y]),
    MPI_INFO_NULL, &ctx->grid[Y]));
  checkMPIErrors(MPI_Alloc_mem(ctx->blockSize[X] * ctx->blockSize[Y]
    * sizeof(*ctx->f), MPI_INFO_NULL, &ctx->f));
  INDEXTYPE i, j;
#pragma omp parallel for private(i)
  for (i = 0; i < ctx->blockSize[X]; ++i) {
    ctx->grid[X][i] = (ctx->A2 - ctx->A1)
      * f(((BASETYPE)i + ctx->offsets[X]) / ctx->N1, ctx->q) + ctx->A1;
  }
#pragma omp parallel for private(j)
  for (j = 0; j < ctx->blockSize[Y]; ++j) {
    ctx->grid[Y][j] = (ctx->B2 - ctx->B1)
      * f(((BASETYPE)j + ctx->offsets[Y]) / ctx->N2, ctx->q) + ctx->B1;
  }
#pragma omp parallel for private(i, j) COLLAPSE(2)
  for (i = 0; i < ctx->blockSize[X]; ++i) {
    for (j = 0; j < ctx->blockSize[Y]; ++j) {
      ctx->f[i * ctx->blockSize[Y] + j] = -F(ctx->grid[X][i], ctx->grid[Y][j]);
    }
  }
  ctx->h[X][0] = ctx->grid[X][0] - (ctx->A2 - ctx->A1)
    * f(((BASETYPE)ctx->offsets[X] - 1) / ctx->N1, ctx->q) + ctx->A1;
#pragma omp parallel for private(i)
  for (i = 1; i < hSize[X] - 1; ++i) {
    ctx->h[X][i] = ctx->grid[X][i] - ctx->grid[X][i - 1];
  }
  ctx->h[X][hSize[X] - 1] = (ctx->A2 - ctx->A1)
    * f(((BASETYPE)hSize[X] - 1 + ctx->offsets[X]) / ctx->N1, ctx->q) + ctx->A1
    - ctx->grid[X][hSize[X] - 2];
  
  ctx->h[Y][0] = ctx->grid[Y][0] - (ctx->B2 - ctx->B1)
    * f(((BASETYPE)ctx->offsets[Y] - 1) / ctx->N2, ctx->q) + ctx->B1;
#pragma omp parallel for private(j)
  for (j = 1; j < hSize[Y] - 1; ++j) {
    ctx->h[Y][j] = ctx->grid[Y][j] - ctx->grid[Y][j - 1];
  }
  ctx->h[Y][hSize[Y] - 1] = (ctx->B2 - ctx->B1)
    * f(((BASETYPE)hSize[Y] - 1 + ctx->offsets[Y]) / ctx->N2, ctx->q) + ctx->B1
    - ctx->grid[Y][hSize[Y] - 2];
#pragma omp parallel for private(i)
  for (i = 0; i < hSize[X] - 1; ++i) {
    ctx->mh[X][i] = 0.5 * (ctx->h[X][i + 1] + ctx->h[X][i]);
  }
#pragma omp parallel for private(j)
  for (j = 0; j < hSize[Y] - 1; ++j) {
    ctx->mh[Y][j] = 0.5 * (ctx->h[Y][j + 1] + ctx->h[Y][j]);
  }
#pragma omp parallel for private(i, j) COLLAPSE(2)
  for (i = 0; i < hSize[X] - 1; ++i) {
    for (j = 0; j < hSize[Y] - 1; ++j) {
      ctx->prod_coeff[i * (hSize[Y] - 1) + j] = ctx->mh[X][i] * ctx->mh[Y][j];
    }
  }
#pragma omp parallel for private(i)
  for (i = 0; i < hSize[X] - 1; ++i) {
    ctx->mh[X][i] = 1.0 / ctx->mh[X][i];
  }
#pragma omp parallel for private(j)
  for (j = 0; j < hSize[Y] - 1; ++j) {
    ctx->mh[Y][j] = 1.0 / ctx->mh[Y][j];
  }
#pragma omp parallel for private(i)
  for (i = 0; i < hSize[X]; ++i) {
    ctx->h[X][i] = 1.0 / ctx->h[X][i];
  }
#pragma omp parallel for private(j)
  for (j = 0; j < hSize[Y]; ++j) {
    ctx->h[Y][j] = 1.0 / ctx->h[Y][j];
  }
  return;
}

void initComputationData(Context* ctx) {
  INDEXTYPE i;
  checkMPIErrors(MPI_Alloc_mem(ctx->blockSize[X] * sizeof(*ctx->shadow[X]),
    MPI_INFO_NULL, &ctx->shadow[X]));
  checkMPIErrors(MPI_Alloc_mem(ctx->blockSize[X] * sizeof(*ctx->shadow[Y]),
    MPI_INFO_NULL, &ctx->shadow[Y]));
  checkMPIErrors(MPI_Alloc_mem(ctx->blockSize[X] * sizeof(*ctx->shadow_copy[X]),
    MPI_INFO_NULL, &ctx->shadow_copy[X]));
  checkMPIErrors(MPI_Alloc_mem(ctx->blockSize[X] * sizeof(*ctx->shadow_copy[Y]),
    MPI_INFO_NULL, &ctx->shadow_copy[Y]));
  checkMPIErrors(MPI_Alloc_mem((ctx->blockSize[X] + 2) * (ctx->blockSize[Y] + 2)
    * sizeof(*ctx->p), MPI_INFO_NULL, &ctx->p));
  checkMPIErrors(MPI_Alloc_mem((ctx->blockSize[X] + 2) * (ctx->blockSize[Y] + 2)
    * sizeof(*ctx->r), MPI_INFO_NULL, &ctx->r));
  checkMPIErrors(MPI_Alloc_mem((ctx->blockSize[X] + 2) * (ctx->blockSize[Y] + 2)
    * sizeof(*ctx->g), MPI_INFO_NULL, &ctx->g));
#pragma omp parallel for private(i)
  for (i = 0; i < ctx->blockSize[X]; ++i) {
    ctx->shadow[X][i] = 0;
    ctx->shadow[Y][i] = 0;
    ctx->shadow_copy[X][i] = 0;
    ctx->shadow_copy[Y][i] = 0;
  }
#pragma omp parallel for private(i)
  for (i = 0; i < (ctx->blockSize[X] + 2) * (ctx->blockSize[Y] + 2); ++i) {
    ctx->p[i] = 0;
  }
#pragma omp parallel for private(i)
  for (i = 0; i < (ctx->blockSize[X] + 2) * (ctx->blockSize[Y] + 2); ++i) {
    ctx->r[i] = 0;
  }
#pragma omp parallel for private(i)
  for (i = 0; i < (ctx->blockSize[X] + 2) * (ctx->blockSize[Y] + 2); ++i) {
    ctx->g[i] = 0;
  }
  return;
}

void releaseMemory(Context* ctx) {
  checkMPIErrors(MPI_Free_mem(ctx->h[X]));
  checkMPIErrors(MPI_Free_mem(ctx->h[Y]));
  checkMPIErrors(MPI_Free_mem(ctx->mh[X]));
  checkMPIErrors(MPI_Free_mem(ctx->mh[Y]));
  checkMPIErrors(MPI_Free_mem(ctx->grid[X]));
  checkMPIErrors(MPI_Free_mem(ctx->grid[Y]));
  checkMPIErrors(MPI_Free_mem(ctx->shadow[X]));
  checkMPIErrors(MPI_Free_mem(ctx->shadow[Y]));
  checkMPIErrors(MPI_Free_mem(ctx->shadow_copy[X]));
  checkMPIErrors(MPI_Free_mem(ctx->shadow_copy[Y]));
  checkMPIErrors(MPI_Free_mem(ctx->prod_coeff));
  checkMPIErrors(MPI_Free_mem(ctx->f));
  checkMPIErrors(MPI_Free_mem(ctx->p));
  checkMPIErrors(MPI_Free_mem(ctx->r));
  checkMPIErrors(MPI_Free_mem(ctx->g));
  return;
}

// five-point stencil
// requires m of size (ctx->blockSize[X] + 2) * (ctx->blockSize[Y] + 2)
// (fictive borders for "general" computation scheme (without explicit ifs))
inline BASETYPE LaplOp(Context* ctx, BASETYPE* m, INDEXTYPE i, INDEXTYPE j) {
  return ((m[(i + 1) * (ctx->blockSize[Y] + 2) + (j + 1)]
      - m[i * (ctx->blockSize[Y] + 2) + (j + 1)]) * ctx->h[X][i]
      - (m[(i + 2) * (ctx->blockSize[Y] + 2) + (j + 1)]
      - m[(i + 1) * (ctx->blockSize[Y] + 2) + (j + 1)]) * ctx->h[X][i + 1])
     * ctx->mh[X][i]
    + ((m[(i + 1) * (ctx->blockSize[Y] + 2) + (j + 1)]
        - m[(i + 1) * (ctx->blockSize[Y] + 2) + j]) * ctx->h[Y][j]
        - (m[(i + 1) * (ctx->blockSize[Y] + 2) + (j + 2)]
        - m[(i + 1) * (ctx->blockSize[Y] + 2) + (j + 1)]) * ctx->h[Y][j + 1])
     * ctx->mh[Y][j];
}

// only for columns, rows can be sent without extra storage
void prepareForUpdate(Context* ctx, BASETYPE* m) {
  INDEXTYPE i;
  // send last column to right
  if (ctx->coords[Y] != ctx->dims[Y] - 1) {
#pragma omp parallel for private(i)
    for (i = 0; i < ctx->blockSize[X]; ++i) {
      ctx->shadow_copy[Y][i] = m[(i + 2) * (ctx->blockSize[Y] + 2) - 2];
    }
  }
  // send first column to left
  if (ctx->coords[Y] != 0) {
#pragma omp parallel for private(i)
    for (i = 0; i < ctx->blockSize[X]; ++i) {
      ctx->shadow_copy[X][i] = m[(i + 1) * (ctx->blockSize[Y] + 2) + 1];
    }
  }
  return;
}

void updateShadows(Context* ctx, BASETYPE* m,
    MPI_Request (*requests)[NDIMS][NDIMS * NDIMS]) {
  // get shadow from up
  if (ctx->coords[X] != 0) {
    checkMPIErrors(MPI_Irecv(&m[1], ctx->blockSize[Y], MPI_BASETYPE,
      ctx->up, ctx->iteration, ctx->comm, &(*requests)[Y][0]));
  }
  // get shadow from left
  if (ctx->coords[Y] != 0) {
    checkMPIErrors(MPI_Irecv(ctx->shadow[X], ctx->blockSize[X], MPI_BASETYPE,
      ctx->left, ctx->iteration, ctx->comm, &(*requests)[X][0]));
  }
  // send last row to down
  if (ctx->coords[X] != ctx->dims[X] - 1) {
    checkMPIErrors(MPI_Isend(&m[ctx->blockSize[X]
      * (ctx->blockSize[Y] + 2) + 1], ctx->blockSize[Y], MPI_BASETYPE,
      ctx->down, ctx->iteration, ctx->comm, &(*requests)[Y][1]));
  }
  // send last column to right
  if (ctx->coords[Y] != ctx->dims[Y] - 1) {
    checkMPIErrors(MPI_Isend(ctx->shadow_copy[Y], ctx->blockSize[X],
      MPI_BASETYPE, ctx->right, ctx->iteration, ctx->comm, &(*requests)[X][1]));
  }
  // send first row to up
  if (ctx->coords[X] != 0) {
    checkMPIErrors(MPI_Isend(&m[(ctx->blockSize[Y] + 2) + 1], ctx->blockSize[Y],
      MPI_BASETYPE, ctx->up, ctx->iteration, ctx->comm, &(*requests)[Y][2]));
  }
  // send first column to left
  if (ctx->coords[Y] != 0) {
    checkMPIErrors(MPI_Isend(ctx->shadow_copy[X], ctx->blockSize[X],
      MPI_BASETYPE, ctx->left, ctx->iteration, ctx->comm, &(*requests)[X][2]));
  }
  // get shadow from down
  if (ctx->coords[X] != ctx->dims[X] - 1) {
    checkMPIErrors(MPI_Irecv(&m[(ctx->blockSize[X] + 1)
      * (ctx->blockSize[Y] + 2) + 1], ctx->blockSize[Y], MPI_BASETYPE,
        ctx->down, ctx->iteration, ctx->comm, &(*requests)[Y][3]));
  }
  // get shadow from right
  if (ctx->coords[Y] != ctx->dims[Y] - 1) {
    checkMPIErrors(MPI_Irecv(ctx->shadow[Y], ctx->blockSize[X], MPI_BASETYPE,
      ctx->right, ctx->iteration, ctx->comm, &(*requests)[X][3]));
  }
  return;
}

// only for columns, rows can be received without extra storage
void loadShadows(Context* ctx, BASETYPE* m) {
  INDEXTYPE i;
  // get shadow from left
  if (ctx->coords[Y] != 0) {
#pragma omp parallel for private(i)
    for (i = 0; i < ctx->blockSize[X]; ++i) {
      m[(i + 1) * (ctx->blockSize[Y] + 2)] = ctx->shadow[X][i];
    }
  }
  // get shadow from right
  if (ctx->coords[Y] != ctx->dims[Y] - 1) {
#pragma omp parallel for private(i)
    for (i = 0; i < ctx->blockSize[X]; ++i) {
      m[(i + 2) * (ctx->blockSize[Y] + 2) - 1] = ctx->shadow[Y][i];
    }
  }
  return;
}

// main interconnection routine
void actualizeShadows(Context* ctx, BASETYPE* m) {
  MPI_Request requests[NDIMS][NDIMS * NDIMS];
  INDEXTYPE i;
  for (i = 0; i < sizeof(requests) / sizeof(requests[0][0]); ++i) {
    ((MPI_Request*)requests)[i] = MPI_REQUEST_NULL;
  }
  prepareForUpdate(ctx, m);
  updateShadows(ctx, m, &requests);
  checkMPIErrors(MPI_Waitall(sizeof(requests) / sizeof(requests[0][0]),
    (MPI_Request*)requests, MPI_STATUSES_IGNORE));
  loadShadows(ctx, m);
  return;
}

void check(Context* ctx) {
  INDEXTYPE i, j;
  double r = 0, gr;
  for (i = ctx->begin[X]; i < ctx->end[X]; ++i) {
    for (j = ctx->begin[Y]; j < ctx->end[Y]; ++j) {
      r +=
        (ctx->p[(i + 1) * (ctx->blockSize[Y] + 2) + (j + 1)]
          - p(ctx->grid[X][i], ctx->grid[Y][j]))
          * (ctx->p[(i + 1) * (ctx->blockSize[Y] + 2) + (j + 1)]
            - p(ctx->grid[X][i], ctx->grid[Y][j]));
    }
  }
  checkMPIErrors(MPI_Reduce(&r, &gr, 1, MPI_DOUBLE, MPI_SUM, 0, ctx->comm));
  gr = sqrt(gr / ((ctx->N1 - 1) * (ctx->N2 - 1)));
  if (ctx->rank == 0) {
    print(ctx, "absolute bias %f\n", gr);
  }
  return;
}

/*
 dumps data "as is", preserves proper order whatever topology is
 format (binary):
 (BASETYPE == double ? 1 : 0)(1b)
 N1(4b)
 N2(4b)
 grid[X]((N1 + 1) * sizeof(BASETYPE)b)
 grid[Y]((N2 + 1) * sizeof(BASETYPE)b)
 p((N1 + 1) * (N2 + 1) * sizeof(BASETYPE)b)
*/
void dumpData(Context* ctx, char filename[]) {
  INDEXTYPE i, j, k;
  MPI_File file;
  checkMPIErrors(MPI_File_open(ctx->comm, filename,
    MPI_MODE_CREATE | MPI_MODE_WRONLY | MPI_MODE_DELETE_ON_CLOSE, MPI_INFO_NULL,
    &file));
  checkMPIErrors(MPI_File_close(&file));
  checkMPIErrors(MPI_File_open(ctx->comm, filename,
    MPI_MODE_CREATE | MPI_MODE_WRONLY, MPI_INFO_NULL, &file));
  if (ctx->rank == 0) {
    char c = sizeof(BASETYPE) == sizeof(double) ? 1 : 0;
    checkMPIErrors(MPI_File_write_shared(file, &c, 1, MPI_CHAR,
      MPI_STATUS_IGNORE));
    checkMPIErrors(MPI_File_write_shared(file, &ctx->N1, 1, MPI_INT,
      MPI_STATUS_IGNORE));
    checkMPIErrors(MPI_File_write_shared(file, &ctx->N2, 1, MPI_INT,
      MPI_STATUS_IGNORE));
  }
  checkMPIErrors(MPI_Barrier(ctx->comm));
  // grid
  for (i = 0; i < ctx->dims[X]; ++i) {
    if (ctx->coords[X] == i && ctx->coords[Y] == 0) {
      checkMPIErrors(MPI_File_write_shared(file, ctx->grid[X],
        ctx->blockSize[X], MPI_BASETYPE, MPI_STATUS_IGNORE));
    }
    checkMPIErrors(MPI_Barrier(ctx->comm));
  }
  checkMPIErrors(MPI_Barrier(ctx->comm));
  for (i = 0; i < ctx->dims[Y]; ++i) {
    if (ctx->coords[X] == 0 && ctx->coords[Y] == i) {
      checkMPIErrors(MPI_File_write_shared(file, ctx->grid[Y],
        ctx->blockSize[Y], MPI_BASETYPE, MPI_STATUS_IGNORE));
    }
    checkMPIErrors(MPI_Barrier(ctx->comm));
  }
  checkMPIErrors(MPI_Barrier(ctx->comm));
  // computed approximation
  int dims[] = {0, 1};
  MPI_Comm row_comm;
  checkMPIErrors(MPI_Cart_sub(ctx->comm, dims, &row_comm));
  for (i = 0; i < ctx->dims[X]; ++i) {
    if (ctx->coords[X] == i) {
      for (j = 0; j < ctx->blockSize[X]; ++j) {
        for (k = 0; k < ctx->dims[Y]; ++k) {
          if (ctx->coords[Y] == k) {
            checkMPIErrors(MPI_File_write_shared(file, &ctx->p[(j + 1)
              * (ctx->blockSize[Y] + 2) + 1], ctx->blockSize[Y],
              MPI_BASETYPE, MPI_STATUS_IGNORE));
          }
          checkMPIErrors(MPI_Barrier(row_comm));
        }
      }
    }
    checkMPIErrors(MPI_Barrier(ctx->comm));
  }
  checkMPIErrors(MPI_File_close(&file));
  return;
}

void startup(Context* ctx) {
  INDEXTYPE i, j;
  if (ctx->begin[X] != 0) {
#pragma omp parallel for private(j)
    for (j = 0; j < ctx->blockSize[Y]; ++j) {
      ctx->p[(0 + 1) * (ctx->blockSize[Y] + 2) + j + 1]
        = fi(ctx->grid[X][0], ctx->grid[Y][j]);
    }
  }
  if (ctx->begin[Y] != 0) {
#pragma omp parallel for private(i)
    for (i = 0; i < ctx->blockSize[X]; ++i) {
      ctx->p[(i + 1) * (ctx->blockSize[Y] + 2) + 1]
        = fi(ctx->grid[X][i], ctx->grid[Y][0]);
    }
  }
  if (ctx->end[X] != ctx->blockSize[X]) {
#pragma omp parallel for private(j)
    for (j = 0; j < ctx->blockSize[Y]; ++j) {
      ctx->p[ctx->blockSize[X] * (ctx->blockSize[Y] + 2) + j + 1]
        = fi(ctx->grid[X][ctx->blockSize[X] - 1], ctx->grid[Y][j]);
    }
  }
  if (ctx->end[Y] != ctx->blockSize[Y]) {
#pragma omp parallel for private(i)
    for (i = 0; i < ctx->blockSize[X]; ++i) {
      ctx->p[(i + 1) * (ctx->blockSize[Y] + 2) + ctx->blockSize[Y]]
        = fi(ctx->grid[X][i], ctx->grid[Y][ctx->blockSize[Y] - 1]);
    }
  }
  return;
}

void computeR(Context* ctx) {
  INDEXTYPE i, j;
#pragma omp parallel for private(i, j) COLLAPSE(2)
  for (i = ctx->begin[X]; i < ctx->end[X]; ++i) {
    for (j = ctx->begin[Y]; j < ctx->end[Y]; ++j) {
      ctx->r[(i + 1) * (ctx->blockSize[Y] + 2) + (j + 1)]
        = LaplOp(ctx, ctx->p, i, j) + ctx->f[i * ctx->blockSize[Y] + j];
    }
  }
  return;
}

void computeG(Context* ctx, BASETYPE dg_g) {
  INDEXTYPE i, j;
  double dr_g = 0, dd, a;
#pragma omp parallel for private(i, j) reduction(+: dr_g) COLLAPSE(2)
  for (i = ctx->begin[X]; i < ctx->end[X]; ++i) {
    for (j = ctx->begin[Y]; j < ctx->end[Y]; ++j) {
      dr_g += ctx->prod_coeff[i * ctx->end[Y] + j] * LaplOp(ctx, ctx->r, i, j)
        * ctx->g[(i + 1) * (ctx->blockSize[Y] + 2) + (j + 1)];
    }
  }
  checkMPIErrors(MPI_Allreduce(&dr_g, &dd, 1, MPI_DOUBLE, MPI_SUM,
    ctx->comm));
  a = dd == 0 ? 0 : -dd / dg_g;
#pragma omp parallel for private(i, j) COLLAPSE(2)
  for (i = ctx->begin[X]; i < ctx->end[X]; ++i) {
    for (j = ctx->begin[Y]; j < ctx->end[Y]; ++j) {
      ctx->g[(i + 1) * (ctx->blockSize[Y] + 2) + (j + 1)] *= a;
      ctx->g[(i + 1) * (ctx->blockSize[Y] + 2) + (j + 1)]
        += ctx->r[(i + 1) * (ctx->blockSize[Y] + 2) + (j + 1)];
    }
  }
  return;
}

void computeP(Context* ctx, double* dg_g) {
  INDEXTYPE i, j;
  double r_g = 0, t, maxr = 0;
  *dg_g = 0;
#pragma omp parallel for private(i, j, t) reduction(+: r_g, maxr) COLLAPSE(2)
  for (i = ctx->begin[X]; i < ctx->end[X]; ++i) {
    for (j = ctx->begin[Y]; j < ctx->end[Y]; ++j) {
      t = ctx->prod_coeff[i * ctx->end[Y] + j]
        * ctx->g[(i + 1) * (ctx->blockSize[Y] + 2) + (j + 1)];
      r_g += t * ctx->r[(i + 1) * (ctx->blockSize[Y] + 2) + (j + 1)];
      maxr += t * LaplOp(ctx, ctx->g, i, j);
    }
  }
  double ld[2] = {r_g, maxr}, gd[2];
  checkMPIErrors(MPI_Allreduce(ld, gd, sizeof(ld) / sizeof(ld[0]),
    MPI_DOUBLE, MPI_SUM, ctx->comm));
  *dg_g = gd[1];
  t = gd[0] == 0 ? 0 : -gd[0] / gd[1];
  maxr = 0;
#pragma omp parallel
{
  double _maxr = 0;
#pragma omp for private(i, j) COLLAPSE(2) nowait
  for (i = ctx->begin[X]; i < ctx->end[X]; ++i) {
    for (j = ctx->begin[Y]; j < ctx->end[Y]; ++j) {
      ctx->p[(i + 1) * (ctx->blockSize[Y] + 2) + (j + 1)]
        += t * ctx->g[(i + 1) * (ctx->blockSize[Y] + 2) + (j + 1)];
      norm(ctx, &_maxr, i, j,
        ctx->g[(i + 1) * (ctx->blockSize[Y] + 2) + (j + 1)]
          * ctx->g[(i + 1) * (ctx->blockSize[Y] + 2) + (j + 1)]);
    }
  }
#pragma omp critical
{
  _norm(&maxr, _maxr);
}
}
  maxr = sqrt(maxr);
  maxr *= -t;
  checkMPIErrors(MPI_Allreduce(&maxr, &ctx->current_eps, 1, MPI_DOUBLE,
    MPI_MAX, ctx->comm));
  return;
}

int main(int argc, char* argv[]) {
  if (argc != 4) {
    printf("Usage: %s <N1> <N2> <output>\n", argv[0]);
    return 0;
  }
  
  int N1 = atoi(argv[1]);
  int N2 = atoi(argv[2]);
  if (N1 <= 0 || N2 <= 0) {
    printf("N1 and N2 must be >0\n");
    return 0;
  }
  
  int provided;
  checkMPIErrors(MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &provided));
  
  Context ctx;
  ctx.iteration = 0;
  
  double time, gtime, dg_g = 0;
  
  time = MPI_Wtime();
  
  initProblemParams(&ctx, N1, N2);
  
  initEnvironment(&ctx);

  initBlockParams(&ctx);
  
#if defined(CUDA)
  cudaConfigure(&ctx);
  cudaInitGrid(&ctx);
  cudaInitComputationData(&ctx);
  time = MPI_Wtime() - time;
  checkMPIErrors(MPI_Reduce(&time, &gtime, 1, MPI_DOUBLE, MPI_MAX, 0,
    ctx.comm));
  if (ctx.rank == 0) {
    print(&ctx, "init: %f seconds elapsed\n", gtime);
  }
  checkMPIErrors(MPI_Barrier(ctx.comm));
  time = MPI_Wtime();
  cudaStartup(&ctx);
  do {
    cudaComputeR(&ctx);
    cudaActualizeShadows(&ctx, ctx.r);
    cudaComputeG(&ctx, dg_g);
    cudaActualizeShadows(&ctx, ctx.g);
    cudaComputeP(&ctx, &dg_g);
    cudaActualizeShadows(&ctx, ctx.p);
#if defined(DEBUG)
    if (ctx.rank == 0)
      print(&ctx, "iteration %d: current_eps = %f (<%f required)\n",
        ctx.iteration, ctx.current_eps, ctx.eps);
#endif
    ++ctx.iteration;
  } while (ctx.current_eps >= ctx.eps);
#else
  initGrid(&ctx);
  initComputationData(&ctx);
  time = MPI_Wtime() - time;
  checkMPIErrors(MPI_Reduce(&time, &gtime, 1, MPI_DOUBLE, MPI_MAX, 0,
    ctx.comm));
  if (ctx.rank == 0) {
    print(&ctx, "init: %f seconds elapsed\n", gtime);
  }
  checkMPIErrors(MPI_Barrier(ctx.comm));
  time = MPI_Wtime();
  startup(&ctx);
  do {
    computeR(&ctx);
    actualizeShadows(&ctx, ctx.r);
    computeG(&ctx, dg_g);
    actualizeShadows(&ctx, ctx.g);
    computeP(&ctx, &dg_g);
    actualizeShadows(&ctx, ctx.p);
#if defined(DEBUG)
    if (ctx.rank == 0)
      print(&ctx, "iteration %d: current_eps = %f (<%f required)\n",
        ctx.iteration, ctx.current_eps, ctx.eps);
#endif
    ++ctx.iteration;
  } while (ctx.current_eps >= ctx.eps);
#endif
  
  time = MPI_Wtime() - time;
  checkMPIErrors(MPI_Reduce(&time, &gtime, 1, MPI_DOUBLE, MPI_MAX, 0,
    ctx.comm));
  if (ctx.rank == 0) {
    print(&ctx, "%d iterations, %f seconds elapsed\n", ctx.iteration, gtime);
  }
  
#if defined(CUDA)
  cudaFinalize(&ctx);
#endif
  
  dumpData(&ctx, argv[3]);
  
  check(&ctx);
  
#if defined(CUDA)
  cudaRelease(&ctx);
#else
  releaseMemory(&ctx);
#endif
  
  checkMPIErrors(MPI_Finalize());
  
  return 0;
}
