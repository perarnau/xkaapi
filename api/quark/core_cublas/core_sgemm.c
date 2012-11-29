/**
 *
 * @file cublas_core_sgemm.c
 *
 *  PLASMA core_blas kernel for CUBLAS
 * (c) INRIA
 *
 * @author Joao Lima
 *
 **/

#include "core_cublas.h"

void CORE_sgemm_cublas(int transA, int transB,
                int M, int N, int K,
                float alpha, float *A, int LDA,
                float *B, int LDB,
                float beta, float *C, int LDC)
{
  cublasStatus_t status = cublasSgemm_v2(
                                         kaapi_cuda_cublas_handle(),
                                         PLASMA_CUBLAS_convertToOp(transA),
                                         PLASMA_CUBLAS_convertToOp(transB),
                                         M, N, K,
                                         &alpha, A, LDA,
                                         B, LDB,
                                         &beta, C, LDC);
  PLASMA_CUBLAS_ASSERT(status);
#if CONFIG_VERBOSE
  fprintf(stdout,"%s: a=%p b=%p c=%p m=%d n=%d k=%d\n", __FUNCTION__,
          A, B, C, M, N, K);
  fflush(stdout);
#endif
}

void CORE_sgemm_quark_cublas(Quark *quark)
{
  int transA;
  int transB;
  int m;
  int n;
  int k;
  float alpha;
  float *A;
  int lda;
  float *B;
  int ldb;
  float beta;
  float *C;
  int ldc;
  
  quark_unpack_args_13(quark, transA, transB, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
  CORE_sgemm_cublas(
              transA, transB,
              m, n, k,
              (alpha), A, lda,
              B, ldb,
              (beta), C, ldc);
}