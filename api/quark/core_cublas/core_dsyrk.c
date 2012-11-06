/**
 *
 * @file cublas_core_dsyrk.c
 *
 *  PLASMA core_blas kernel for CUBLAS
 * (c) INRIA
 *
 * @version 2.4.6
 * @author Thierry Gautier
 *
 **/

#include "core_cublas.h"

void CORE_dsyrk_cublas(int uplo, int trans,
                int n, int k,
                double alpha, const double *A, int LDA,
                double beta, double *C, int LDC)
{
  cublasStatus_t status = cublasDsyrk_v2(
                                         kaapi_cuda_cublas_handle(),
                                         uplo,
                                         trans,
                                         n, k,
                                         &alpha, A, LDA,
                                         &beta, C, LDC);
  PLASMA_CUBLAS_ASSERT(status);
#if 0
  fprintf(stdout,"%s: a=%p c=%p n=%d k=%d\n", __FUNCTION__,
          A, C, n, k);
  fflush(stdout);
#endif
}


/***************************************************************************
 *
 **/
void CORE_dsyrk_quark_cublas(Quark *quark)
{
    PLASMA_enum uplo;
    PLASMA_enum trans;
    int n;
    int k;
    double alpha;
    double *A;
    int lda;
    double beta;
    double *C;
    int ldc;

    quark_unpack_args_10(quark, uplo, trans, n, k, alpha, A, lda, beta, C, ldc);
    CORE_dsyrk_cublas(
                     PLASMA_CUBLAS_convertToFillMode(uplo),
                     PLASMA_CUBLAS_convertToOp(trans),
                     n, k,
                     alpha, A, lda,
                     beta, C, ldc);
}
