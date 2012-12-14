/*
 ** xkaapi
 **
 ** Copyright 2009, 2010, 2011, 2012 INRIA.
 **
 ** Contributors :
 **
 ** thierry.gautier@inrialpes.fr
 ** Joao.Lima@imag.fr / joao.lima@inf.ufrgs.br
 **
 ** This software is a computer program whose purpose is to execute
 ** multithreaded computation with data flow synchronization between
 ** threads.
 **
 ** This software is governed by the CeCILL-C license under French law
 ** and abiding by the rules of distribution of free software.  You can
 ** use, modify and/ or redistribute the software under the terms of
 ** the CeCILL-C license as circulated by CEA, CNRS and INRIA at the
 ** following URL "http://www.cecill.info".
 **
 ** As a counterpart to the access to the source code and rights to
 ** copy, modify and redistribute granted by the license, users are
 ** provided only with a limited warranty and the software's author,
 ** the holder of the economic rights, and the successive licensors
 ** have only limited liability.
 **
 ** In this respect, the user's attention is drawn to the risks
 ** associated with loading, using, modifying and/or developing or
 ** reproducing the software by the user in light of its specific
 ** status of free software, that may mean that it is complicated to
 ** manipulate, and that also therefore means that it is reserved for
 ** developers and experienced professionals having in-depth computer
 ** knowledge. Users are therefore encouraged to load and test the
 ** software's suitability as regards their requirements in conditions
 ** enabling the security of their systems and/or data to be ensured
 ** and, more generally, to use and operate it in the same conditions
 ** as regards security.
 **
 ** The fact that you are presently reading this means that you have
 ** had knowledge of the CeCILL-C license and that you accept its
 ** terms.
 **
 */
 
#if defined(CONFIG_USE_CUBLAS)
/* from cublas.h */

#include <cuda_runtime_api.h>
#include "cublas_v2.h"

/* Helper functions */
static inline cublasOperation_t convertToOp( int trans ) 
{
    switch(trans) {
        case CblasNoTrans:
            return CUBLAS_OP_N;
        case CblasTrans:
            return CUBLAS_OP_T;
        case CblasConjTrans:
            return CUBLAS_OP_C;                        
        default:
            return CUBLAS_OP_N;
    }

}

static inline cublasFillMode_t convertToFillMode( int uplo ) 
{
    switch (uplo) {
        case CblasUpper:
            return CUBLAS_FILL_MODE_UPPER;
	case CblasLower:
        default:
         return CUBLAS_FILL_MODE_LOWER;
    }        
}

static inline cublasDiagType_t convertToDiagType( int diag ) 
{
    switch (diag) {
	case CblasUnit:
            return CUBLAS_DIAG_UNIT;
	case CblasNonUnit:
        default:
         return CUBLAS_DIAG_NON_UNIT;
    }        
}

static inline cublasSideMode_t convertToSideMode( int side ) 
{
    switch (side) {
	case CblasRight:
            return CUBLAS_SIDE_RIGHT;
	case CblasLeft:
        default:
         return CUBLAS_SIDE_LEFT;
    }        
}
/* for cublas v2 */
template<class T>
struct CUBLAS {
  typedef T value_type;
  static cublasStatus_t trsm(cublasHandle_t handle,
                     cublasSideMode_t side,
                     cublasFillMode_t uplo,
                     cublasOperation_t trans,
                     cublasDiagType_t diag,
                     int m,
                     int n,
                     const value_type *alpha, /* host or device pointer */
                     const value_type *A,
                     int lda,
                     value_type *B,
                     int ldb);

  static cublasStatus_t trsm(
                     cublasSideMode_t side,
                     cublasFillMode_t uplo,
                     cublasOperation_t trans,
                     cublasDiagType_t diag,
                     int m,
                     int n,
                     const value_type *alpha, /* host or device pointer */
                     const value_type *A,
                     int lda,
                     value_type *B,
                     int ldb);
                                         
  static cublasStatus_t gemm( cublasHandle_t handle,
                    cublasOperation_t transa,
                    cublasOperation_t transb,
                    int m,
                    int n,
                    int k,
                    const value_type *alpha, /* host or device pointer */
                    const value_type *A,
                    int lda,
                    const value_type *B,
                    int ldb,
                    const value_type *beta, /* host or device pointer */
                    value_type *C,
                    int ldc);

  static cublasStatus_t gemm( 
                    cublasOperation_t transa,
                    cublasOperation_t transb,
                    int m,
                    int n,
                    int k,
                    const value_type *alpha, /* host or device pointer */
                    const value_type *A,
                    int lda,
                    const value_type *B,
                    int ldb,
                    const value_type *beta, /* host or device pointer */
                    value_type *C,
                    int ldc);

  static cublasStatus_t syrk(cublasHandle_t handle,
                     cublasFillMode_t uplo,
                     cublasOperation_t trans,
                     int n,
                     int k,
                     const value_type *alpha,  /* host or device pointer */
                     const value_type *A,
                     int lda,
                     const value_type *beta,  /* host or device pointer */
                     value_type *C,
                     int ldc);

  static cublasStatus_t swap(cublasHandle_t handle,
	int n,
	value_type* x,
	int incx,
	value_type* y,
	int incy);

  static cublasStatus_t swap( int n,
      value_type* x, int incx, value_type* y, int incy);

};

template<>
struct CUBLAS<double> {
  typedef double value_type;
  static cublasStatus_t trsm(cublasHandle_t handle,
                   cublasSideMode_t side,
                   cublasFillMode_t uplo,
                   cublasOperation_t trans,
                   cublasDiagType_t diag,
                   int m,
                   int n,
                   const value_type *alpha, /* host or device pointer */
                   const value_type *A,
                   int lda,
                   value_type *B,
                   int ldb)
  { return cublasDtrsm_v2(handle, side, uplo, trans, diag, m, n, alpha, A, lda, B, ldb); }

  static cublasStatus_t trsm(
                   cublasSideMode_t side,
                   cublasFillMode_t uplo,
                   cublasOperation_t trans,
                   cublasDiagType_t diag,
                   int m,
                   int n,
                   const value_type *alpha, /* host or device pointer */
                   const value_type *A,
                   int lda,
                   value_type *B,
                   int ldb)
  { return cublasDtrsm_v2(kaapi_cuda_cublas_handle(), side, uplo, trans, diag, m, n, alpha, A, lda, B, ldb); }

  static cublasStatus_t gemm( cublasHandle_t handle,
                    cublasOperation_t transa,
                    cublasOperation_t transb,
                    int m,
                    int n,
                    int k,
                    const value_type *alpha, /* host or device pointer */
                    const value_type *A,
                    int lda,
                    const value_type *B,
                    int ldb,
                    const value_type *beta, /* host or device pointer */
                    value_type *C,
                    int ldc)
  { return cublasDgemm_v2(handle, transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc); }

  static cublasStatus_t gemm(
                    cublasOperation_t transa,
                    cublasOperation_t transb,
                    int m,
                    int n,
                    int k,
                    const value_type *alpha, /* host or device pointer */
                    const value_type *A,
                    int lda,
                    const value_type *B,
                    int ldb,
                    const value_type *beta, /* host or device pointer */
                    value_type *C,
                    int ldc)
  { return cublasDgemm_v2(kaapi_cuda_cublas_handle(), transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc); }

  static cublasStatus_t syrk(cublasHandle_t handle,
                     cublasFillMode_t uplo,
                     cublasOperation_t trans,
                     int n,
                     int k,
                     const value_type *alpha,  /* host or device pointer */
                     const value_type *A,
                     int lda,
                     const value_type *beta,  /* host or device pointer */
                     value_type *C,
                     int ldc)
  { return cublasDsyrk_v2(handle, uplo, trans, n, k, alpha, A, lda, beta, C, ldc); }

  static cublasStatus_t swap(cublasHandle_t handle,
	int n,
	value_type* x,
	int incx,
	value_type* y,
	int incy)
  { return cublasDswap_v2(handle, n, x, incx, y, incy); }

  static cublasStatus_t swap( int n,
	value_type* x, int incx,
	value_type* y, int incy)
  { return cublasDswap_v2(kaapi_cuda_cublas_handle(), n, x, incx, y, incy); }
};

template<>
struct CUBLAS<float> {
  typedef float value_type;
  static cublasStatus_t trsm(cublasHandle_t handle,
                   cublasSideMode_t side,
                   cublasFillMode_t uplo,
                   cublasOperation_t trans,
                   cublasDiagType_t diag,
                   int m,
                   int n,
                   const value_type *alpha, /* host or device pointer */
                   const value_type *A,
                   int lda,
                   value_type *B,
                   int ldb)
  { return cublasStrsm_v2(handle, side, uplo, trans, diag, m, n, alpha, A, lda, B, ldb); }

  static cublasStatus_t trsm(
                   cublasSideMode_t side,
                   cublasFillMode_t uplo,
                   cublasOperation_t trans,
                   cublasDiagType_t diag,
                   int m,
                   int n,
                   const value_type *alpha, /* host or device pointer */
                   const value_type *A,
                   int lda,
                   value_type *B,
                   int ldb)
  { return cublasStrsm_v2(kaapi_cuda_cublas_handle(), side, uplo, trans, diag, m, n, alpha, A, lda, B, ldb); }

  static cublasStatus_t gemm( cublasHandle_t handle,
                    cublasOperation_t transa,
                    cublasOperation_t transb,
                    int m,
                    int n,
                    int k,
                    const value_type *alpha, /* host or device pointer */
                    const value_type *A,
                    int lda,
                    const value_type *B,
                    int ldb,
                    const value_type *beta, /* host or device pointer */
                    value_type *C,
                    int ldc)
  { return cublasSgemm_v2(handle, transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc); }

  static cublasStatus_t gemm(
                    cublasOperation_t transa,
                    cublasOperation_t transb,
                    int m,
                    int n,
                    int k,
                    const value_type *alpha, /* host or device pointer */
                    const value_type *A,
                    int lda,
                    const value_type *B,
                    int ldb,
                    const value_type *beta, /* host or device pointer */
                    value_type *C,
                    int ldc)
  { return cublasSgemm_v2(kaapi_cuda_cublas_handle(), transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc); }

  static cublasStatus_t syrk(cublasHandle_t handle,
                     cublasFillMode_t uplo,
                     cublasOperation_t trans,
                     int n,
                     int k,
                     const value_type *alpha,  /* host or device pointer */
                     const value_type *A,
                     int lda,
                     const value_type *beta,  /* host or device pointer */
                     value_type *C,
                     int ldc)
  { return cublasSsyrk_v2(handle, uplo, trans, n, k, alpha, A, lda, beta, C, ldc); }

  static cublasStatus_t swap(cublasHandle_t handle,
	int n,
	value_type* x,
	int incx,
	value_type* y,
	int incy)
  { return cublasSswap_v2(handle, n, x, incx, y, incy); }

  static cublasStatus_t swap( int n,
	value_type* x, int incx,
	value_type* y, int incy)
  { return cublasSswap_v2(kaapi_cuda_cublas_handle(), n, x, incx, y, incy); }
};

#endif /* CONFIG_USE_CUBLAS */

