
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
};

#endif /* CONFIG_USE_CUBLAS */

