
extern "C" {
#include <cblas.h> 
}

template<class T>
struct CBLAS {
  typedef T value_type;
  static void trmm(const enum CBLAS_ORDER Order, const enum CBLAS_SIDE Side,
                   const enum CBLAS_UPLO Uplo, const enum CBLAS_TRANSPOSE TransA,
                   const enum CBLAS_DIAG Diag, const int M, const int N,
                   const value_type alpha, const value_type *A, const int lda,
                   value_type *B, const int ldb);
  static void trsm(const enum CBLAS_ORDER Order, const enum CBLAS_SIDE Side,
                   const enum CBLAS_UPLO Uplo, const enum CBLAS_TRANSPOSE TransA,
                   const enum CBLAS_DIAG Diag, const int M, const int N,
                   const value_type alpha, const value_type *A, const int lda,
                   value_type *B, const int ldb);
  static void gemm(const enum CBLAS_ORDER Order, const enum CBLAS_TRANSPOSE TransA,
                   const enum CBLAS_TRANSPOSE TransB, const int M, const int N,
                   const int K, const value_type alpha, const value_type *A,
                   const int lda, const value_type *B, const int ldb,
                   const value_type beta, value_type *C, const int ldc);
  static void syrk(const enum CBLAS_ORDER Order, const enum CBLAS_UPLO Uplo,
                   const enum CBLAS_TRANSPOSE Trans, const int N, const int K,
                   const value_type alpha, const value_type *A, const int lda,
                   const value_type beta, value_type *C, const int ldc);

  static void axpy(const int N, const value_type alpha, const value_type *X,
                   const int incX, value_type *Y, const int incY);

  static void scal(const int N, const value_type alpha, value_type *X, const int incX);

  static void ger(const enum CBLAS_ORDER order, const int M, const int N,
		  const value_type alpha, const value_type *X, const int incX,
		  const value_type *Y, const int incY, value_type *A, const int lda);

};

/* specialization for double */
template<>
struct CBLAS<double> {
  typedef double value_type;
  static void trmm(const enum CBLAS_ORDER Order, const enum CBLAS_SIDE Side,
                   const enum CBLAS_UPLO Uplo, const enum CBLAS_TRANSPOSE TransA,
                   const enum CBLAS_DIAG Diag, const int M, const int N,
                   const value_type alpha, const value_type *A, const int lda,
                   value_type *B, const int ldb)
  { cblas_dtrmm( Order, Side, Uplo, TransA, Diag, M, N, alpha, A, lda, B, ldb); }

  static void trsm(const enum CBLAS_ORDER Order, const enum CBLAS_SIDE Side,
                   const enum CBLAS_UPLO Uplo, const enum CBLAS_TRANSPOSE TransA,
                   const enum CBLAS_DIAG Diag, const int M, const int N,
                   const value_type alpha, const value_type *A, const int lda,
                   value_type *B, const int ldb)
  { cblas_dtrsm( Order, Side, Uplo, TransA, Diag, M, N, alpha, A, lda, B, ldb); }

  static void gemm(const enum CBLAS_ORDER Order, const enum CBLAS_TRANSPOSE TransA,
                   const enum CBLAS_TRANSPOSE TransB, const int M, const int N,
                   const int K, const value_type alpha, const value_type *A,
                   const int lda, const value_type *B, const int ldb,
                   const value_type beta, value_type *C, const int ldc)
  { cblas_dgemm( Order, TransA, TransB, M, N, K, alpha, A, lda, B, ldb, beta, C, ldc); }

  static void syrk(const enum CBLAS_ORDER Order, const enum CBLAS_UPLO Uplo,
                   const enum CBLAS_TRANSPOSE Trans, const int N, const int K,
                   const value_type alpha, const value_type *A, const int lda,
                   const value_type beta, value_type *C, const int ldc)
  { cblas_dsyrk( Order, Uplo, Trans, N, K, alpha, A, lda, beta, C, ldc); }

  static void axpy(const int N, const value_type alpha, const value_type *X,
                   const int incX, value_type *Y, const int incY)
  { cblas_daxpy(N, alpha, X, incX, Y, incY); }

  static void scal(const int N, const value_type alpha, value_type *X, const int incX)
  { cblas_dscal(N, alpha, X, incX); }

  static void ger(const enum CBLAS_ORDER order, const int M, const int N,
		  const value_type alpha, const value_type *X, const int incX,
		  const value_type *Y, const int incY, value_type *A, const int lda)
  { cblas_dger(order, M, N, alpha, X, incX, Y, incY, A, lda); }
};

/* specialization for float */
template<>
struct CBLAS<float> {
  typedef float value_type;
  static void trmm(const enum CBLAS_ORDER Order, const enum CBLAS_SIDE Side,
                   const enum CBLAS_UPLO Uplo, const enum CBLAS_TRANSPOSE TransA,
                   const enum CBLAS_DIAG Diag, const int M, const int N,
                   const value_type alpha, const value_type *A, const int lda,
                   value_type *B, const int ldb)
  { cblas_strmm( Order, Side, Uplo, TransA, Diag, M, N, alpha, A, lda, B, ldb); }

  static void trsm(const enum CBLAS_ORDER Order, const enum CBLAS_SIDE Side,
                   const enum CBLAS_UPLO Uplo, const enum CBLAS_TRANSPOSE TransA,
                   const enum CBLAS_DIAG Diag, const int M, const int N,
                   const value_type alpha, const value_type *A, const int lda,
                   value_type *B, const int ldb)
  { cblas_strsm( Order, Side, Uplo, TransA, Diag, M, N, alpha, A, lda, B, ldb); }

  static void gemm(const enum CBLAS_ORDER Order, const enum CBLAS_TRANSPOSE TransA,
                   const enum CBLAS_TRANSPOSE TransB, const int M, const int N,
                   const int K, const value_type alpha, const value_type *A,
                   const int lda, const value_type *B, const int ldb,
                   const value_type beta, value_type *C, const int ldc)
  { cblas_sgemm( Order, TransA, TransB, M, N, K, alpha, A, lda, B, ldb, beta, C, ldc); }

  static void syrk(const enum CBLAS_ORDER Order, const enum CBLAS_UPLO Uplo,
                   const enum CBLAS_TRANSPOSE Trans, const int N, const int K,
                   const value_type alpha, const value_type *A, const int lda,
                   const value_type beta, value_type *C, const int ldc)
  { cblas_ssyrk( Order, Uplo, Trans, N, K, alpha, A, lda, beta, C, ldc); }

  static void axpy(const int N, const value_type alpha, const value_type *X,
                   const int incX, value_type *Y, const int incY)
  { cblas_saxpy(N, alpha, X, incX, Y, incY); }

  static void scal(const int N, const value_type alpha, value_type *X, const int incX)
  { cblas_sscal(N, alpha, X, incX); }

  static void ger(const enum CBLAS_ORDER order, const int M, const int N,
		  const value_type alpha, const value_type *X, const int incX,
		  const value_type *Y, const int incY, value_type *A, const int lda)
  { cblas_sger(order, M, N, alpha, X, incX, Y, incY, A, lda); }
};

