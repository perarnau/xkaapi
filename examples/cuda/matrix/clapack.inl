
extern "C" {
#include <clapack.h>
}

template<class T>
struct CLAPACK {
  typedef T value_type;
  static int getrf(const enum CBLAS_ORDER Order, const int M, const int N,
                   value_type *A, const int lda, int *ipiv);
  static int potrf(const enum ATLAS_ORDER Order, const enum ATLAS_UPLO Uplo,
                   const int N, value_type *A, const int lda);
};

template<>
struct CLAPACK<double> {
  typedef double value_type;
  static int getrf(const enum CBLAS_ORDER Order, const int M, const int N,
                   value_type *A, const int lda, int *ipiv)
  { return clapack_dgetrf(Order, M, N, A, lda, ipiv); }
                   
  static int potrf(const enum ATLAS_ORDER Order, const enum ATLAS_UPLO Uplo,
                   const int N, value_type *A, const int lda)
  { return clapack_dpotrf(Order, Uplo, N, A, lda); }
};

template<>
struct CLAPACK<float> {
  typedef float value_type;
  static int getrf(const enum CBLAS_ORDER Order, const int M, const int N,
                   value_type *A, const int lda, int *ipiv)
  { return clapack_sgetrf(Order, M, N, A, lda, ipiv); }
                   
  static int potrf(const enum ATLAS_ORDER Order, const enum ATLAS_UPLO Uplo,
                   const int N, value_type *A, const int lda)
  { return clapack_spotrf(Order, Uplo, N, A, lda); }
};

