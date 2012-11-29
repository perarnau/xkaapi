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

