/*
** kaapi_impl.h
** xkaapi
** 
** Created on Tue Mar 31 15:19:09 2009
** Copyright 2009 INRIA.
**
** Contributors :
**
** thierry.gautier@inrialpes.fr
** fabien.lementec@gmail.com / fabien.lementec@imag.fr
** Joao.Lima@imag.fr
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

#ifndef MATRIX_CPU_INL_INCLUDED
# define MATRIX_CPU_INL_INCLUDED

#include <string>
#include "kaapi++"

extern "C" {
#include <cblas.h>   // 
#include <clapack.h> // assume MKL/ATLAS clapack version
#include <lapacke.h>
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

};

template<class T>
struct CLAPACK {
  typedef T value_type;
  static int getrf(const enum CBLAS_ORDER Order, const int M, const int N,
                   value_type *A, const int lda, int *ipiv);
  static int potrf(const enum ATLAS_ORDER Order, const enum ATLAS_UPLO Uplo,
                   const int N, value_type *A, const int lda);
};

template<class T>
struct LAPACKE {
  typedef T value_type;
  static lapack_int lacpy( int matrix_order, char uplo, lapack_int m,
                             lapack_int n, const value_type* a, lapack_int lda, value_type* b,
                             lapack_int ldb );
  static lapack_int larnv( lapack_int idist, lapack_int* iseed, lapack_int n,
                             value_type* x );
  static value_type lamch_work( char cmach );

  static value_type lange_work( int matrix_order, char norm, lapack_int m,
                                  lapack_int n, const value_type* a, lapack_int lda,
                                  value_type* work );

  static lapack_int laswp_work( int matrix_order, lapack_int n, value_type* a,
                                  lapack_int lda, lapack_int k1, lapack_int k2,
                                  const lapack_int* ipiv, lapack_int incx );

  static lapack_int lacpy_work( int matrix_order, char uplo, lapack_int m,
                                lapack_int n, const value_type* a, lapack_int lda,
                                value_type* b, lapack_int ldb );
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
struct LAPACKE<double> {
  typedef double value_type;
  static lapack_int lacpy( int matrix_order, char uplo, lapack_int m,
                             lapack_int n, const value_type* a, lapack_int lda, value_type* b,
                             lapack_int ldb )
  { return LAPACKE_dlacpy( matrix_order, uplo, m, n, a, lda, b, ldb ); }
                           
  static lapack_int larnv( lapack_int idist, lapack_int* iseed, lapack_int n,
                             value_type* x )
  { return LAPACKE_dlarnv( idist, iseed, n, x ); }

  static value_type lamch_work( char cmach )
  { return LAPACKE_dlamch_work( cmach ); }

  static value_type lange_work( int matrix_order, char norm, lapack_int m,
                                  lapack_int n, const value_type* a, lapack_int lda,
                                  value_type* work )
  { return LAPACKE_dlange_work( matrix_order, norm, m, n, a, lda, work ); }

  static lapack_int laswp_work( int matrix_order, lapack_int n, value_type* a,
                                  lapack_int lda, lapack_int k1, lapack_int k2,
                                  const lapack_int* ipiv, lapack_int incx )
  { return LAPACKE_dlaswp_work( matrix_order, n, a, lda, k1, k2, ipiv, incx ); }

  static lapack_int lacpy_work( int matrix_order, char uplo, lapack_int m,
                                lapack_int n, const value_type* a, lapack_int lda,
                                value_type* b, lapack_int ldb )
  { return LAPACKE_dlacpy_work( matrix_order, uplo, m, n, a, lda, b, ldb ); }
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


template<>
struct LAPACKE<float> {
  typedef float value_type;
  static lapack_int lacpy( int matrix_order, char uplo, lapack_int m,
                             lapack_int n, const value_type* a, lapack_int lda, value_type* b,
                             lapack_int ldb )
  { return LAPACKE_slacpy( matrix_order, uplo, m, n, a, lda, b, ldb ); }
                           
  static lapack_int larnv( lapack_int idist, lapack_int* iseed, lapack_int n,
                             value_type* x )
  { return LAPACKE_slarnv( idist, iseed, n, x ); }

  static value_type lamch_work( char cmach )
  { return LAPACKE_slamch_work( cmach ); }

  static value_type lange_work( int matrix_order, char norm, lapack_int m,
                                  lapack_int n, const value_type* a, lapack_int lda,
                                  value_type* work )
  { return LAPACKE_slange_work( matrix_order, norm, m, n, a, lda, work ); }

  static lapack_int laswp_work( int matrix_order, lapack_int n, value_type* a,
                                  lapack_int lda, lapack_int k1, lapack_int k2,
                                  const lapack_int* ipiv, lapack_int incx )
  { return LAPACKE_slaswp_work( matrix_order, n, a, lda, k1, k2, ipiv, incx ); }

  static lapack_int lacpy_work( int matrix_order, char uplo, lapack_int m,
                                lapack_int n, const value_type* a, lapack_int lda,
                                value_type* b, lapack_int ldb )
  { return LAPACKE_slacpy_work( matrix_order, uplo, m, n, a, lda, b, ldb ); }
};


#if defined(CONFIG_USE_PLASMA)
static inline int
convertToSidePlasma( int side ) 
{
    switch (side) {
	case CblasRight:
            return PlasmaRight;
	case CblasLeft:
        default:
         return PlasmaLeft;
    }        
}

static inline int
convertToTransPlasma( int trans ) 
{
    switch(trans) {
        case CblasNoTrans:
            return PlasmaNoTrans;
        case CblasTrans:
            return PlasmaTrans;
        case CblasConjTrans:
            return PlasmaConjTrans;                        
        default:
            return PlasmaNoTrans;
    }
}
#endif

/* Note: this file defines 
    - tasks for some BLAS-3 over double/float 
    - tasks for some LAPACK functions.
  The LAPACK C interface is assumed to be the ATLAS version
  where it exist both version of LU or LLt factorization for
  row major or column major representation of matrix.
  
  Here, we assume row major representation of matrix. The
*/


/* Task Print Matrix
 * this task prints the matrix using Maple matrix format
 */
template<typename T>
struct TaskBodyCPU<TaskPrintMatrix<T> > {
  void operator() ( std::string msg, ka::range2d_r<T> A  )
  {
    size_t d0 = A->dim(0);
    size_t d1 = A->dim(1);
    std::cerr << msg << " :=matrix(" << d0 << "," << d1 << ")( [" << std::endl;
    for (size_t i=0; i < d0; ++i)
    {
      std::cerr << "[";
      for (size_t j=0; j < d1; ++j)
      {
        //std::cout << std::setw(18) << std::setprecision(15) << std::scientific << A(i,j) << (j == d1-1 ? "" : ", ");
	fprintf( stderr, " %4.2f", A(i, j));
      }
      std::cerr << "]" << (i == d0-1 ? ' ' : ',') << std::endl;
    }
    std::cerr << "]);" << std::endl;
  }
};

template<>
struct TaskBodyCPU<TaskPrintMatrixInt> {
  void operator() ( std::string msg, ka::range2d_r<int> A  )
  {
    size_t d0 = A->dim(0);
    size_t d1 = A->dim(1);
    std::cerr << msg << " :=matrix( [" << std::endl;
    for (size_t i=0; i < d0; ++i)
    {
      std::cerr << "[";
      for (size_t j=0; j < d1; ++j)
      {
        //std::cout << std::setw(18) << std::setprecision(15) << std::scientific << A(i,j) << (j == d1-1 ? "" : ", ");
	fprintf( stderr, " %.2d", A(i, j));
      }
      std::cerr << "]" << (i == d0-1 ? ' ' : ',') << std::endl;
    }
    std::cerr << "]);" << std::endl;
  }
};


/* =================== CBLAS routines =================== */

/* Solve : L(rk,rk) * X =  * A(rk,rj) 
    ie:  X <- L(rk,rk)^-1 * A(rk,rj) 
*/
template<typename T>
struct TaskBodyCPU<TaskTRSM_left<T> > {
  void operator()( ka::range2d_r<T> Akk, ka::range2d_rw<T> Akj )
  { 
    const T* const a = Akk->ptr();
    const int lda = Akk->lda();

    T* const b = Akj->ptr();
    const int ldb   = Akj->lda();
    const int n     = Akj->dim(0);
    const int m     = Akj->dim(1);

    CBLAS<T>::trsm
    (
      CblasRowMajor, CblasLeft, CblasLower, CblasNoTrans, CblasUnit,
      n, m, 1., a, lda, b, ldb
    );
  }
};


/* Solve : X * U(rk,rk) =  A(ri,rk) 
    ie:  X <- A(ri,rk) * U(rk,rk)^-1
*/
template<typename T>
struct TaskBodyCPU<TaskTRSM_right<T> > {
  void operator()( ka::range2d_r<T> Akk, ka::range2d_rw<T> Aik )
  {
    const T* const a = Akk->ptr();
    const int lda = Akk->lda();

    T* const b = Aik->ptr();
    const int ldb = Aik->lda();
    const int n = Aik->dim(0); // b.rows();
    const int m = Aik->dim(1); // b.cols();

    CBLAS<T>::trsm
    (
      CblasRowMajor, CblasRight, CblasUpper, CblasNoTrans, CblasNonUnit,
      n, m, 1., a, lda, b, ldb
    );
  }
};


/* DGEMM rountine to compute
    Aij <- alpha* Aik * Akj + beta Aij
*/
template<typename T>
struct TaskBodyCPU<TaskGEMM<T> > {
  void operator()
  (
    CBLAS_ORDER		   order, 
    CBLAS_TRANSPOSE transA,
    CBLAS_TRANSPOSE transB,
    T alpha,
    ka::range2d_r<T> Aik,
    ka::range2d_r<T> Akj,
    T beta,
    ka::range2d_rw<T> Aij
  )
  {
    const T* const a = Aik->ptr();
    const T* const b = Akj->ptr();
    T* const c       = Aij->ptr();

    const int m = Aik->dim(0); 
    const int n = Aik->dim(1); // eq. to Akj->rows();
    const int k = Akj->dim(1); 

    const int lda = Aik->lda();
    const int ldb = Akj->lda();
    const int ldc = Aij->lda();

    KAAPI_TIMING_BEGIN();
    CBLAS<T>::gemm
    (
      order, transA, transB,
      m, n, k, alpha, a, lda, b, ldb, beta, c, ldc
    );
    KAAPI_TIMING_END("CPU DGEMM", n);
  }
};



/* Rank k update
*/
template<typename T>
struct TaskBodyCPU<TaskSYRK<T> > {
  void operator()(
    CBLAS_ORDER		    order, 
    CBLAS_UPLO        uplo,
    CBLAS_TRANSPOSE   trans,
    T                 alpha,
    ka::range2d_r <T> A, 
    T                 beta,
    ka::range2d_rw<T> C 
  )
  {
    const int n     = C->dim(0); 
    //const int k     = A->dim(1); // eq. to Akj->rows();
    const int k = (trans == CblasNoTrans ? A->dim(0) : A->dim(1) );
    const int lda   = A->lda();
    const T* const a = A->ptr();
    const int ldc   = C->lda();
    T* const c = C->ptr();

    KAAPI_TIMING_BEGIN();
    CBLAS<T>::syrk
    (
      order, uplo, trans,
      n, k, alpha, a, lda, beta, c, ldc
    );
    KAAPI_TIMING_END("CPU DSYRK", n);

  }
};


/* DTRSM
*/
template<typename T>
struct TaskBodyCPU<TaskTRSM<T> > {
  void operator()( 
    CBLAS_ORDER		    order, 
    CBLAS_SIDE        side,
    CBLAS_UPLO        uplo,
    CBLAS_TRANSPOSE   transA,
    CBLAS_DIAG        diag,
    T                 alpha,
    ka::range2d_r <T> A, 
    ka::range2d_rw<T> C
  )
  {
    const T* const a = A->ptr();
    const int lda = A->lda();

    T* const c = C->ptr();
    const int ldc = C->lda();

    const int n = C->dim(0);
    //const int k = C->dim(1);
    const int k = (transA == CblasNoTrans ? A->dim(1) : A->dim(0) );

    KAAPI_TIMING_BEGIN();
    CBLAS<T>::trsm
    (
      order, side, uplo, transA, diag,
      n, k, alpha, a, lda, c, ldc
    );
    KAAPI_TIMING_END("CPU DTRSM", n);
  }
};


/* =================== CLAPACK routines =================== */

/* Compute inplace LU factorization of A.
*/
template<typename T>
struct TaskBodyCPU<TaskGETRF<T> > {
  void operator()( 
    CBLAS_ORDER order, 
    ka::range2d_rw<T> A, 
    ka::range1d_w<int> piv
  )
  {
    const int m     = A->dim(0); 
    const int n     = A->dim(1); 
    const int lda   = A->lda();
    T* const a      = A->ptr();
    int* const ipiv = piv->ptr();

#if defined(CONFIG_USE_PLASMA)
/* TODO: wrap the call to a PLASMA<T>:: struct in order to have code specialization
   for double or float */
    const int ib = IB; // from PLASMA
    int info;
    CORE_dgetrf_incpiv(
	    m, n, ib, 
	    a, lda,
	    ipiv,
	    &info
	);
#else
    CLAPACK<T>::getrf(CblasRowMajor, m, n, a, lda, ipiv);
#endif
  }
};

/* Compute inplace LU factorization of A.
*/
template<typename T>
struct TaskBodyCPU<TaskGETRFNoPiv<T> > {
  void operator()( 
    CBLAS_ORDER order, 
    ka::range2d_rw<T> A
  )
  {
    const int m        = A->dim(0); 
    const int n        = A->dim(0); 
    const int lda      = A->lda();
    T* const a    = A->ptr();
    const int ione   = 1;
    int* piv = (int*) calloc(m, sizeof(int));

    CLAPACK<T>::getrf( order, m, n, a, lda, piv );
    LAPACKE<T>::laswp( order, m, a, lda, ione, n, piv, ione);
    free( piv );
  }
};


/* Compute inplace LLt factorization of A, ie L such that A = L * Lt
   with L lower triangular.
*/
template<typename T>
struct TaskBodyCPU<TaskPOTRF<T> > {
  void operator()( 
    CBLAS_ORDER order, CBLAS_UPLO uplo, ka::range2d_rw<T> A 
  )
  {
    const int n     = A->dim(0); 
    const int lda   = A->lda();
    T* const a = A->ptr();

    KAAPI_TIMING_BEGIN();
#if 1
    CLAPACK<T>::potrf( order, uplo, n, a, lda );
#else
    LAPACKE_dpotrf_work(
//	    convertToOrderLapack(order),
	    LAPACK_COL_MAJOR,
//	    convertToFillModeLapack(uplo),
	    'l',
	    n, a, lda );
#endif
    KAAPI_TIMING_END("CPU DPOTRF", n);
  }
};

template<typename T>
struct TaskBodyCPU<TaskLACPY<T> > {
  void operator()( 
	CBLAS_ORDER order,
	CBLAS_UPLO uplo,
	ka::range2d_rw<T> A,
	ka::range2d_rw<T> B
  )
  {
    const int m     = A->dim(0); 
    const int n = A->dim(1);
    const int lda   = A->lda();
    const int ldb = B->lda();
    T* const a = A->ptr();
    T* const b = B->ptr();

    LAPACKE<T>::lacpy( order, uplo, m, n, a, lda, b, ldb );
  }
};

template<typename T>
struct TaskBodyCPU<TaskLARNV<T> > {
	void operator() (
		ka::range2d_w<T> A
	)
	{
		const int IDIST = 1;
		int ISEED[4] = {0,0,0,1};
 		const int mn     = A->dim(0)*A->dim(1); 
		T* const a = A->ptr();

		LAPACKE<T>::larnv( IDIST, ISEED, mn, a );
	}
};

template<>
struct TaskBodyCPU<TaskPlasmaDGESSM> {
  void operator()( 
    CBLAS_ORDER order, 
    ka::range1d_r<int> piv,
    ka::range2d_r<double> L, 
    ka::range2d_rw<double> A
  )
  {
#if defined(CONFIG_USE_PLASMA)
    const int m = A->dim(0); 
    const int n = A->dim(1);
    const int k = A->dim(1); /* TODO check */
    const int lda = A->lda();
    const int ldl = L->lda();
    double* const a = A->ptr();
    double* const l = (double*)L->ptr();
    int* const ipiv = (int*)piv->ptr();

    const int ib = IB; // from PLASMA
    CORE_dgessm( 
	    m, n, k, ib,
	    ipiv,
	    l, ldl,
	    a, lda
	);
#else
#endif
  }
};

template<>
struct TaskBodyCPU<TaskPlasmaDTSTRF> {
  void operator()( 
    CBLAS_ORDER order, 
    int nb,
    ka::range2d_rw<double> U, 
    ka::range2d_rw<double> A,
    ka::range2d_rw<double> L,
    ka::range1d_w<int> piv,
    ka::range2d_rw<double> WORK
  )
  {
#if defined(CONFIG_USE_PLASMA)
    const int m = A->dim(0); 
    const int n = A->dim(1);
    const int lda = A->lda();
    const int ldl = L->lda();
    const int ldu = U->lda();
    const int ldw = WORK->lda();
    double* const a = A->ptr();
    double* const l = L->ptr();
    double* const u = U->ptr();
    double* const work = WORK->ptr();
    int* const ipiv = piv->ptr();

#if 0
    fprintf( stdout, "TaskDTSTRF L(%lu,%lu,%d)\n", L->dim(0), L->dim(1), L->lda() );
    fflush(stdout);
#endif
    const int ib = IB; // from PLASMA
    int info;
    /* TODO */
    CORE_dtstrf(
	m, n, ib, nb,
	u, ldu,
	a, lda,
	l, ldl,
	ipiv,
	work, ldw,
	&info
    );
#else
#endif
  }
};

template<>
struct TaskBodyCPU<TaskPlasmaDSSSSM> {
  void operator()( 
    CBLAS_ORDER order, 
    ka::range2d_rw<double> A1, 
    ka::range2d_rw<double> A2,
    ka::range2d_r<double> L1,
    ka::range2d_r<double> L2,
    ka::range1d_r<int> piv
  )
  {
#if defined(CONFIG_USE_PLASMA)
    const int m1 = A1->dim(0); 
    const int n1 = A1->dim(1);
    const int m2 = A2->dim(0); 
    const int n2 = A2->dim(1);
    const int k = L1->dim(0);
    const int lda1 = A1->lda();
    const int lda2 = A2->lda();
    const int ldl1 = L1->lda();
    const int ldl2 = L2->lda();
    double* const a1 = A1->ptr();
    double* const a2 = A2->ptr();
    double* const l1 = (double*)L1->ptr();
    double* const l2 = (double*)L2->ptr();
    int* const ipiv = (int*)piv->ptr();

#if 0
    fprintf( stdout, "TaskDSSSSM L(%lu,%lu), k=%d\n",
	    L1->dim(0), L1->dim(1), k );
#endif
    const int ib = IB; // from PLASMA
    CORE_dssssm( 
	m1, n1,
	m2, n2,
	k, ib,
	a1, lda1,
	a2, lda2,
	l1, ldl1,
	l2, ldl2,
	ipiv
    );
#else
#endif
  }
};

#if defined(CONFIG_USE_DOUBLE)
template<>
struct TaskBodyCPU<TaskPlasmaDGEQRT> {
  void operator()( 
	CBLAS_ORDER order,
	ka::range2d_rw<double> A,
	ka::range2d_w<double>  T,
	ka::range1d_w<double>  TAU,
	ka::range1d_w<double>  WORK
  )
  {
    const int m = A->dim(0); 
    const int n = A->dim(1);
    const int lda = A->lda();
    const int ldt = T->lda();
    double* const a = A->ptr();
    double* const t = T->ptr();
    double* const work = WORK->ptr();
    const int ib = IB; // PLASMA(control/auxiliary.c)

#if defined(CONFIG_USE_PLASMA)
    double* const tau = TAU->ptr();
    CORE_dgeqrt(
	m, n, ib,
	a, lda,
	t, ldt,
	tau, work
    );
#else
    LAPACKE_dgeqrt_work( order, 
	m, n, ib,
	a, lda,
	t, ldt,
	work
	);
#endif
  }
};

template<>
struct TaskBodyCPU<TaskPlasmaDORMQR> {
  void operator()( 
	CBLAS_ORDER		order,
	CBLAS_SIDE		side,
	CBLAS_TRANSPOSE		trans,
	ka::range2d_r<double>  A,
	ka::range2d_w<double>  T,
	ka::range2d_rw<double> C,
	ka::range1d_rw<double> WORK
  )
  {
    const int m = A->dim(0); 
    const int n = A->dim(1);
    const int lda = A->lda();
    const int ldt = T->lda();
    double* const a = A->ptr();
    double* const t = T->ptr();
    double* const work = WORK->ptr();
    const int ib = IB; // PLASMA(control/auxiliary.c)

#if defined(CONFIG_USE_PLASMA)
    const int k = std::min(m, n);
    const int ldc = C->lda();
    const int ldwork = WORK->size();
    double* const c = C->ptr();
    CORE_dormqr(
	    convertToSidePlasma(side),
	    convertToTransPlasma(trans),
	    m, n, k, ib,
	    a, lda,
	    t, ldt,
	    c, ldc,
	    work, ldwork
    );
#else
    LAPACKE_dgeqrt_work( order, 
	m, n, ib,
	a, lda,
	t, ldt,
	work
	);
#endif
  }
};

template<>
struct TaskBodyCPU<TaskPlasmaDTSQRT> {
  void operator()( 
	CBLAS_ORDER order,
	ka::range2d_rw<double> A1,
	ka::range2d_rw<double> A2,
	ka::range2d_w<double>  T,
	ka::range1d_w<double>  TAU,
	ka::range1d_w<double>  WORK
  )
  {
#if defined(CONFIG_USE_PLASMA)
    const int m = A2->dim(1); 
    const int n = A1->dim(0);
    const int lda1 = A1->lda();
    const int lda2 = A2->lda();
    const int ldt = T->lda();
    double* const a1 = A1->ptr();
    double* const a2 = A2->ptr();
    double* const t = T->ptr();
    double* const work = WORK->ptr();
    double* const tau = TAU->ptr();
    const int ib = IB; // PLASMA(control/auxiliary.c)

    CORE_dtsqrt(
	m, n, ib,
	a1, lda1,
	a2, lda2,
	t, ldt,
	tau, work
    );
#endif
  }
};

template<>
struct TaskBodyCPU<TaskPlasmaDTSMQR> {
  void operator()( 
	CBLAS_ORDER		order,
	CBLAS_SIDE		side,
	CBLAS_TRANSPOSE		trans,
	ka::range2d_rw<double> A1,
	ka::range2d_rw<double> A2,
	ka::range2d_r<double>  V,
	ka::range2d_w<double>  T,
	ka::range1d_w<double>  WORK
  )
  {
#if defined(CONFIG_USE_PLASMA)
    const int m1 = A1->dim(0); 
    const int n1 = A1->dim(1);
    const int m2 = A2->dim(0); 
    const int n2 = A2->dim(1);
    const int lda1 = A1->lda();
    const int lda2 = A2->lda();
    const int ldv = V->lda();
    const int ldt = T->lda();
    double* const a1 = A1->ptr();
    double* const a2 = A2->ptr();
    double* const v = V->ptr();
    double* const t = T->ptr();
    double* const work = WORK->ptr();
    const int ib = IB; // PLASMA(control/auxiliary.c)
    const int k = A1->dim(1);
    const int ldwork = WORK->size();

    CORE_dtsmqr(
	convertToSidePlasma(side),
	convertToTransPlasma(trans),
	m1, n1,
	m2, n2,
	k, ib,
	a1, lda1,
	a2, lda2,
	v, ldv,
	t, ldt,
	work, ldwork
    );
#endif
  }
};
#endif /* CONFIG_USE_DOUBLE */

#endif /* ! MATRIX_CPU_INL_INCLUDED */
