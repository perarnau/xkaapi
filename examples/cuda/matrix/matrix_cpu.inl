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
}

// specialize for double_type
#if CONFIG_USE_FLOAT

# define cblas_trsm	cblas_strsm
#define cblas_trmm	cblas_strmm
# define cblas_gemm	cblas_sgemm
# define cblas_syrk	cblas_ssyrk
# define cblas_getrf	cblas_sgetrf
#define cblas_axpy	cblas_saxpy
# define clapack_getrf	clapack_sgetrf
# define clapack_potrf	clapack_spotrf
#define LAPACKE_lacpy	LAPACKE_slacpy
#define LAPACKE_larnv	LAPACKE_slarnv
#define LAPACKE_lamch	LAPACKE_slamch
#define LAPACKE_lange	LAPACKE_slange_work
#define LAPACKE_laswp	LAPACKE_slaswp

#else

#define cblas_trsm	cblas_dtrsm
#define cblas_trmm	cblas_dtrmm
#define cblas_gemm	cblas_dgemm
#define cblas_syrk	cblas_dsyrk
#define cblas_axpy	cblas_daxpy
#define clapack_getrf	clapack_dgetrf
#define clapack_potrf	clapack_dpotrf
#define LAPACKE_lacpy	LAPACKE_dlacpy
#define LAPACKE_larnv	LAPACKE_dlarnv
#define LAPACKE_lamch	LAPACKE_dlamch
#define LAPACKE_lange	LAPACKE_dlange_work
#define LAPACKE_laswp	LAPACKE_dlaswp

#endif // CONFIG_USE_FLOAT


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
template<>
struct TaskBodyCPU<TaskPrintMatrix> {
  void operator() ( std::string msg, ka::range2d_r<double_type> A  )
  {
    size_t d0 = A.dim(0);
    size_t d1 = A.dim(1);
    std::cerr << msg << " :=matrix( [" << std::endl;
    for (size_t i=0; i < d0; ++i)
    {
      std::cerr << "[";
      for (size_t j=0; j < d1; ++j)
      {
        //std::cout << std::setw(18) << std::setprecision(15) << std::scientific << A(i,j) << (j == d1-1 ? "" : ", ");
	fprintf( stderr, " %.2f", A(i, j));
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
template<>
struct TaskBodyCPU<TaskDTRSM_left> {
  void operator()( ka::range2d_r<double_type> Akk, ka::range2d_rw<double_type> Akj )
  {
    const double_type* const a = Akk.ptr();
    const int lda = Akk.lda();

    double_type* const b = Akj.ptr();
    const int ldb   = Akj.lda();
    const int n     = Akj.dim(0);
    const int m     = Akj.dim(1);

    cblas_trsm
    (
     CblasRowMajor, CblasLeft, CblasLower, CblasNoTrans, CblasUnit,
     n, m, 1., a, lda, b, ldb
    );
  }
};


/* Solve : X * U(rk,rk) =  A(ri,rk) 
    ie:  X <- A(ri,rk) * U(rk,rk)^-1
*/
template<>
struct TaskBodyCPU<TaskDTRSM_right> {
  void operator()( ka::range2d_r<double_type> Akk, ka::range2d_rw<double_type> Aik )
  {
    const double_type* const a = Akk.ptr();
    const int lda = Akk.lda();

    double_type* const b = Aik.ptr();
    const int ldb = Aik.lda();
    const int n = Aik.dim(0); // b.rows();
    const int m = Aik.dim(1); // b.cols();

    cblas_trsm
    (
      CblasRowMajor, CblasRight, CblasUpper, CblasNoTrans, CblasNonUnit,
      n, m, 1., a, lda, b, ldb
    );
  }
};


/* DGEMM rountine to compute
    Aij <- alpha* Aik * Akj + beta Aij
*/
template<>
struct TaskBodyCPU<TaskDGEMM> {
  void operator()
  (
    CBLAS_ORDER		   order, 
    CBLAS_TRANSPOSE transA,
    CBLAS_TRANSPOSE transB,
    double_type alpha,
    ka::range2d_r<double_type> Aik,
    ka::range2d_r<double_type> Akj,
    double_type beta,
    ka::range2d_rw<double_type> Aij
  )
  {
    const double_type* const a = Aik.ptr();
    const double_type* const b = Akj.ptr();
    double_type* const c       = Aij.ptr();

    const int m = Aik.dim(0); 
    const int n = Aik.dim(1); // eq. to Akj.rows();
    const int k = Akj.dim(1); 

    const int lda = Aik.lda();
    const int ldb = Akj.lda();
    const int ldc = Aij.lda();

#if 0
    fprintf(stdout, "TaskCPU GEMM m=%d n=%d k=%d A=%p B=%p C=%p\n", m, n, k, (void*)a, (void*)b, (void*)c ); fflush(stdout);
#endif

    cblas_gemm
    (
      order, transA, transB,
      m, n, k, alpha, a, lda, b, ldb, beta, c, ldc
    );
  }
};



/* Rank k update
*/
template<>
struct TaskBodyCPU<TaskDSYRK> {
  void operator()(
    CBLAS_ORDER		   order, 
    CBLAS_UPLO uplo,
    CBLAS_TRANSPOSE trans,
    double_type alpha,
    ka::range2d_r <double_type>  A, 
    double_type beta,
    ka::range2d_rw<double_type> C 
  )
  {
    const int n     = A.dim(0); 
    const int k     = A.dim(1); // eq. to Akj.rows();
    const int lda   = A.lda();
    const double_type* const a = A.ptr();

    const int ldc   = C.lda();
    double_type* const c = C.ptr();

    cblas_syrk
    (
      order, uplo, trans,
      n, k, alpha, a, lda, beta, c, ldc
    );

  }
};


/* DTRSM
*/
template<>
struct TaskBodyCPU<TaskDTRSM> {
  void operator()( 
    CBLAS_ORDER		   order, 
    CBLAS_SIDE             side,
    CBLAS_UPLO             uplo,
    CBLAS_TRANSPOSE        transA,
    CBLAS_DIAG             diag,
    double_type                 alpha,
    ka::range2d_r <double_type> A, 
    ka::range2d_rw<double_type> C
  )
  {
    const double_type* const a = A.ptr();
    const int lda = A.lda();

    double_type* const c = C.ptr();
    const int ldc = C.lda();

    const int n = C.dim(0);
    const int k = (transA == CblasNoTrans ? A.dim(1) : A.dim(0) );

    cblas_trsm
    (
      order, side, uplo, transA, diag,
      n, k, alpha, a, lda, c, ldc
    );
  }
};


/* =================== CLAPACK routines =================== */

/* Compute inplace LU factorization of A.
*/
template<>
struct TaskBodyCPU<TaskDGETRF> {
  void operator()( 
    ka::range2d_rw<double_type> A, 
    ka::range1d_w<int> piv
  )
  {
    const int m        = A.dim(0); 
    const int n        = A.dim(0); 
    const int lda      = A.lda();
    double_type* const a    = A.ptr();
    int* const ipiv = piv.ptr();

    clapack_getrf(CblasRowMajor, m, n, a, lda, ipiv);
  }
};

/* Compute inplace LU factorization of A.
*/
template<>
struct TaskBodyCPU<TaskDGETRFNoPiv> {
  void operator()( 
	CBLAS_ORDER order, 
	ka::range2d_rw<double_type> A
  )
  {
    const int m        = A.dim(0); 
    const int n        = A.dim(0); 
    const int lda      = A.lda();
    double_type* const a    = A.ptr();
    const int ione   = 1;
    int* piv = (int*) calloc(m, sizeof(int));

    clapack_getrf( order, m, n, a, lda, piv );
    LAPACKE_laswp( order, m, a, lda, ione, n, piv, ione);
    free( piv );
  }
};


/* Compute inplace LLt factorization of A, ie L such that A = L * Lt
   with L lower triangular.
*/
template<>
struct TaskBodyCPU<TaskDPOTRF> {
  void operator()( 
    CBLAS_ORDER order, CBLAS_UPLO uplo, ka::range2d_rw<double_type> A 
  )
  {
    const int n     = A.dim(0); 
    const int lda   = A.lda();
    double_type* const a = A.ptr();
#if 0
    fprintf(stdout, "TaskCPU DPOTRF m=%d A=%p lda=%d\n", n, (void*)a, lda ); fflush(stdout);
#endif
    clapack_potrf( order, uplo, n, a, lda );
  }
};

template<>
struct TaskBodyCPU<TaskDLACPY> {
  void operator()( 
	CBLAS_ORDER order,
	CBLAS_UPLO uplo,
	ka::range2d_rw<double_type> A,
	ka::range2d_rw<double_type> B
  )
  {
    int m     = A.dim(0); 
    int n = A.dim(1);
    int lda   = A.lda();
    int ldb = B.lda();
    double_type* const a = A.ptr();
    double_type* const b = B.ptr();

    LAPACKE_lacpy( order, uplo, m, n, a, lda, b, ldb );
  }
};

template<>
struct TaskBodyCPU<TaskDLARNV> {
	void operator() (
		ka::range2d_w<double_type> A
	)
	{
		const int IDIST = 1;
		int ISEED[4] = {0,0,0,1};
 		const int mn     = A.dim(0)*A.dim(1); 
		double_type* const a = A.ptr();

		LAPACKE_larnv( IDIST, ISEED, mn, a );
	}
};

#endif /* ! MATRIX_CPU_INL_INCLUDED */
