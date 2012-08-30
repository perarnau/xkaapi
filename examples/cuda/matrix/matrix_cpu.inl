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
#include <cblas.h>
#include <clapack.h>
#include <lapacke.h>
}

#include "cblas.inl"
#include "clapack.inl"
#include "lapacke.inl"

#if defined(CONFIG_USE_PLASMA)
#include "plasma.inl"
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

#define TASK_GEMM_THRESHOLD	  256
template<typename T>
struct TaskBodyCPU<TaskRecursiveGEMM<T> > {
  void operator()
  (
    CBLAS_ORDER		   order, 
    CBLAS_TRANSPOSE transA,
    CBLAS_TRANSPOSE transB,
    T alpha,
    ka::range2d_r<T> A,
    ka::range2d_r<T> B,
    T beta,
    ka::range2d_rw<T> C
  )
  {
    const T* const a = A->ptr();
    const T* const b = B->ptr();
    T* const c       = C->ptr();

    const size_t M = A->dim(0);
    const size_t K = B->dim(0);
    const size_t N = B->dim(1);

    const int lda = A->lda();
    const int ldb = B->lda();
    const int ldc = C->lda();
    const int bloc = TASK_GEMM_THRESHOLD;

#if 1
    fprintf(stdout, "TaskCPU RecursiveGEMM m=%d n=%d k=%d A=%p alpha=%.2f B=%p beta=%.2f C=%p lda=%d ldb=%d ldc=%d\n", M, N, K, (void*)a, alpha, (void*)b, beta, (void*)c, lda, ldb, ldc ); fflush(stdout);
#endif
    if( M > TASK_GEMM_THRESHOLD )
    {
      for (size_t i=0; i<M; i += bloc)
      {
	ka::rangeindex ri(i, i+bloc);
	for (size_t j=0; j<N; j += bloc)
	{
	  ka::rangeindex rj(j, j+bloc);
	  for (size_t k=0; k<K; k += bloc)
	  {
	    ka::rangeindex rk(k, k+bloc);
	      ka::Spawn<TaskGEMM<T> >(ka::SetArch(ka::ArchHost))
		(
		 order, transA, transB,
		 alpha, A(rk,ri), B(rj,rk), beta, C(rj,ri)
		);
	  }
	}
      }
    } else {
      CBLAS<T>::gemm
      (
	order, transA, transB,
	M, N, K, alpha, a, lda, b, ldb, beta, c, ldc
      );
    }
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
    const int ib = IB; // from PLASMA
    int info;
    PLASMA<T>::getrf(m, n, ib, a, lda, ipiv, &info);
#else
    CLAPACK<T>::getrf(CblasColMajor, m, n, a, lda, ipiv);
#endif
  }
};

/* Compute inplace LU factorization of A.
*/
template<typename T>
struct TaskBodyCPU<TaskGETF2NoPiv<T> > {
  void operator()( 
    CBLAS_ORDER order, 
    ka::range2d_rw<T> A
  )
  {
    const int m        = A->dim(0); 
    const int n        = A->dim(0); 
    const int lda      = A->lda();
    T* const a    = A->ptr();

#if 0
    int res =
#endif 
    LAPACKE<T>::getf2_nopiv(
	((order == CblasColMajor) ? LAPACK_COL_MAJOR : LAPACK_ROW_MAJOR),
	m, n, a, lda );
#if 0
    fprintf(stdout, "TaskGETF2NoPiv n=%d res=%d\n", m, res );
    fflush(stdout);
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

    int res = CLAPACK<T>::getrf( order, m, n, a, lda, piv );
#if 1
    fprintf(stdout, "TaskGETRFNoPiv n=%d res=%d\n", m, res );
    fflush(stdout);
#endif
    LAPACKE<T>::laswp_work( order, m, a, lda, ione, n, piv, ione);
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
#if 0
    fprintf(stdout, "TaskCPU DPOTRF res=%d\n", res );
    fflush(stdout);
#endif
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

template<typename T>
struct TaskBodyCPU<TaskGEQRT<T> > {
  void operator()( 
	CBLAS_ORDER order,
	ka::range2d_rw<T> A,
	ka::range2d_w<T>  _T,
	ka::range1d_w<T>  TAU,
	ka::range1d_w<T>  WORK
  )
  {
    const int m = A->dim(0); 
    const int n = A->dim(1);
    const int lda = A->lda();
    const int ldt = _T->lda();
    T* const a = A->ptr();
    T* const t = _T->ptr();
    T* const work = WORK->ptr();
    const int ib = IB; // PLASMA(control/auxiliary.c)

#if defined(CONFIG_USE_PLASMA)
    T* const tau = TAU->ptr();
    int res = PLASMA<T>::geqrt( m, n, ib, a, lda, t, ldt, tau, work );
    if(res){
      fprintf(stdout, "TaskGEQRT error (%d) from PLASMA\n", res );
      fflush(stdout);
    }
#else
    LAPACKE<T>::geqrt(
	((order == CblasColMajor) ? LAPACK_COL_MAJOR : LAPACK_ROW_MAJOR),
	m, n, ib, a, lda, t, ldt, work );
#endif
  }
};

template<typename T>
struct TaskBodyCPU<TaskORMQR<T> > {
  void operator()( 
	CBLAS_ORDER		order,
	CBLAS_SIDE		side,
	CBLAS_TRANSPOSE		trans,
	ka::range2d_r<T>  A,
	ka::range2d_w<T>  _T,
	ka::range2d_rw<T> C,
	ka::range1d_rw<T> WORK
  )
  {
    const int m = A->dim(0); 
    const int n = A->dim(1);
    const int lda = A->lda();
    const int ldt = _T->lda();
    T* const a = A->ptr();
    T* const t = _T->ptr();
    T* const work = WORK->ptr();
    const int ib = IB; // PLASMA(control/auxiliary.c)

#if defined(CONFIG_USE_PLASMA)
    const int k = std::min(m, n);
    const int ldc = C->lda();
    const int ldwork = WORK->size();
    T* const c = C->ptr();
    PLASMA<T>::ormqr(side, trans, m, n, k, ib, a, lda, t, ldt, c, ldc, work, ldwork);
#endif
  }
};

template<typename T>
struct TaskBodyCPU<TaskTSQRT<T> > {
  void operator()( 
	CBLAS_ORDER order,
	ka::range2d_rw<T> A1,
	ka::range2d_rw<T> A2,
	ka::range2d_w<T>  _T,
	ka::range1d_w<T>  TAU,
	ka::range1d_w<T>  WORK
  )
  {
#if defined(CONFIG_USE_PLASMA)
    const int m = A2->dim(1); 
    const int n = A1->dim(0);
    const int lda1 = A1->lda();
    const int lda2 = A2->lda();
    const int ldt = _T->lda();
    T* const a1 = A1->ptr();
    T* const a2 = A2->ptr();
    T* const t = _T->ptr();
    T* const work = WORK->ptr();
    T* const tau = TAU->ptr();
    const int ib = IB; // PLASMA(control/auxiliary.c)

    PLASMA<T>::tsqrt( m, n, ib, a1, lda1, a2, lda2, t, ldt, tau, work );
#endif
  }
};

template<typename T>
struct TaskBodyCPU<TaskTSMQR<T> > {
  void operator()( 
	CBLAS_ORDER		order,
	CBLAS_SIDE		side,
	CBLAS_TRANSPOSE		trans,
	ka::range2d_rw<T> A1,
	ka::range2d_rw<T> A2,
	ka::range2d_r<T>  V,
	ka::range2d_w<T>  _T,
	ka::range1d_w<T>  WORK
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
    const int ldt = _T->lda();
    T* const a1 = A1->ptr();
    T* const a2 = A2->ptr();
    T* const v = V->ptr();
    T* const t = _T->ptr();
    T* const work = WORK->ptr();
    const int ib = IB; // PLASMA(control/auxiliary.c)
    const int k = A1->dim(1);
    const int ldwork = WORK->size();

    PLASMA<T>::tsmqr( side, trans, m1, n1, m2, n2, k, ib, a1, lda1, a2, lda2, v, ldv, t, ldt, work, ldwork );
#endif
  }
};

template<typename T>
struct TaskBodyCPU<TaskGESSM<T> > {
  void operator()( 
    CBLAS_ORDER order, 
    ka::range1d_r<int> piv,
    ka::range2d_r<T> L, 
    ka::range2d_rw<T> A
  )
  {
#if defined(CONFIG_USE_PLASMA)
    const int m = A->dim(0); 
    const int n = A->dim(1);
    const int k = A->dim(1); /* TODO check */
    const int lda = A->lda();
    const int ldl = L->lda();
    T* const a = A->ptr();
    T* const l = (T*)L->ptr();
    int* const ipiv = (int*)piv->ptr();

    const int ib = IB; // from PLASMA
    PLASMA<T>::gessm(m, n, k, ib, ipiv, l, ldl, a, lda);
#else
    /* TODO */
#endif
  }
};

template<typename T>
struct TaskBodyCPU<TaskTSTRF<T> > {
  void operator()( 
    CBLAS_ORDER order, 
    int nb,
    ka::range2d_rw<T> U, 
    ka::range2d_rw<T> A,
    ka::range2d_rw<T> L,
    ka::range1d_w<int> piv,
    ka::range2d_rw<T> WORK
  )
  {
#if defined(CONFIG_USE_PLASMA)
    const int m = A->dim(0); 
    const int n = A->dim(1);
    const int lda = A->lda();
    const int ldl = L->lda();
    const int ldu = U->lda();
    const int ldw = WORK->lda();
    T* const a = A->ptr();
    T* const l = L->ptr();
    T* const u = U->ptr();
    T* const work = WORK->ptr();
    int* const ipiv = piv->ptr();

#if 0
    fprintf( stdout, "TaskDTSTRF L(%lu,%lu,%d)\n", L->dim(0), L->dim(1), L->lda() );
    fflush(stdout);
#endif
    const int ib = IB; // from PLASMA
    int info;
    PLASMA<T>::tstrf(m, n, ib, nb, u, ldu, a, lda, l, ldl, ipiv, work, ldw, &info);
#else
    /* TODO */
#endif
  }
};

template<typename T>
struct TaskBodyCPU<TaskSSSSM<T> > {
  void operator()( 
    CBLAS_ORDER order, 
    ka::range2d_rw<T> A1, 
    ka::range2d_rw<T> A2,
    ka::range2d_r<T> L1,
    ka::range2d_r<T> L2,
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
    T* const a1 = A1->ptr();
    T* const a2 = A2->ptr();
    T* const l1 = (T*)L1->ptr();
    T* const l2 = (T*)L2->ptr();
    int* const ipiv = (int*)piv->ptr();

#if 0
    fprintf( stdout, "TaskDSSSSM L(%lu,%lu), k=%d\n",
	    L1->dim(0), L1->dim(1), k );
#endif
    const int ib = IB; // from PLASMA
    PLASMA<T>::ssssm(m1, n1, m2, n2, k, ib, a1, lda1, a2, lda2, l1, ldl1, l2, ldl2, ipiv);
#else
    /* TODO */
#endif
  }
};

#endif /* ! MATRIX_CPU_INL_INCLUDED */
