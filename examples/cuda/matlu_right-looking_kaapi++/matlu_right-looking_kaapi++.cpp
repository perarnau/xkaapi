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
#include <iostream>
#include <iomanip>
#include <string>
#include <string.h>
#include <math.h>
#include <sys/types.h>
#include "kaapi++" // this is the new C++ interface for Kaapi

#if defined(CONFIG_USE_DOUBLE)
typedef double double_type;
#elif defined(CONFIG_USE_FLOAT)
typedef float double_type;
#endif

#include "../matrix/matrix.h"


/* Block LU factorization
*/
template<typename T>
struct TaskLU: public ka::Task<2>::Signature<
      ka::RPWP<ka::range2d<T> >,	/* A */ 
      ka::RPWP<ka::range1d<int> >		/* ipiv */ 
>{};

static size_t global_blocsize = 2;

template<typename T>
struct TaskGETRF2: public ka::Task<4>::Signature
<
  CBLAS_ORDER,               /* row / col */
  ka::RW<ka::range2d<T> >,   /* A */
  size_t,
  ka::RPWP<ka::range1d <int> >  /* pivot */
>{};

template<typename T>
struct TaskBodyCPU<TaskGETRF2<T> > {
  void operator()( 
    CBLAS_ORDER order, 
    ka::range2d_rw<T> A, 
    size_t size,
    ka::range1d_rpwp<int> piv
  )
  {
    const int m     = size; 
    const int n     = A->dim(1); 
    const int lda   = A->lda();
    T* const a      = A->ptr();
    int* const ipiv = piv->ptr();

#if 1
    fprintf(stdout, "TaskCPU DGETRF m=%d n=%d lda=%d A=%p ipiv=%p\n",
		m, n, lda, (void*)a, (void*)ipiv ); fflush(stdout);
#endif
#if defined(CONFIG_USE_PLASMA)
    const int ib = CONFIG_IB; // from PLASMA
    int info;
    PLASMA<T>::getrf(m, n, ib, a, lda, ipiv, &info);
#else
    CLAPACK<T>::getrf(order, m, n, a, lda, ipiv);
#endif
  }
};

template<typename T>
struct TaskBodyCPU<TaskLU<T> > {
  void operator()( 
		  ka::range2d_rpwp<T> A,
		  ka::range1d_rpwp<int>		ipiv
	  )
{
    size_t M = std::min(A->dim(0), A->dim(1));
    size_t N = A->dim(1);
    size_t blocsize = global_blocsize;
    const T zone = (T)1.0;
    const T mzone = (T)-1.0;

    for(size_t k=0; k < M; k += blocsize) {
	ka::rangeindex rk(k, k+blocsize);

	// A(rk,rk) = L(rk,rk) * U(rk,rk) * P(rk,rk) <- LU( A(rk,rk) )
	ka::Spawn<TaskGETRF<T> >(ka::SetArch(ka::ArchHost))(CblasColMajor, A(rk,rk), ipiv(rk));

	for (size_t n=k+blocsize; n<N; n += blocsize) {
	    ka::rangeindex rn(n, n+blocsize);

	    ka::Spawn<TaskLASWP<T> >(ka::SetArch(ka::ArchHost))
	      (CblasColMajor, A(rn, rk), 1, blocsize, ipiv(rk), 1);

	    ka::Spawn<TaskTRSM<T> >( ka::SetArch(ka::ArchCUDA) )
	      (CblasColMajor, CblasLeft, CblasLower, CblasNoTrans, CblasUnit, zone, A(rk,rk), A(rn,rk));

	  for( size_t m=k+blocsize; m<M; m += blocsize ){
	      ka::rangeindex rm(m, m+blocsize);
	      ka::Spawn<TaskGEMM<T> >( ka::SetArch(ka::ArchCUDA) )
		(CblasColMajor, CblasNoTrans, CblasNoTrans, mzone, A(rk,rm), A(rn,rk), zone, A(rn,rm));
	  }
	}
    }

    for(size_t k=0; k < M; k += blocsize) {
	ka::rangeindex rk(k, k+blocsize);
	for (size_t n= 0; n < k; n += blocsize) {
	    ka::rangeindex rn(n, n+blocsize);
	    ka::Spawn<TaskLASWP<T> >(ka::SetArch(ka::ArchHost))
	      (CblasColMajor, A(rn, rk), 1, blocsize, ipiv(rk), 1);
	}
    }
}
};

#if 0
template<typename T>
int check(int n, T* A, T* LU, int lda, int* ipiv)
{
  T* X = (T*)calloc(n*n, sizeof(T));
  T* B = (T*)calloc(n*n, sizeof(T));
  TaskBodyCPU<TaskLARNV<T> >()( ka::range2d_w<T>(ka::array<2,T>(X,n,n,n)) );
  memcpy(B, X, n*n*sizeof(T) );
  
  LAPACKE<T>::getrs(LAPACK_COL_MAJOR, 'n', n, n, LU, lda, ipiv, X, lda);

  T zone  =  1.0;
  T mzone = -1.0;
  T* work = (T*)malloc(n * sizeof(T));

  T Anorm = LAPACKE<T>::lange_work(LAPACK_COL_MAJOR, 'i', 
					     n, n, A, lda, work);
  T Xnorm = LAPACKE<T>::lange_work(LAPACK_COL_MAJOR, 'i', 
					     n, n, X, lda, work);
  T Bnorm = LAPACKE<T>::lange_work(LAPACK_COL_MAJOR, 'i', 
					     n, n, B, lda, work);

  CBLAS<T>::gemm
  (
      CblasColMajor, CblasNoTrans, CblasNoTrans,
      n, n, n, zone, A, lda, X, lda, mzone, B, lda
  );

  T Rnorm = LAPACKE<T>::lange_work(LAPACK_COL_MAJOR, 'i', 
					   n, n, B, lda, work);

  T eps = LAPACKE<T>::lamch_work('e');
  fprintf(stdout, "# ||Ax-B||_oo/((||A||_oo||x||_oo+||B||_oo).N.eps) = %e\n",
      Rnorm/((Anorm*Xnorm+Bnorm)*n*eps));
  
  if ( isnan(Rnorm/((Anorm*Xnorm+Bnorm)*n*eps)) || (Rnorm/((Anorm*Xnorm+Bnorm)*n*eps) > 10.0) ){
      fprintf(stdout, "# ERROR - the solution is suspicious ! \n");
  }
  else{
      fprintf(stdout, "# OK - The solution is CORRECT ! \n");
  }
  fflush(stdout);

  free(work);
  free( X );
  free( B );
  return 0;
}
#endif

#if 1
template<typename T>
int check(int n, T* A, T* LU, int lda, int* ipiv)
{
  T *work     = (T *)calloc(n, sizeof(T));
  int *ipiv2    = (int *)calloc(n, sizeof(int));
  
  LAPACKE<T>::getrf_work(LAPACK_COL_MAJOR, n, n, A, lda, ipiv2);

  /* Check ipiv */
  for(int i=0; i<n; i++) {
    if( ipiv[i] != (ipiv2[i]-1) ) {
	fprintf(stderr, "# LU (ipiv[%d] = %d, A[%d] = %e) / LAPACK (ipiv[%d] = %d, A[%d] = [%e])\n",
		i, ipiv[i],  i, (LU[  i * lda + i ]), 
		i, ipiv2[i], i, (A[ i * lda + i ])); 
	break;
    }
  }

  T Anorm = LAPACKE<T>::lange_work(LAPACK_COL_MAJOR, 'M', 
					     n, n, LU, lda, work);
  T Xnorm = LAPACKE<T>::lange_work(LAPACK_COL_MAJOR, 'M', 
					     n, n, A, lda, work);
  T Bnorm = 0.0;

  CBLAS<T>::axpy(n*n, -1.0, LU, 1, A, 1);

  T Rnorm = LAPACKE<T>::lange_work(LAPACK_COL_MAJOR, 'M', 
					   n, n, A, lda, work);

  T eps = LAPACKE<T>::lamch_work('e');
  fprintf(stdout, "# ||Ax-B||_oo/((||A||_oo||x||_oo+||B||_oo).N.eps) = %e \n",
      Rnorm/((Anorm*Xnorm+Bnorm)*n*eps));

  if ( isnan(Rnorm/((Anorm*Xnorm+Bnorm)*n*eps)) || (Rnorm/((Anorm*Xnorm+Bnorm)*n*eps) > 10.0) ){
      fprintf(stdout, "# ERROR - the solution is suspicious ! \n");
  }
  else{
      fprintf(stdout, "# OK - The solution is CORRECT ! \n");
  }
  fflush(stdout);

  free( ipiv2 );
  free( work );
  return 0;
}

#endif

/* Main of the program
*/
struct doit {
  void operator()(int argc, char** argv )
  {
    // matrix dimension
    int n = 32;
    if (argc > 1)
      n = atoi(argv[1]);

    // block count
    int block_size = 2;
    if (argc > 2)
      block_size = atoi(argv[2]);
      
    // Number of iterations
    int niter = 1;
    if (argc >3)
      niter = atoi(argv[3]);

    // Make verification ?
    int verif = 0;
    if (argc >4)
      verif = atoi(argv[4]);

    global_blocsize = block_size;

    double t0, t1;

    int* dIpiv = (int*) calloc(n, sizeof(int));
    if (0 == dIpiv) {
	std::cout << "Fatal Error. Cannot allocate matrices A, "
	    << std::endl;
	abort();
    }
    ka::range1d<int> ipiv(dIpiv, n);

    double_type* dAcopy;
    double_type* dA = (double_type*) calloc(n* n, sizeof(double_type));
    if (0 == dA) {
      std::cout << "Fatal Error. Cannot allocate matrices A, "
                << std::endl;
      return;
    }
    ka::array<2,double_type> A(dA, n, n, n);
    TaskBodyCPU<TaskLARNV<double_type> >()( ka::range2d_w<double_type>(A) );
#if CONFIG_USE_CUDA
    ka::Memory::Register( A );
    ka::Memory::Register( ipiv );
#endif

    if (verif) {
	/* copy the matrix to compute the norm */
	dAcopy = (double_type*) calloc(n* n, sizeof(double_type));
	memcpy(dAcopy, dA, n*n*sizeof(double_type) );
	if( dAcopy == NULL ) {
	      std::cout << "Fatal Error. Cannot allocate matrices A, "
			<< std::endl;
	      return;
	    }
    }

    // LU factorization of A using ka::
    double ggflops = 0;
    double gtime = 0;
    fprintf( stdout, "# size blocksize #threads time Gflops\n" );
    for (int i=0; i<niter; ++i) {

	t0 = kaapi_get_elapsedtime();
	ka::Spawn<TaskLU<double_type> >(ka::SetStaticSched())( A, ipiv );
	ka::Sync();
#if CONFIG_USE_CUDA
      ka::MemorySync();
#endif
	{ 
	  for(int i= block_size; i < n; i+= block_size) {
	      for (int j=0; j < block_size; j++)
		  dIpiv[i+j] = dIpiv[i+j] + i;
	  }
	}
	t1 = kaapi_get_elapsedtime();

	/* formula used by plasma */
	double fp_per_mul = 1;
	double fp_per_add = 1;
	double fmuls = (n * (1.0 / 3.0 * n )      * n);
	double fadds = (n * (1.0 / 3.0 * n - 0.5) * n);
	double gflops = 1e-9 * (fmuls * fp_per_mul + fadds * fp_per_add) / (t1-t0);
	gtime += t1-t0;
	ggflops += gflops;
	fprintf( stdout, "GETRF %6d %5d %5d %9.10f %9.6f\n",
	    (int)n,
	    (int)global_blocsize,
	    (int)kaapi_getconcurrency(),
	    t1-t0, gflops );
	fflush(stdout);
    }

    if (verif) {
      // /* compute the norm || A - L*U ||inf */
      check<double_type>(n, dAcopy, dA, n, dIpiv);
      free(dAcopy);
    }

#if CONFIG_USE_CUDA
    ka::Memory::Unregister( A );
    ka::Memory::Unregister( ipiv );
#endif
    free(dA);
    free(dIpiv);
  }
};


/* main entry point : Kaapi initialization
*/
int main(int argc, char** argv)
{
  try {
    /* Join the initial group of computation : it is defining
       when launching the program by a1run.
    */
    ka::Community com = ka::System::join_community( argc, argv );
    
    /* Start computation by forking the main task */
    ka::SpawnMain<doit>()(argc, argv); 
    
    /* Leave the community: at return to this call no more athapascan
       tasks or shared could be created.
    */
    com.leave();

    /* */
    ka::System::terminate();
  }
  catch (const std::exception& E) {
    ka::logfile() << "Catch : " << E.what() << std::endl;
  }
  catch (...) {
    ka::logfile() << "Catch unknown exception: " << std::endl;
  }
  
  return 0;
}
