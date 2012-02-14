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

#include "../matrix/matrix.h"

/* Task Print Matrix LU
 * assume that the matrix stores an LU decomposition with L lower triangular matrix with unit diagonal
   and U upper triangular matrix, then print both L and U using the Maple matrix format.
 */
struct TaskPrintMatrixLU : public ka::Task<2>::Signature<std::string,  ka::R<ka::range2d<double_type> > > {};
template<>
struct TaskBodyCPU<TaskPrintMatrixLU> {
  void operator() ( std::string msg, ka::range2d_r<double_type> A  )
  {
    size_t d0 = A.dim(0);
    size_t d1 = A.dim(1);
    std::cout << "U :=matrix( [" << std::endl;
    for (size_t i=0; i < d0; ++i)
    {
      std::cout << "[";
      for (size_t j=0; j<i; ++j)
      {
        std::cout << std::setw(18) 
                  << std::setprecision(15) 
                  << std::scientific 
                  << 0 << (j == d1-1 ? "" : ", ");
      }
      for (size_t j=i; j < d1; ++j)
      {
        std::cout << std::setw(18) 
                  << std::setprecision(15) 
                  << std::scientific 
                  << A(i,j) << (j == d1-1 ? "" : ", ");
      }
      std::cout << "]" << (i == d0-1 ? ' ' : ',') << std::endl;
    }
    std::cout << "]);" << std::endl;

    std::cout << "L :=matrix( [" << std::endl;
    for (size_t i=0; i < d0; ++i)
    {
      std::cout << "[";
      for (size_t j=0; j < i; ++j)
      {
        std::cout << std::setw(18) 
                  << std::setprecision(15) 
                  << std::scientific 
                  << A(i,j) << (j == d1-1 ? "" : ", ");
      }
      std::cout << std::setw(18) 
                << std::setprecision(15) 
                << std::scientific 
                << 1 << (i == d1-1 ? "" : ", ");
      for (size_t j=i+1; j<d1; ++j)
      {
        std::cout << std::setw(18) 
                  << std::setprecision(15) 
                  << std::scientific << 0 << (j == d1-1 ? "" : ", ");
      }
      std::cout << "]" << (i == d0-1 ? ' ' : ',') << std::endl;
    }
    std::cout << "]);" << std::endl;
    std::cout << "evalm( L &* U  - A);" << std::endl;
  }
};

#if 1
struct TaskNormMatrix: public ka::Task<5>::Signature
<
	CBLAS_ORDER,
	CBLAS_UPLO,
	ka::W<double_type>, /* norm */
	ka::RW<ka::range2d<double_type> >, /* A */
	ka::RW<ka::range2d<double_type> >  /* LU */
>{};
template<>
struct TaskBodyCPU<TaskNormMatrix> {
	void operator() (
		CBLAS_ORDER order,
		CBLAS_UPLO uplo,
		ka::pointer_w<double_type> norm, 
		ka::range2d_rw<double_type> A,
		ka::range2d_rw<double_type> LU
	)
	{
		int N = A.dim(0);
		int M = A.dim(1);
		int lda = A.lda();
		int min_mn = N;
		int i, j;
		double_type  alpha = 1;
		double_type beta  = 0;
		double_type work[1], matnorm, residual;
		       
		double_type *L = (double_type *)malloc(N*N*(sizeof(double_type)));
		double_type *U = (double_type *)malloc(N*N*(sizeof(double_type)));
		memset( L, 0, N*N*sizeof(double_type) );
		memset( U, 0, N*N*sizeof(double_type) );

		LAPACKE_lacpy( order, 'l', N, N, LU.ptr(), LU.lda(), L, N );
		LAPACKE_lacpy( order, 'u', N, N, LU.ptr(), LU.lda(), U, N );

		for(j=0; j<min_mn; j++)
		L[j+j*M] = 1.0;

		matnorm = LAPACKE_lange( order, 'f', N, N, A.ptr(), A.lda(), work);

		cblas_gemm( order, CblasNoTrans, CblasNoTrans, M, N, N,
		  alpha, L, M, U, N, beta, LU.ptr(), LU.lda() );

		double_type* dLU = LU.ptr();
		double_type* dA = A.ptr();
		for( j = 0; j < N; j++ ) {
		for( i = 0; i < M; i++ ) {
			dLU[i+j*lda] = dLU[i+j*lda] - dA[i+j*lda] ;
		}
		}
		residual = LAPACKE_lange( order, 'f', M, N, LU.ptr(), LU.lda(), work);

		free(L); free(U);

		*norm= residual / (matnorm * N);
	}
};

#endif

/* Compute the norm || A - L*U ||infinity
 * The norm value serves to detec an error in the computation
 */
#if 0
struct TaskNormMatrix: public ka::Task<3>::Signature<
  ka::W<double_type>, /* norm */
  ka::RW<ka::range2d<double_type> >, /* A */
  ka::RW<ka::range2d<double_type> >  /* LU */
>{};
template<>
struct TaskBodyCPU<TaskNormMatrix> {
  void operator() ( ka::pointer_w<double_type> norm, ka::range2d_rw<double_type> A, ka::range2d_rw<double_type> LU )
  {
    const double_type* dA = A.ptr();
    const double_type* dLU = LU.ptr();
    int lda_LU = LU.lda();
    int M = A.dim(0);
    int N = A.dim(1);
    *norm = 0.0;

    double_type max = 0.0;
    double_type* L = new double_type[M*N];
    double_type* U = new double_type[M*N];

    for (int i=0; i<M; ++i)
    {
      int j;
      /* copy L */
      const double_type* ALik = dLU+i*lda_LU;
      double_type* Lik = L+i*N;
      for (j=0; j < i; ++j, ++Lik, ++ALik)
        *Lik = *ALik;
      *Lik = 1.0; /* diag entry */
      ++Lik; 
      ++j;
      for ( ; j<N; ++j, ++Lik)
        *Lik = 0.0;
      
      /* copy U */
      const double_type* AUik = dLU+i*lda_LU + i;
      double_type* Uik = U+i*N;
      for (j=0; j < i; ++j, ++Uik)
        *Uik = 0.0;
      for ( ; j < N; ++j, ++Uik, ++AUik)
        *Uik = *AUik;
    }

    TaskBodyCPU<TaskDGEMM>()(
        CblasRowMajor,CblasNoTrans,CblasNoTrans,
        1.0,
        ka::range2d_r<double_type>(ka::range2d<double_type>(L, M, N, N)),
        ka::range2d_r<double_type>(ka::range2d<double_type>(U, M, N, N)),
        -1.0,
        A
    );
    max = 0;
    for (int i=0; i<M*N; ++i)
    {
        double_type error_ij = fabs(dA[i]);
        if (error_ij > max) 
          max = error_ij;
    }
    delete [] U;
    delete [] L;
    *norm = max;
  }
};
#endif

/* Block LU factorization
*/
struct TaskLU: public ka::Task<2>::Signature<
      ka::RPWP<ka::range2d<double_type> >,  /* A */
      ka::W<ka::range1d<int> > /* pivot */
>{};

static size_t global_blocsize = 2;

template<>
struct TaskBodyCPU<TaskLU> {
  void operator()( 
		  ka::range2d_rpwp<double_type> A,
		  ka::range1d_w<int> Piv )
  {
    size_t N = A.dim(0);
    size_t blocsize = global_blocsize;

    for (size_t k=0; k<N; k += blocsize)
    {
      ka::rangeindex rk(k, k+blocsize);
      // A(rk,rk) = L(rk,rk) * U(rk,rk) <- LU( A(rk,rk) 
      ka::Spawn<TaskDGETRFNoPiv>()( CblasColMajor, A(rk,rk) );

      for (size_t j=k+blocsize; j<N; j += blocsize)
      {
        ka::rangeindex rj(j, j+blocsize);
        // A(rk,rj) <- L(rk,rk)^-1 * A(rk,rj) 
        ka::Spawn<TaskDTRSM>()( CblasColMajor, CblasLeft, CblasLower, CblasNoTrans, CblasUnit, 1.0, A(rk,rk), A(rk,rj) );
      }
      for (size_t i=k+blocsize; i<N; i += blocsize)
      {
        ka::rangeindex ri(i, i+blocsize);
        // A(ri,rk) <- A(ri,rk) * U(rk,rk)^-1
        ka::Spawn<TaskDTRSM>()( CblasColMajor, CblasRight, CblasUpper, CblasNoTrans, CblasNonUnit, 1.0, A(rk,rk), A(ri,rk));
      }

      for (size_t i=k+blocsize; i<N; i += blocsize)
      {
        ka::rangeindex ri(i, i+blocsize);
        for (size_t j=k+blocsize; j<N; j += blocsize)
        {
          ka::rangeindex rj(j, j+blocsize);
          // A(ri,rj) <- A(ri,rj) - A(ri,rk)*A(rk,rj)
          ka::Spawn<TaskDGEMM>()( CblasColMajor, CblasNoTrans, CblasNoTrans, -1.0, A(ri,rk), A(rk,rj), 1.0, A(ri,rj));
        }
      }
    }
  }
};



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
    n = (n / block_size) * block_size;

    double t0, t1;
    double_type* dA = (double_type*) calloc(n* n, sizeof(double_type));
    int *dPiv = (int*) calloc( n, sizeof(int) );
    double_type* dAcopy;
    if (0 == dA) {
      std::cout << "Fatal Error. Cannot allocate matrices A, "
                << std::endl;
      return;
    }

    ka::array<2,double_type> A(dA, n, n, n);
    ka::Spawn<TaskDLARNV>()( A );
    ka::Sync();
    kaapi_memory_register( dA, n*n*sizeof(double_type) );

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

    ka::array<1,int> Piv(dPiv, n);

#if 0
    if (n <= 32) 
    {
      /* output respect the Maple format */
      ka::Spawn<TaskPrintMatrix<double_type> >()("A", A);
      ka::Sync();
    }
#endif

    // LU factorization of A using ka::
    double ggflops = 0;
    double gtime = 0;
    fprintf( stdout, "# size blocksize #threads time Gflops\n" );
    for (int i=0; i<niter; ++i)
    {
      t0 = kaapi_get_elapsedtime();
      ka::Spawn<TaskLU>(ka::SetStaticSched())( A, Piv );
      ka::Sync();
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
	/* If n is small, print the results */
	if (n <= 32) {
		ka::Spawn<TaskPrintMatrixLU>()( std::string(""), A );
		ka::Sync();
	}
	// /* compute the norm || A - L*U ||inf */
	{
	double_type norm;
	double_type* dAcopy2 = (double_type*) calloc(n* n, sizeof(double_type));
	double_type* dA2 = (double_type*) calloc(n* n, sizeof(double_type));
	memcpy(dAcopy2, dAcopy, n*n*sizeof(double_type) );
	memcpy(dA2, dAcopy, n*n*sizeof(double_type) );
    	ka::array<2,double_type> A2(dA2, n, n, n);

	t0 = kaapi_get_elapsedtime();
//	ka::Spawn<TaskNormMatrix>()( &norm, ka::array<2,double_type>(dAcopy, n, n, n), A );
#if 1
	ka::Spawn<TaskNormMatrix>()( CblasColMajor, CblasUpper,
		&norm, ka::array<2,double_type>(dAcopy, n, n, n), A );
#endif
	ka::Sync();
	t1 = kaapi_get_elapsedtime();
	std::cout << "# Error ||A-LU||inf " << norm 
		<< ", in " << (t1-t0) << " seconds." 
		<< std::endl;

	t0 = kaapi_get_elapsedtime();
	clapack_getrf( CblasColMajor, n, n, A2.ptr(), A2.lda(), dPiv );
//	ka::Spawn<TaskNormMatrix>()( &norm, ka::array<2,double_type>(dAcopy2, n, n, n), A2 );
#if 1
	ka::Spawn<TaskNormMatrix>()( CblasColMajor, CblasUpper, 
		&norm, ka::array<2,double_type>(dAcopy2, n, n, n), A2 );
#endif
	ka::Sync();
	t1 = kaapi_get_elapsedtime();
	std::cout << "# sequential Error ||A-LU||inf " << norm 
		<< ", in " << (t1-t0) << " seconds." 
		<< std::endl;
	free( dAcopy2 );
	free( dA2 );
	}

	free(dAcopy);
	}

    free(dA);
    free(dPiv);
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
