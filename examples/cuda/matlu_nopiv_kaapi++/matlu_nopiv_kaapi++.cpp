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

template<typename T>
static void generate_matrix( T* A, size_t N )
{
  srand48(0);
  for (size_t i = 0; i< (N*N); i++) {
    A[i] = drand48()*1000.0;
  }
}

/* Block LU factorization without pivoting
 */
template<typename T>
struct TaskLUNoPiv: public ka::Task<1>::Signature<
ka::RPWP<ka::range2d<T> >  /* A */
>{};

static size_t global_blocsize = 2;

template<typename T>
struct TaskBodyCPU<TaskLUNoPiv<T> > {
  void operator()(
                  const ka::StaticSchedInfo* info,
                  ka::range2d_rpwp<T> A )
  {
    size_t N = A->dim(0);
    size_t blocsize = global_blocsize;
    
    for (size_t k=0; k<N; k += blocsize)
    {
      ka::rangeindex rk(k, k+blocsize);
      // A(rk,rk) = L(rk,rk) * U(rk,rk) <- LU( A(rk,rk)
      ka::Spawn<TaskGETF2NoPiv<T> >( ka::SetArch(ka::ArchHost) )( CblasColMajor, A(rk,rk) );
      
      for (size_t j=k+blocsize; j<N; j += blocsize)
      {
        ka::rangeindex rj(j, j+blocsize);
        // A(rk,rj) <- L(rk,rk)^-1 * A(rk,rj)
        ka::Spawn<TaskTRSM<T> >( ka::SetArch(ka::ArchCUDA) )( CblasColMajor, CblasLeft, CblasLower, CblasNoTrans, CblasUnit, 1.0, A(rk,rk), A(rj,rk) );
      }
      for (size_t i=k+blocsize; i<N; i += blocsize)
      {
        ka::rangeindex ri(i, i+blocsize);
        // A(ri,rk) <- A(ri,rk) * U(rk,rk)^-1
        ka::Spawn<TaskTRSM<T> >( ka::SetArch(ka::ArchCUDA) )( CblasColMajor, CblasRight, CblasUpper, CblasNoTrans, CblasNonUnit, 1.0, A(rk,rk), A(rk,ri));
      }
      
      for (size_t i=k+blocsize; i<N; i += blocsize)
      {
        ka::rangeindex ri(i, i+blocsize);
        for (size_t j=k+blocsize; j<N; j += blocsize)
        {
          ka::rangeindex rj(j, j+blocsize);
          // A(ri,rj) <- A(ri,rj) - A(ri,rk)*A(rk,rj)
          ka::Spawn<TaskGEMM<T> >( ka::SetArch(ka::ArchCUDA) )( CblasColMajor, CblasNoTrans, CblasNoTrans, -1.0, A(rk,ri), A(rj,rk), 1.0, A(rj,ri));
        }
      }
    }
  }
};

template<typename T>
int check(int n, T* A, T* LU, int LDA)
{
  T norm, residual;
  T alpha, beta;
  int i,j;
  
  T *L       = (T *)malloc(n*n*sizeof(T));
  T *U       = (T *)malloc(n*n*sizeof(T));
  T *work     = (T *)malloc(n*sizeof(T));
  
  memset((void*)L, 0, n*n*sizeof(T));
  memset((void*)U, 0, n*n*sizeof(T));
  
  alpha= 1.0;
  beta= 0.0;
  
  LAPACKE<T>::lacpy_work(LAPACK_COL_MAJOR,'l', n, n, LU, LDA, L, n);
  LAPACKE<T>::lacpy_work(LAPACK_COL_MAJOR,'u', n, n, LU, LDA, U, n);
  
  for (j = 0; j < n; j++)
    L[j*n+j] = 1.0;
  
  norm = LAPACKE<T>::lange_work(LAPACK_COL_MAJOR, 'f', n, n, A, n, work);
  
  CBLAS<T>::gemm
  (
   CblasColMajor, CblasNoTrans, CblasNoTrans,
   n, n, n, alpha, L, n, U, n, beta, LU, n
   );
  
  /* Compute the Residual || A -L'L|| */
  for (j = 0; j < n; j++)
    for (i = 0; i < n; i++)
      LU[j*n+i] = LU[j*n+i] - A[j*n+i];
  
  residual = LAPACKE<T>::lange_work(LAPACK_COL_MAJOR, 'f', n, n, LU, n, work);
  fprintf( stdout, "# LU error: %e\n", residual/ (norm*n) );
  fflush(stdout);
  
  free(L); free(U); free(work);
  
  return 0;
}

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
    double_type* dAcopy;
    if (0 == dA) {
      std::cout << "Fatal Error. Cannot allocate matrices A, "
      << std::endl;
      return;
    }
    
    ka::array<2,double_type> A(dA, n, n, n);
    //TaskBodyCPU<TaskLARNV<double_type> >()( ka::range2d_w<double_type>(A) );
    generate_matrix<double_type>(dA, n);
    ka::Memory::Register( A );
    
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
      ka::Spawn<TaskLUNoPiv<double_type> >(ka::SetStaticSched())( A );
      ka::Sync();
#if CONFIG_USE_CUDA
      ka::MemorySync();
#endif
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
      check<double_type>( n, dAcopy, dA, n);
      free(dAcopy);
    }
    
    ka::Memory::Unregister( A );
    free(dA);
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
