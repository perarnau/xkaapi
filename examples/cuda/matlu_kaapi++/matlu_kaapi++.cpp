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

#if defined(CONFIG_USE_DOUBLE)
typedef double double_type;
#elif defined(CONFIG_USE_FLOAT)
typedef float double_type;
#endif

#include "../matrix/matrix.h"


/* Block LU factorization
 */
template<typename T>
struct TaskLU: public ka::Task<4>::Signature<
ka::RPWP<ka::range2d<T> >,	/* A */
ka::RPWP<ka::range2d<T> >,	/* L */
ka::RPWP<ka::range1d<int> >,		/* ipiv */
uintptr_t	/* WORK */
>{};

static size_t global_blocsize = 2;

template<typename T>
struct TaskBodyCPU<TaskLU<T> > {
  void operator()(
                  ka::range2d_rpwp<T> A,
                  ka::range2d_rpwp<T> L,
                  ka::range1d_rpwp<int> piv,
                  uintptr_t WORK
                  )
  {
    size_t N = A->dim(0);
    size_t blocsize = global_blocsize;
    int* ipiv = (int*)piv->ptr();
    int* rkpiv_ptr;
    int* rmpiv_ptr;
    
    for(size_t k=0; k < N; k += blocsize) {
      size_t ik = ((k+blocsize) < N) ? (k+blocsize) : N;
      ka::rangeindex rk(k, ik);
      rkpiv_ptr = ipiv+k*blocsize+k;
      ka::range1d<int> rkpiv(rkpiv_ptr, blocsize*blocsize);
      // A(rk,rk) = L(rk,rk) * U(rk,rk) * P(rk,rk) <- LU( A(rk,rk) )
      ka::Spawn<TaskGETRF<T> >(ka::SetArch(ka::ArchHost))( CblasColMajor, A(rk,rk), rkpiv );
      
      for (size_t n=k+blocsize; n<N; n += blocsize) {
        size_t in = ((n+blocsize) < N) ? (n+blocsize) : N;
        ka::rangeindex rn(n, in);
        ka::Spawn<kplasma::TaskGESSM<T> >(ka::SetArch(ka::ArchCUDA))(
                                                                     CblasColMajor,
                                                                     rkpiv,
                                                                     (uintptr_t)rkpiv_ptr,
                                                                     A(rk, rk),
                                                                     A(rn, rk)
                                                                     );
      }
      
      for( size_t m=k+blocsize; m<N; m += blocsize ){
        size_t im = ((m+blocsize) < N) ? (m+blocsize) : N;
        ka::rangeindex rm(m, im);
        rmpiv_ptr = ipiv+k*blocsize+m;
        ka::range1d<int> rmpiv(rmpiv_ptr, blocsize*blocsize);
        
        ka::Spawn<kplasma::TaskTSTRF<T> >(ka::SetArch(ka::ArchHost))(
                                                                     CblasColMajor, blocsize,
                                                                     A(rk, rk),
                                                                     A(rk, rm),
                                                                     L(rk, rm),
                                                                     rmpiv,
                                                                     WORK
                                                                     );
        
        for (size_t n=k+blocsize; n<N; n += blocsize) {
          size_t in = ((n+blocsize) < N) ? (n+blocsize) : N;
          ka::rangeindex rn(n, in);
          // A(rm,rn) <- A(rm,rn) - A(rm,rk)*A(rk,rn)
          ka::Spawn<kplasma::TaskSSSSM<T> >(ka::SetArch(ka::ArchCUDA))(
                                                                       CblasColMajor,
                                                                       A(rn, rk),
                                                                       A(rn, rm),
                                                                       L(rk, rm),
                                                                       A(rk, rm),
                                                                       (uintptr_t)rmpiv_ptr
                                                                       );
        }
      }
    }
  }
};

template<typename T>
int check(int n, T* A, T* LU, int LDA, int* piv)
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
  
  LAPACKE<T>::laswp_work(LAPACK_COL_MAJOR, n, A, n, 1, n, piv, 1);
  
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
    size_t n = 32;
    if (argc > 1)
      n = atoll(argv[1]);
    
    // block count
    size_t block_size = 2;
    if (argc > 2)
      block_size = atoll(argv[2]);
    
    // Number of iterations
    int niter = 1;
    if (argc >3)
      niter = atoi(argv[3]);
    
    // Make verification ?
    int verif = 0;
    if (argc >4)
      verif = atoi(argv[4]);
    
    global_blocsize = block_size;
    size_t nblock = (n%block_size==0) ? (n/block_size) : ((n/block_size)+1);
    size_t na = nblock * block_size;
    const int ib = CONFIG_IB_CPU; // from PLASMA
    
    double t0, t1;
    
    double_type* dL = (double_type*) calloc( na*na, sizeof(double_type) );
    ka::array<2,double_type> L(dL, n, n, n);
    
    int* ipiv = (int*) calloc(na*na, sizeof(int));
    if (0 == ipiv) {
      std::cout << "Fatal Error. Cannot allocate matrices A, "
	    << std::endl;
      abort();
    }
    
    ka::range1d<int> piv(ipiv, n*n);
    
    double_type* WORK = (double_type*) calloc(
                                               ib*block_size, sizeof(double_type) );
    
    double_type* dAcopy;
    double_type* dA = (double_type*) calloc(na * na, sizeof(double_type));
    if (0 == dA) {
      std::cout << "Fatal Error. Cannot allocate matrices A, "
      << std::endl;
      return;
    }
    ka::array<2,double_type> A(dA, n, n, n);
#if defined(CONFIG_USE_PLASMA)
    for(size_t m= 0; m < n; m += block_size){
      for(size_t k= 0; k < n; k += block_size){
        PLASMA<double_type>::plrnt(block_size, block_size, dA+k*n+m, block_size, n, m, k, 3453);
      }
    }
#else
    TaskBodyCPU<TaskLARNV<double_type> >()( ka::range2d_w<double_type>(A) );
#endif
#if CONFIG_USE_CUDA
    ka::Memory::Register( A );
    ka::Memory::Register( L );
    ka::Memory::Register( piv );
#endif
    
    if (verif) {
      /* copy the matrix to compute the norm */
      dAcopy = (double_type*) calloc(na * na, sizeof(double_type));
      memcpy(dAcopy, dA, na*na*sizeof(double_type) );
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
      ka::Spawn<TaskLU<double_type> >(ka::SetStaticSched())( A, L, piv, (uintptr_t)WORK );
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
#if 0
      fprintf(stdout, "DGETRF,xkaapi,%d,%d,%d,%d,%.10f,%.10f\n",
              atoi(getenv("KAAPI_NCPU")),
              atoi(getenv("KAAPI_NGPU")),
              (int)n,
              (int)global_blocsize,
              t1-t0, gflops );
#endif
      fflush(stdout);
    }
    
    if (verif) {
      // /* compute the norm || A - L*U ||inf */
      check<double_type>(n, dAcopy, dA, n, ipiv);
      free(dAcopy);
    }
    
#if CONFIG_USE_CUDA
    ka::Memory::Unregister( A );
    ka::Memory::Unregister( L );
    ka::Memory::Unregister( piv );
#endif
    free(dA);
    free(dL);
    free(ipiv);
    free(WORK);
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
