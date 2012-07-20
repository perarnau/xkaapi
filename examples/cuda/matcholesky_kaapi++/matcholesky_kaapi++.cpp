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
#include "../matrix/matrix.h"

#include "kaapi++" // this is the new C++ interface for Kaapi

#include "lapacke.h"

/* Generate a random matrix symetric definite positive matrix of size m x m 
 - it will be also interesting to generate symetric diagonally dominant 
 matrices which are known to be definite postive.
 Based from MAGMA
 */
static void
generate_matrix( double_type* A, size_t N )
{
  for (size_t i = 0; i< N; i++) {
    A[i*N+i] = A[i*N+i] + 1.*N; 
    for (size_t j = 0; j < i; j++)
      A[i*N+j] = A[j*N+i];
  }
}

/*------------------------------------------------------------------------
 *  Check the factorization of the matrix A2
 *  from PLASMA (examples/example_dpotrf.c)
 */
int check_factorization(int N, double *A1, double *A2, int LDA, int uplo)
{
  double Anorm, Rnorm;
  double alpha;
  int info_factorization;
  int i,j;
  double eps;
  
  eps = LAPACKE_dlamch_work('e');
  
  double *Residual = (double *)malloc(N*N*sizeof(double));
  double *L1       = (double *)malloc(N*N*sizeof(double));
  double *L2       = (double *)malloc(N*N*sizeof(double));
  double *work              = (double *)malloc(N*sizeof(double));
  
  memset((void*)L1, 0, N*N*sizeof(double));
  memset((void*)L2, 0, N*N*sizeof(double));
  
  alpha= 1.0;
  
  LAPACKE_dlacpy_work(LAPACK_COL_MAJOR,' ', N, N, A1, LDA, Residual, N);
  
  /* Dealing with L'L or U'U  */
  if (uplo == CblasUpper){
    LAPACKE_dlacpy_work(LAPACK_COL_MAJOR,'u', N, N, A2, LDA, L1, N);
    LAPACKE_dlacpy_work(LAPACK_COL_MAJOR,'u', N, N, A2, LDA, L2, N);
    cblas_dtrmm(CblasColMajor, CblasLeft, CblasUpper, CblasTrans, CblasNonUnit, N, N, (alpha), L1, N, L2, N);
  }
  else{
    LAPACKE_dlacpy_work(LAPACK_COL_MAJOR,'l', N, N, A2, LDA, L1, N);
    LAPACKE_dlacpy_work(LAPACK_COL_MAJOR,'l', N, N, A2, LDA, L2, N);
    cblas_dtrmm(CblasColMajor, CblasRight, CblasLower, CblasTrans, CblasNonUnit, N, N, (alpha), L1, N, L2, N);
  }
  
  /* Compute the Residual || A -L'L|| */
  for (i = 0; i < N; i++)
    for (j = 0; j < N; j++)
      Residual[j*N+i] = L2[j*N+i] - Residual[j*N+i];
  
  Rnorm = LAPACKE_dlange_work(LAPACK_COL_MAJOR, 'I', N, N, Residual, N, work);
  Anorm = LAPACKE_dlange_work(LAPACK_COL_MAJOR, 'I', N, N, A1, LDA, work);
  
  printf("# ============\n");
  printf("# Checking the Cholesky Factorization \n");
  printf("# -- ||L'L-A||_oo/(||A||_oo.N.eps) = %e \n",Rnorm/(Anorm*N*eps));
  
  if ( isnan(Rnorm/(Anorm*N*eps)) || (Rnorm/(Anorm*N*eps) > 10.0) ){
    printf("# ERROR -- Factorization is suspicious ! \n");
    info_factorization = 1;
  }
  else{
    printf("# OK -- Factorization is CORRECT ! \n");
    info_factorization = 0;
  }
  
  free(Residual); free(L1); free(L2); free(work);
  
  return info_factorization;
}

/* Block Cholesky factorization A <- L * L^t
 Lower triangular matrix, with the diagonal, stores the Cholesky factor.
 Based on PLASMA library.
 */
/* TODO: inverted indexes as XKaapi does not have ColMajor matrix */
struct TaskCholesky: public ka::Task<2>::Signature<
ka::RPWP<ka::range2d<double_type> >, /* A */
CBLAS_UPLO
>{};
static size_t global_blocsize = 2;
template<>
struct TaskBodyCPU<TaskCholesky> {
  void operator()(
      const ka::StaticSchedInfo* info, 
      ka::range2d_rpwp<double_type> A,
      const enum CBLAS_UPLO uplo
  )
  {
    size_t N = A->dim(0);
    size_t blocsize = global_blocsize;
    
    /* Distribute matrix A to bloc cyclic 1D (grid = ncpu x 1) over GPU set */
    ka::Distribute( A, ka::BlockCyclic2D(info->count_gpu(), blocsize, 1, blocsize) );
    
    if( uplo == CblasLower ) {
      for (size_t k=0; k < N; k += blocsize) {
        ka::rangeindex rk(k, k+blocsize);
        ka::Spawn<TaskDPOTRF>()
	      ( CblasColMajor, CblasLower, A(rk,rk) );
        
        for (size_t m=k+blocsize; m < N; m += blocsize) {
          ka::rangeindex rm(m, m+blocsize);
          ka::Spawn<TaskDTRSM>()
          ( CblasColMajor, CblasRight, uplo,
           CblasTrans, CblasNonUnit, 1.0, A(rk,rk), A(rk,rm));
        }
        
        for (size_t m=k+blocsize; m < N; m += blocsize) {
          ka::rangeindex rm(m, m+blocsize);
          ka::Spawn<TaskDSYRK>()
          ( CblasColMajor, uplo,
           CblasNoTrans, -1.0, A(rk,rm), 1.0, A(rm,rm));
          
          for (size_t n=k+blocsize; n < m; n += blocsize) {
            ka::rangeindex rn(n, n+blocsize);
            ka::Spawn<TaskDGEMM>()
            (   CblasColMajor, CblasNoTrans, CblasTrans,
             -1.0, A(rk,rm), A(rk,rn), 1.0, A(rn,rm));
          }
        }
      }
    } else if( uplo == CblasUpper ) {
      for (size_t k=0; k < N; k += blocsize) {
        ka::rangeindex rk(k, k+blocsize);
        ka::Spawn<TaskDPOTRF>()
	      ( CblasColMajor, uplo, A(rk,rk) );
        
        for (size_t m=k+blocsize; m < N; m += blocsize) {
          ka::rangeindex rm(m, m+blocsize);
          ka::Spawn<TaskDTRSM>()
          ( CblasColMajor, CblasLeft, uplo,
           CblasTrans, CblasNonUnit, 1.0, A(rk,rk), A(rm,rk) );
        }
        
        for (size_t m=k+blocsize; m < N; m += blocsize) {
          ka::rangeindex rm(m, m+blocsize);
          ka::Spawn<TaskDSYRK>()
          ( CblasColMajor, uplo,
           CblasTrans, -1.0, A(rm,rk), 1.0, A(rm,rm) );
          
          for (size_t n=k+blocsize; n < m; n += blocsize) {
            ka::rangeindex rn(n, n+blocsize);
            ka::Spawn<TaskDGEMM>()
            ( CblasColMajor, CblasTrans, CblasNoTrans,
             -1.0, A(rn,rk),
             A(rm,rk),
             1.0, A(rm,rn));
          }
        }
      }
    }
  }
};

/* Main of the program
 */
struct doit {
  void doone_exp( int n, int block_size, int niter, int verif )
  {
    global_blocsize = block_size;
    double t0, t1;
    double_type* dA = (double_type*) calloc(n* n, sizeof(double_type));
    if (0 == dA) {
      std::cout << "Fatal Error. Cannot allocate matrice A "
      << std::endl;
      return;
    }
    
    double_type* dAcopy = 0;
    if (verif) {
      dAcopy = (double_type*) calloc(n* n, sizeof(double_type));
      if (dAcopy ==0)
      {
        std::cout << "Fatal Error. Cannot allocate matrice Acopy "
        << std::endl;
        return;
      }
    }
    
    ka::array<2,double_type> A(dA, n, n, n);
#if CONFIG_USE_CUDA
    ka::Memory::Register( A );
#endif
    
#if 0
    std::cout << "Start Cholesky with " 
    << block_count << 'x' << block_count 
    << " blocs of matrix of size " << n << 'x' << n 
    << std::endl;
#endif
    
    // Cholesky factorization of A 
    double sumt = 0.0;
    double sumgf = 0.0;
    double sumgf2 = 0.0;
    double gflops;
    double gflops_max = 0.0;
    
    /* formula used by PLASMA in time_dpotrf.c */
    double fp_per_mul = 1;
    double fp_per_add = 1;
#define FMULS_POTRF(n) ((n) * (((1. / 6.) * (n) + 0.5) * (n) + (1. / 3.)))
#define FADDS_POTRF(n) ((n) * (((1. / 6.) * (n)      ) * (n) - (1. / 6.)))
    double fmuls = FMULS_POTRF(n);
    double fadds = FADDS_POTRF(n);
    
    for (int i=0; i<niter; ++i) {
      /* based on MAGMA */
      TaskBodyCPU<TaskDLARNV>()( ka::range2d_w<double_type>(A) );
      generate_matrix(dA, n); 
      if (verif)
        memcpy(dAcopy, dA, n*n*sizeof(double_type) );
      
      t0 = kaapi_get_elapsedtime();
      ka::Spawn<TaskCholesky>(ka::SetStaticSched())( A, CblasLower );
      ka::Sync();
#if CONFIG_USE_CUDA
      ka::MemorySync();
#endif
      t1 = kaapi_get_elapsedtime();
      
      gflops = 1e-9 * (fmuls * fp_per_mul + fadds * fp_per_add) / (t1-t0);
      if (gflops > gflops_max) gflops_max = gflops;
      
      sumt += double(t1-t0);
      sumgf += gflops;
      sumgf2 += gflops*gflops;
      
      if (verif) {
        check_factorization( n, dAcopy, dA, n, CblasLower );
        free( dAcopy );
      }
    }
    
    gflops = sumgf/niter;
    printf("POTRF %6d %5d %5d %9.10f %9.6f\n",
           (int)n,
           (int)global_blocsize,
           (int)kaapi_getconcurrency(),
           sumt/niter, gflops );
    
#if CONFIG_USE_CUDA
    ka::Memory::Unregister( A );
#endif
    free(dA);
  }
  
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
    
    printf("# size  blocksize  #threads   time      GFlop/s\n");
    for (int k=0; k<1; ++k, ++n )
    {
      doone_exp( n, block_size, niter, verif );
    }
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
