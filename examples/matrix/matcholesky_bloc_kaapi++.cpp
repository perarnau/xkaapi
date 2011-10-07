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
#include <iostream>
#include <iomanip>
#include <string>
#include <string.h>
#include <math.h>
#include <sys/types.h>

extern "C" {
#include <cblas.h>   // 
#include <clapack.h> // assume MKL/ATLAS clapack version
}


#include "kaapi++" // this is the new C++ interface for Kaapi


/* Compute inplace LLt factorization of A, ie L such that A = L * Lt
   with L lower triangular.
*/
struct TaskDPOTRF: public ka::Task<3>::Signature<
  CBLAS_UPLO,                  /* upper / lower */
  ka::RW<double>,   /* A */
  int /* NB */
>{};
template<>
struct TaskBodyCPU<TaskDPOTRF> {
  void operator()( 
    CBLAS_UPLO uplo, ka::pointer_rw<double> A, int NB
  )
  {
    const int n     = NB;
    const int lda   = NB;
    double* const a = A;
    clapack_dpotrf(
      CblasRowMajor, uplo, n, a, lda
    );
  }
};


/* DTRSM
*/
struct TaskDTRSM: public ka::Task<8>::Signature<
      CBLAS_SIDE,                  /* side */
      CBLAS_UPLO,                  /* uplo */
      CBLAS_TRANSPOSE,             /* transA */
      CBLAS_DIAG,                  /* diag */
      double,                      /* alpha */
      ka::R<double>, /* A */
      ka::RW<double>, /* B */
      int /* NB */
>{};
template<>
struct TaskBodyCPU<TaskDTRSM> {
  void operator()( 
    CBLAS_SIDE             side,
    CBLAS_UPLO             uplo,
    CBLAS_TRANSPOSE        transA,
    CBLAS_DIAG             diag,
    double                 alpha,
    ka::pointer_r<double> A, 
    ka::pointer_rw<double> C,
    int NB
  )
  {
    const double* const a = A;
    const int lda = NB;

    double* const c = C;
    const int ldc = NB;

    const int n = NB;
    const int k = NB;

    cblas_dtrsm
    (
      CblasRowMajor, side, uplo, transA, diag,
      n, k, alpha, a, lda, c, ldc
    );
  }
};


/* Rank k update
*/
struct TaskDSYRK: public ka::Task<7>::Signature<
  CBLAS_UPLO,                  /* CBLAS Upper / Lower */
  CBLAS_TRANSPOSE,             /* transpose flag */
  double,                      /* alpha */
  ka::R<double>, /* A */
  double,                      /* beta */
  ka::RW<double>, /* C */
  int /* NB */
>{};
template<>
struct TaskBodyCPU<TaskDSYRK> {
  void operator()(
    CBLAS_UPLO uplo,
    CBLAS_TRANSPOSE trans,
    double alpha,
    ka::pointer_r <double>  A, 
    double beta,
    ka::pointer_rw<double> C,
    int NB
  )
  {
    const int n     = NB;
    const int k     = NB;
    const int lda   = NB;
    const double* const a = A;

    const int ldc   = NB;
    double* const c = C;

    cblas_dsyrk
    (
      CblasRowMajor, uplo, trans,
      n, k, alpha, a, lda, beta, c, ldc
    );

  }
};


/* DGEMM rountine to compute
    Aij <- alpha* Aik * Akj + beta Aij
*/
struct TaskDGEMM: public ka::Task<8>::Signature<
      CBLAS_TRANSPOSE,             /* NoTrans/Trans for A */
      CBLAS_TRANSPOSE,             /* NoTrans/Trans for B */
      double,                      /* alpha */
      ka::R<double>, /* Aik   */
      ka::R<double>, /* Akj   */
      double,                      /* beta */
      ka::RW<double>, /* Aij   */
      int /* NB */
>{};
template<>
struct TaskBodyCPU<TaskDGEMM> {
  void operator()
  (
    CBLAS_TRANSPOSE transA,
    CBLAS_TRANSPOSE transB,
    double alpha,
    ka::pointer_r<double> Aik,
    ka::pointer_r<double> Akj,
    double beta,
    ka::pointer_rw<double> Aij,
    int NB
  )
  {
    const double* const a = Aik;
    const double* const b = Akj;
    double* const c       = Aij;

    const int m = NB;
    const int n = NB;
    const int k = NB;

    const int lda = NB;
    const int ldb = NB;
    const int ldc = NB;

    cblas_dgemm
    (
      CblasRowMajor, transA, transB,
      m, n, k, alpha, a, lda, b, ldb, beta, c, ldc
    );
  }
};


/* Generate a random matrix symetric definite positive matrix of size m x m 
   - it will be also interesting to generate symetric diagonally dominant 
   matrices which are known to be definite postive.
*/
static void generate_matrix(double* A, size_t m)
{
  // 
  for (size_t i = 0; i< m; ++i)
  {
    for (size_t j = 0; j< m; ++j)
      A[i*m+j] = 1.0 / (1.0+i+j);
    A[i*m+i] = m*1.0; 
  }
}


/*
*/
#define DIM 64
static void convert_to_blocks(long NB, long N, double *Alin, double* A[DIM][DIM] )
{
  for (long i = 0; i < N; i++)
  {
    for (long j = 0; j < N; j++)
    {
      A[j/NB][i/NB][(i%NB)*NB+j%NB] = Alin[i*N+j];
    }
  }

}

/* Block Cholesky factorization A <- L * L^t
   Lower triangular matrix, with the diagonal, stores the Cholesky factor.
*/
struct TaskCholesky: public ka::Task<3>::Signature<
      int,       /* N */
      int,       /* NB */
      uintptr_t  /* A */
>{};
template<>
struct TaskBodyCPU<TaskCholesky> {
  void operator()( int N, int NB, uintptr_t ptr )
  {
    typedef double* TYPE[DIM][DIM]; 
    TYPE* A = (TYPE*)ptr;
    for (int k=0; k < N; ++k)
    {
      ka::Spawn<TaskDPOTRF>()( CblasLower, (*A)[k][k], NB );

      for (int m=k+1; m < N; ++m)
      {
        ka::Spawn<TaskDTRSM>()(CblasRight, CblasLower, CblasTrans, CblasNonUnit, 1.0, (*A)[k][k], (*A)[m][k], NB);
      }

      for (int m=k+1; m < N; ++m)
      {
        ka::Spawn<TaskDSYRK>()( CblasLower, CblasNoTrans, -1.0, (*A)[m][k], 1.0, (*A)[m][m], NB);
        for (int n=k+1; n < m; ++n)
        {
          ka::Spawn<TaskDGEMM>()(CblasNoTrans, CblasTrans, -1.0, (*A)[m][k], (*A)[n][k], 1.0, (*A)[m][n], NB);
        }
      }
    }
  }
};



/* Main of the program
*/
struct doit {
  void doone_exp( int n, int block_count, int niter, int verif )
  {
    int NB = n / block_count;
    
    //n = block_count * global_blocsize;

    double t0, t1;
    double* dA = (double*) calloc(n* n, sizeof(double));
    if (0 == dA)
    {
      std::cout << "Fatal Error. Cannot allocate matrice A "
                << std::endl;
      return;
    }

    // blocked matrix
    double* dAbloc[DIM][DIM];
    for (long i = 0; i < DIM; i++)
      for (long j = 0; j < DIM; j++)
      {
       dAbloc[i][j] = 0;
#if 1
       posix_memalign( (void**)&(dAbloc[i][j]), 128, NB*NB*sizeof(double));
#else
       posix_memalign( (void**)&(dAbloc[i][j]), 4096, NB*NB*sizeof(double));
       if (kaapi_numa_bind( dAbloc[i][j], NB*NB*sizeof(double), (j < i ? i-j : j-i)%8 ) != 0)
       {
         std::cout << "Error cannot bind addr: " << dAbloc[i][j] << " on node " << (i+j)%8 << std::endl;
         return;
       }
#endif
      }


              
    // Cholesky factorization of A 
    double sumt = 0.0;
    double sumgf = 0.0;
    double sumgf2 = 0.0;
    double sd;
    double sumt_exec = 0.0;
    double sumgf_exec = 0.0;
    double sumgf2_exec = 0.0;
    double sd_exec;
    double gflops;
    double gflops_max = 0.0;
    double gflops_exec;
    double gflops_exec_max = 0.0;

    /* formula used by plasma in time_dpotrf.c */
    double fp_per_mul = 1;
    double fp_per_add = 1;
    double fmuls = (n * (1.0 / 6.0 * n + 0.5 ) * n);
    double fadds = (n * (1.0 / 6.0 * n ) * n);
        
    printf("%6d %5d %5d ", (int)n, (int)kaapi_getconcurrency(), (int)(n/NB) );

    generate_matrix( dA, n );
    convert_to_blocks(NB, n, dA, dAbloc);
    for (int i=0; i<niter; ++i)
    {
#if 0
{ // display the memory mapping
      std::cout << "Memory mapping for A:" << std::endl;
      for (int i=0; i<DIM; ++i)
      {
        for (int j=0; j<DIM; ++j)
        {
           double* addr = dAbloc[i][j];
           int node = kaapi_numa_get_page_node( addr );
           //std::cout << " @:" << addr  << " " << node << ",  ";
           std::cout << node << ",  ";
        }
        std::cout << std::endl;
      }
}
#endif

      t0 = kaapi_get_elapsedtime();
      ka::Spawn<TaskCholesky>( ka::SetStaticSched() )(block_count, NB, (uintptr_t)&dAbloc);
      ka::Sync();
      t1 = kaapi_get_elapsedtime();

      gflops = 1e-9 * (fmuls * fp_per_mul + fadds * fp_per_add) / (t1-t0);
      if (gflops > gflops_max) gflops_max = gflops;
//printf("Gflops= %9.3f\n", gflops );

      sumt += double(t1-t0);
      sumgf += gflops;
      sumgf2 += gflops*gflops;

      gflops_exec = 1e-9 * (fmuls * fp_per_mul + fadds * fp_per_add) / 1.0;
      if (gflops_exec > gflops_exec_max) gflops_exec_max = gflops_exec;
      
      sumt_exec += double(1.0);
      sumgf_exec += gflops_exec;
      sumgf2_exec += gflops_exec*gflops_exec;
    }

    gflops = sumgf/niter;
    gflops_exec = sumgf_exec/niter;
    if (niter ==1) 
    {
      sd = 0.0;
      sd_exec = 0.0;
    } else {
      sd = sqrt((sumgf2 - (sumgf*sumgf)/niter)/niter);
      sd_exec = sqrt((sumgf2_exec - (sumgf_exec*sumgf_exec)/niter)/niter);
    }
    
    printf("%9.3f %9.3f %9.3f %9.3f %9.3f %9.3f %9.3f %9.3f\n", 
        sumt/niter, 
        gflops, 
        sd, 
        sumt_exec/niter, 
        gflops_exec, 
        sd_exec, 
        gflops_max, 
        gflops_exec_max );

    free(dA);
  }

  void operator()(int argc, char** argv )
  {
    // matrix dimension
    int n = 32;
    if (argc > 1)
      n = atoi(argv[1]);

    // block count
    int block_count = 2;
    if (argc > 2)
      block_count = atoi(argv[2]);
      
    // Number of iterations
    int niter = 1;
    if (argc >3)
      niter = atoi(argv[3]);

    // Make verification ?
    int verif = 0;
    if (argc >4)
      verif = atoi(argv[4]);
    
    //int nmax = n+16*incr;
    printf("size   #threads #bs    time      GFlop/s   Deviation\n");
    //for (int k=9; k<15; ++k, ++n )
    {
      //doone_exp( 1<<k, block_count, niter, verif );
      doone_exp( n, block_count, niter, verif );
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
