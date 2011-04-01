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
#include "matrix.h"
#include "kaapi++" // this is the new C++ interface for Kaapi


#if 0
/* Generate a random matrix symetric definite positive matrix of size m x m 
   - it will be also interesting to generate symetric diagonally dominant 
   matrices which are known to be definite postive.
*/
static void generate_matrix(double* A, size_t m)
{
  // 
  for (size_t i = 0; i< m; ++i)
    for (size_t j = 0; j< m; ++j)
      A[i*m+j] = 1.0 / (1.0+i+j);
}
#else
/* Generate a random matrix symetric definite positive matrix of size m x m 
*/
static void generate_matrix(double* A, size_t N)
{
  double* L = new double[N*N];
  // Lower random part
  for (size_t i = 0; i< N; ++i)
  {  
    size_t j;
    for (j = 0; j< i+1; ++j)
      L[i*N+j] = drand48();
    for (; j< N; ++j)
      L[i*N+j] = 0.0;
    L[i*N+i] = N; /* add dominant diagonal, else very ill conditionning */
  }

  TaskBodyCPU<TaskDGEMM>()(
      CblasNoTrans,CblasTrans,
      1.0,
      ka::range2d_r<double>(ka::range2d<double>(L, N, N, N)),
      ka::range2d_r<double>(ka::range2d<double>(L, N, N, N)),
      0.0,
      ka::range2d_rw<double>(ka::range2d<double>(A, N, N, N))
  );
  
  delete []L;
}
#endif


/* Task Print Matrix LLt
 * assume that the matrix stores an LLt decomposition with L lower triangular matrix with unit diagonal
   and U upper triangular matrix, then print both L and Lt using the Maple matrix format.
 */
struct TaskPrintMatrixLLt : public ka::Task<2>::Signature<std::string,  ka::R<ka::range2d<double> > > {};
template<>
struct TaskBodyCPU<TaskPrintMatrixLLt> {
  void operator() ( std::string msg, ka::range2d_r<double> A  )
  {
    size_t d0 = A.dim(0);
    size_t d1 = A.dim(1);
    std::cout << "L :=matrix( [" << std::endl;
    for (size_t i=0; i < d0; ++i)
    {
      std::cout << "[";
      for (size_t j=0; j < i+1; ++j)
      {
        std::cout << std::setw(18) 
                  << std::setprecision(15) 
                  << std::scientific 
                  << A(i,j) << (j == d1-1 ? "" : ", ");
      }
      for (size_t j=i+1; j<d1; ++j)
      {
        std::cout << std::setw(18) 
                  << std::setprecision(15) 
                  << std::scientific << 0 << (j == d1-1 ? "" : ", ");
      }
      std::cout << "]" << (i == d0-1 ? ' ' : ',') << std::endl;
    }
    std::cout << "]);" << std::endl;
//    std::cout << "evalm( transpose(L) &* L    - A);" << std::endl;
    std::cout << "evalm( L &* transpose(L) - A);" << std::endl;
  }
};


/* Compute the norm || A - L*Lt ||infinity
 * The norm value serves to detec an error in the computation
 */
struct TaskNormMatrix: public ka::Task<3>::Signature<
  ka::W<double>, /* norm */
  ka::RW<ka::range2d<double> >, /* A */
  ka::RW<ka::range2d<double> >  /* LU */
>{};
template<>
struct TaskBodyCPU<TaskNormMatrix> {
  void operator() ( ka::pointer_w<double> norm, ka::range2d_rw<double> A, ka::range2d_rw<double> LU )
  {
    const double* dA = A.ptr();
    const double* dLU = LU.ptr();
    int lda_LU = LU.lda();
    int M = A.dim(0);
    int N = A.dim(1);
    *norm = 0.0;

    double max = 0.0;
    double* L = new double[M*N];
    double* U = new double[M*N];

    for (int i=0; i<M; ++i)
    {
      int j;
      /* copy L */
      const double* ALik = dLU+i*lda_LU;
      double* Lik = L+i*N;
      for (j=0; j < i; ++j, ++Lik, ++ALik)
        *Lik = *ALik;
      *Lik = *ALik; /* diag entry */
      ++Lik;
      ++ALik;
      ++j;
      for ( ; j<N; ++j, ++Lik)
        *Lik = 0.0;      
    }

    TaskBodyCPU<TaskDGEMM>()(
        CblasNoTrans,CblasTrans,
        1.0,
        ka::range2d_r<double>(ka::range2d<double>(L, M, N, N)),
        ka::range2d_r<double>(ka::range2d<double>(L, M, N, N)),
        -1.0,
        A
    );
    max = 0;
    for (int i=0; i<M*N; ++i)
    {
        double error_ij = fabs(dA[i]);
        if (error_ij > max) 
          max = error_ij;
    }
    delete [] U;
    delete [] L;
    *norm = max;
  }
};


/* Block Cholesky factorization A <- L * L^t
   Lower triangular matrix, with the diagonal, stores the Cholesky factor.
*/
struct TaskCholesky: public ka::Task<1>::Signature<
      ka::RPWP<ka::range2d<double> > /* A */
>{};
static size_t global_blocsize = 2;
template<>
struct TaskBodyCPU<TaskCholesky> {
  void operator()( const ka::StaticSchedInfo* info, ka::range2d_rpwp<double> A )
  {
    //int ncpu = info->count_cpu();
    //int sncpu = (int)sqrt( (double)ncpu );
    size_t N = A.dim(0);
    size_t blocsize = global_blocsize;

    for (size_t k=0; k < N; k += blocsize)
    {
      ka::rangeindex rk(k, k+blocsize);
      ka::Spawn<TaskDPOTRF>()( CblasLower, A(rk,rk) );

      for (size_t m=k+blocsize; m < N; m += blocsize)
      {
        ka::rangeindex rm(m, m+blocsize);
        ka::Spawn<TaskDTRSM>()(CblasRight, CblasLower, CblasTrans, CblasNonUnit, 1.0, A(rk,rk), A(rm,rk));
      }

      for (size_t m=k+blocsize; m < N; m += blocsize)
      {
        ka::rangeindex rm(m, m+blocsize);
        ka::Spawn<TaskDSYRK>()( CblasLower, CblasNoTrans, -1.0, A(rm,rk), 1.0, A(rm,rm));
        for (size_t n=k+blocsize; n < m; n += blocsize)
        {
          ka::rangeindex rn(n, n+blocsize);
          ka::Spawn<TaskDGEMM>()(CblasNoTrans, CblasTrans, -1.0, A(rm,rk), A(rn,rk), 1.0, A(rm,rn));
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
    global_blocsize = n / block_count;
    n = block_count * global_blocsize;

    double t0, t1;
    double* dA = (double*) calloc(n* n, sizeof(double));
    if (0 == dA)
    {
      std::cout << "Fatal Error. Cannot allocate matrice A "
                << std::endl;
      return;
    }

    double* dAcopy = 0;
    if (verif)
    {
      dAcopy = (double*) calloc(n* n, sizeof(double));
      if (dAcopy ==0)
      {
        std::cout << "Fatal Error. Cannot allocate matrice Acopy "
                  << std::endl;
        return;
      }
    }

    ka::array<2,double> A(dA, n, n, n);

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
    double sd;
    double gflops;

    /* formula used by plasma in time_dpotrf.c */
    double fp_per_mul = 1;
    double fp_per_add = 1;
    double fmuls = (n * (1.0 / 6.0 * n + 0.5 ) * n);
    double fadds = (n * (1.0 / 6.0 * n ) * n);
        
    for (int i=0; i<niter; ++i)
    {
      generate_matrix(dA, n);
      if (verif)
        memcpy(dAcopy, dA, n*n*sizeof(double) );

      t0 = kaapi_get_elapsedtime();
      ka::Spawn<TaskCholesky>(ka::SetStaticSched())(A);
      ka::Sync();
      t1 = kaapi_get_elapsedtime();

      gflops = 1e-9 * (fmuls * fp_per_mul + fadds * fp_per_add) / (t1-t0);
      
      sumt += double(t1-t0);
      sumgf += gflops;
      sumgf2 += gflops*gflops;

      if (verif)
      {
#if 0
        generate_matrix(dA, n);
        if (verif)
          memcpy(dAcopy, dA, n*n*sizeof(double) );

        ka::Spawn<TaskDPOTRF>()( CblasLower, A );
        ka::Sync();
#endif

        /* If n is small, print the results */
        if (n <= 32) 
        {
          /* output respect the Maple format */
          ka::Spawn<TaskPrintMatrix<double> >()("A", ka::range2d<double>(dAcopy, n, n, n));
          ka::Sync();

          ka::Spawn<TaskPrintMatrixLLt>()( std::string(""), A );
          ka::Sync();
        }
        // /* compute the norm || A - L*U ||inf */
        {
          double norm;
          t0 = kaapi_get_elapsedtime();
          ka::Spawn<TaskNormMatrix>()( &norm, ka::array<2,double>(dAcopy, n, n, n), A );
          ka::Sync();
          t1 = kaapi_get_elapsedtime();
          
          std::cout << "Error ||A-LU||inf :" << norm 
                    << ", in " << (t1-t0) << " seconds." 
                    << std::endl;
        }
      }
    }

    gflops = sumgf/niter;
    if (niter ==1) 
      sd = 0.0;
    else
      sd = sqrt((sumgf2 - (sumgf*sumgf)/niter)/niter);
    
    printf("%6d %5d %5d %9.3f %9.3f %9.3f\n", (int)n, (int)kaapi_getconcurrency(), (int)(n/global_blocsize), sumt/niter, gflops, sd );

    free(dA);
    if (verif)
      free(dAcopy);
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
    
    int incr = 512;
    int nmax = n+8*incr;
    printf("size   #threads #bs    time      GFlop/s   Deviation\n");
    for (int k=n; k<nmax; k += incr )
    {
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
