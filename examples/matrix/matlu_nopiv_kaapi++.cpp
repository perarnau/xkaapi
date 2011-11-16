/*
** kaapi_impl.h
** xkaapi
** 
**
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


/* Generate a random matrix of size m x m */
static void generate_matrix(double* c, size_t m)
{
  srand48(0);
  // generate a random matrix
  const size_t mm = m * m;

  // generate a down triangular matrix
  for (size_t i = 0; i < mm; ++i) c[i] = drand48()*1000.0;
}



/* Task Print Matrix LU
 * assume that the matrix stores an LU decomposition with L lower triangular matrix with unit diagonal
   and U upper triangular matrix, then print both L and U using the Maple matrix format.
 */
struct TaskPrintMatrixLU : public ka::Task<2>::Signature<std::string,  ka::R<ka::range2d<double> > > {};
template<>
struct TaskBodyCPU<TaskPrintMatrixLU> {
  void operator() ( std::string msg, ka::range2d_r<double> A  )
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



/* Compute inplace LU factorization of A, ie L and U such that A = L * U
   with L lower triangular, with unit diagonal and U upper triangular.
*/
struct TaskDGETRFNoPiv: public ka::Task<1>::Signature<
  ka::RW<ka::range2d<double> > /* A */
>{};
template<>
struct TaskBodyCPU<TaskDGETRFNoPiv> {
  void operator()
  ( ka::range2d_rw<double> A )
  {
    int N = A.dim(0);
    int M = A.dim(1);
    int k_stop = std::min(N, M);

#if 0
    /* Simple code: correct ||L*U -A || very small */
    for ( int k = 0; k < k_stop; ++k )
    {
      double pivot = 1.0 / A(k,k);
      for (int i = k+1 ; i < N ; ++i )
        A(i,k) *= pivot;
      for (int i= k+1 ; i < N; ++i)
        for (int j= k+1 ; j < M; ++j)
          A(i,j) -=  A(i,k)*A(k,j);
    }
#else
    /* More optimized code: correct ||L*U -A || very small */
    double* dk = A.ptr(); /* &A(k,0) */
    int lda = A.lda();
    
    for ( int k = 0; k < k_stop; ++k, dk += lda )
    {
      double pivot = 1.0 / dk[k];

      /* column normalization */
      double* dik = dk+k+lda; /* A(i,k) i=k+1 */
      for (int i = k+1 ; i < N ; ++i, dik += lda )
        *dik *= pivot;

      /* submatrix update */
      dik = dk+k+lda; /* A(i,k) i=k+1 */
      for (int i= k+1 ; i < N; ++i, dik += lda)
      {
        double* dij = dik+1;
        double* dkj = dk+k+1;
        for (int j= k+1 ; j < M; ++j, ++dij,++dkj)
          *dij -=  *dik * *dkj;
      } 
    }
#endif
  }
};


/* Compute the norm || A - L*U ||infinity
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
      *Lik = 1.0; /* diag entry */
      ++Lik; 
      ++j;
      for ( ; j<N; ++j, ++Lik)
        *Lik = 0.0;
      
      /* copy U */
      const double* AUik = dLU+i*lda_LU + i;
      double* Uik = U+i*N;
      for (j=0; j < i; ++j, ++Uik)
        *Uik = 0.0;
      for ( ; j < N; ++j, ++Uik, ++AUik)
        *Uik = *AUik;
    }

    TaskBodyCPU<TaskDGEMM>()(
        CblasNoTrans,CblasNoTrans,
        1.0,
        ka::range2d_r<double>(ka::range2d<double>(L, M, N, N)),
        ka::range2d_r<double>(ka::range2d<double>(U, M, N, N)),
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


/* Block LU factorization
*/
struct TaskLU: public ka::Task<1>::Signature<
      ka::RPWP<ka::range2d<double> > /* A */
>{};
static size_t global_blocsize = 2;
template<>
struct TaskBodyCPU<TaskLU> {
  void operator()( const ka::StaticSchedInfo* info, ka::range2d_rpwp<double> A )
  {
    //int ncpu = info->count_cpu();
    //int sncpu = (int)sqrt( (double)ncpu );
    size_t N = A.dim(0);
    size_t blocsize = global_blocsize;

#if 0
    std::cout << kaapi_get_self_kid() << "::" << __PRETTY_FUNCTION__ 
              << "-> blocsize " << blocsize
              << std::endl;
#endif

    for (size_t k=0; k<N; k += blocsize)
    {
      ka::rangeindex rk(k, k+blocsize);
      // A(rk,rk) = L(rk,rk) * U(rk,rk) <- LU( A(rk,rk) 
      ka::Spawn<TaskDGETRFNoPiv>()( A(rk,rk) );

      for (size_t j=k+blocsize; j<N; j += blocsize)
      {
        ka::rangeindex rj(j, j+blocsize);
        // A(rk,rj) <- L(rk,rk)^-1 * A(rk,rj) 
        ka::Spawn<TaskDTRSM>()(CblasLeft, CblasLower, CblasNoTrans, CblasUnit, 1.0, A(rk,rk), A(rk,rj));
      }
      for (size_t i=k+blocsize; i<N; i += blocsize)
      {
        ka::rangeindex ri(i, i+blocsize);
        // A(ri,rk) <- A(ri,rk) * U(rk,rk)^-1
        ka::Spawn<TaskDTRSM>()(CblasRight, CblasUpper, CblasNoTrans, CblasNonUnit, 1.0, A(rk,rk), A(ri,rk));
      }

      for (size_t i=k+blocsize; i<N; i += blocsize)
      {
        ka::rangeindex ri(i, i+blocsize);
        for (size_t j=k+blocsize; j<N; j += blocsize)
        {
          ka::rangeindex rj(j, j+blocsize);
          // A(ri,rj) <- A(ri,rj) - A(ri,rk)*A(rk,rj)
          ka::Spawn<TaskDGEMM>()(CblasNoTrans, CblasNoTrans, -1.0, A(ri,rk), A(rk,rj), 1.0, A(ri,rj));
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

    global_blocsize = n / block_count;
    n = block_count * global_blocsize;

    double t0, t1;
    double* dA = (double*) calloc(n* n, sizeof(double));
    double* dAcopy = (double*) calloc(n* n, sizeof(double));
    if ((0 == dA) || (dAcopy ==0))
    {
      std::cout << "Fatal Error. Cannot allocate matrices A, "
                << std::endl;
      return;
    }

    generate_matrix(dA, n);
    /* copy the matrix to compute the norm */
    memcpy(dAcopy, dA, n*n*sizeof(double) );

    ka::array<2,double> A(dA, n, n, n);

    if (n <= 32) 
    {
      /* output respect the Maple format */
      ka::Spawn<TaskPrintMatrix<double> >()("A", A);
      ka::Sync();
    }

    std::cout << "Start LU with " 
              << block_count << 'x' << block_count 
              << " blocs of matrix of size " << n << 'x' << n 
              << std::endl;
              
    // LU factorization of A using ka::
    double ggflops = 0;
    double gtime = 0;
    for (int i=0; i<niter; ++i)
    {
      generate_matrix(dA, n);

      t0 = kaapi_get_elapsedtime();
      ka::Spawn<TaskLU>(ka::SetStaticSched())(A);
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
      std::cout << " LU took " << t1-t0 << " seconds   " <<  gflops << " GFlops" << std::endl;
    }
    std::cout << "*** Average LU took " << gtime/niter << " seconds   " <<  ggflops/niter << " GFlops" << std::endl;

    if (verif)
    {
      /* If n is small, print the results */
      if (n <= 32) 
      {
        ka::Spawn<TaskPrintMatrixLU>()( std::string(""), A );
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

    free(dA);
    free(dAcopy);
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
