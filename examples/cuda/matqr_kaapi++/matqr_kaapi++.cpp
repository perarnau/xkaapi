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

template<typename T>
struct TaskQR: public ka::Task<4>::Signature<
ka::RPWP<ka::range2d<T> >,  /* A */
ka::RPWP<ka::range2d<T> >,  /* T */
ka::RPWP<ka::range1d<T> >,  /* T */
ka::RPWP<ka::range1d<T> >  /* WORK */
>{};

static size_t global_blocsize = 2;

template<typename T>
struct TaskBodyCPU<TaskQR<T> > {
  void operator()(
                  ka::range2d_rpwp<T> A,
                  ka::range2d_rpwp<T> T,
                  ka::range1d_rpwp<T> TAU,
                  ka::range1d_rpwp<T> WORK
                  )
  {
    size_t M = A->dim(0);
    size_t N = A->dim(1);
    size_t K = std::min(M, N);
    size_t blocsize = global_blocsize;
    
    for (size_t k=0; k < K; k += blocsize) {
      ka::rangeindex rk(k, k+blocsize);
      ka::Spawn<TaskGEQRT<T> >()( CblasColMajor,
                                 A(rk,rk),
                                 T(rk,rk),
                                 TAU, WORK
                                 );
      
      for (size_t j=k+blocsize; j < N; j += blocsize) {
        ka::rangeindex rj(j, j+blocsize);
        ka::Spawn<TaskORMQR<T> >()(
                                   CblasColMajor, CblasLeft, CblasTrans,
                                   A(rk, rk),
                                   T(rk,rk),
                                   A(rk, rj),
                                   WORK
                                   );
      }
      
      for (size_t i=k+blocsize; i < M; i += blocsize) {
        ka::rangeindex ri(i, i+blocsize);
        ka::Spawn<TaskTSQRT<T> >()(
                                   CblasColMajor,
                                   A(rk,rk),
                                   A(ri,rk),
                                   _T(ri, rk),
                                   TAU,
                                   WORK
                                   );
        
        for (size_t j=k+blocsize; j<N; j += blocsize) {
          ka::rangeindex rj(j, j+blocsize);
          ka::Spawn<TaskTSMQR<T> >()(
                                     CblasColMajor, CblasLeft, CblasTrans,
                                     A(rk, rj),
                                     A(ri, rj),
                                     A(ri, rk),
                                     _T(ri, rk),
                                     WORK
                                     );
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
    const int ib = 48;
    
    double t0, t1;
    double_type* dA = (double_type*) calloc(n* n, sizeof(double_type));
    double_type* dAcopy;
    if (0 == dA) {
      std::cout << "Fatal Error. Cannot allocate matrices A, "
      << std::endl;
      return;
    }
    
    ka::array<2,double_type> A(dA, n, n, n);
    //TaskBodyCPU<TaskDLARNV>()( ka::range2d_w<double_type>(A) );
    TaskBodyCPU<TaskLARNV<double_type> >()( ka::range2d_w<double_type>(A) );
    //    ka::Memory::Register( A );
    
    double_type* dT = (double_type*) malloc(n * n * sizeof(double_type));
    ka::array<2,double_type> T( dT, n, n, n);
    
    double_type* dTAU = (double_type*) malloc(n * sizeof(double_type));
    ka::array<1,double_type> TAU( dTAU, n);
    
    double_type* dWORK = (double_type*) malloc( ib * n * block_size * sizeof(double_type));
    ka::array<1,double_type> WORK( dWORK, ib * n);
    
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
    
    double ggflops = 0;
    double gtime = 0;
    fprintf( stdout, "# size blocksize #threads time Gflops\n" );
    for (int i=0; i<niter; ++i)
    {
      t0 = kaapi_get_elapsedtime();
      ka::Spawn<TaskQR<double_type> >(ka::SetStaticSched())( A, T, TAU, WORK );
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
      fprintf( stdout, "GEQRF %6d %5d %5d %9.10f %9.6f\n",
              (int)n,
              (int)global_blocsize,
              (int)kaapi_getconcurrency(),
              t1-t0, gflops );
      fflush(stdout);
    }
    
#if 0
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
        TaskBodyCPU<TaskNormMatrix>()( CblasColMajor, CblasUpper,
                                      &norm,
                                      ka::range2d_rw<double_type>(ka::array<2,double_type>(dAcopy, n,
                                                                                           n, n)),
                                      ka::range2d_rw<double_type>(A) );
#endif
        ka::Sync();
        t1 = kaapi_get_elapsedtime();
        std::cout << "# Error ||A-LU||inf " << norm
        << ", in " << (t1-t0) << " seconds."
        << std::endl;
        
        t0 = kaapi_get_elapsedtime();
        TaskBodyCPU<TaskDGETRFNoPiv>() ( CblasColMajor,
                                        ka::range2d_rw<double_type>(A2) );
#if 1
        TaskBodyCPU<TaskNormMatrix>()( CblasColMajor, CblasUpper, 
                                      &norm,
                                      ka::range2d_rw<double_type>(ka::array<2,double_type>(dAcopy2, n,
                                                                                           n, n)),
                                      ka::range2d_rw<double_type>(A2) );
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
    ka::Memory::Unregister( A );
#endif
    
    free(dA);
    free(dT);
    free(dTAU);
    free(dWORK);
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
