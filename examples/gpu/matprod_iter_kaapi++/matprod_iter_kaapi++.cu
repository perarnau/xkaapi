/*
 * matrix-mulitpy.cilk
 *
 * An implementation of matrix multiply based on Cilk parallelization (matrix_multiply.cilk) 
 * but using Kaapi C++ construction

 * First of five matrix multiply examples to compare dense matrix multiplication 
 * algorithms using Cilk parallelization.
 *   Example 1: Straightforward loop parallelization of matrix multiplication.
 *
 * Copyright (c) 2007-2008 Cilk Arts, Inc.  55 Cambridge Street,
 * Burlington, MA 01803.  Patents pending.  All rights reserved. You may
 * freely use the sample code to guide development of your own works,
 * provided that you reproduce this notice in any works you make that
 * use the sample code.  This sample code is provided "AS IS" without
 * warranty of any kind, either express or implied, including but not
 * limited to any implied warranty of non-infringement, merchantability
 * or fitness for a particular purpose.  In no event shall Cilk Arts,
 * Inc. be liable for any direct, indirect, special, or consequential
 * damages, or any other damages whatsoever, for any use of or reliance
 * on this sample code, including, without limitation, any lost
 * opportunity, lost profits, business interruption, loss of programs or
 * data, even if expressly advised of or otherwise aware of the
 * possibility of such damages, whether in an action of contract,
 * negligence, tort, or otherwise.
 *
 */
#include <iostream>
#include <iomanip>
#include <string>
#include <cuda.h>
#include "kaapi++" // this is the new C++ interface for Kaapi
#if 0
#include <cblas.h>
#endif

// missing definition
extern "C" int kaapi_memory_synchronize(void);

static int BLOCSIZE = 0;

// no double type on gtx280
typedef float double_type;

/* Task Print
 * this task prints the sum of the entries of an array 
 * each entries is view as a pointer object:
    array<1,R<int> > means that each entry may be read by the task
 */
struct TaskPrintMatrix : public ka::Task<2>::Signature<std::string,  ka::R<ka::range2d<double_type> > > {};

template<>
struct TaskBodyCPU<TaskPrintMatrix> {
  void operator() ( std::string msg, ka::range2d_r<double_type> A  )
  {
    size_t d0 = A.dim(0);
    size_t d1 = A.dim(1);

    for (size_t i = 0; i < d0; ++i)
    {
      for (size_t j = 0; j < d1; ++j)
	printf(" %.2f", A(i, j));
      printf("\n");
    }
    printf("\n");
  }
};

/**
*/
struct TaskSeqMatProduct: public ka::Task<3>::Signature<
      ka::R<ka::range2d<double_type> >, /* A */
      ka::R<ka::range2d<double_type> >,  /* B */
      ka::RW<ka::range2d<double_type> >   /* C */
>{};

template<>
struct TaskBodyCPU<TaskSeqMatProduct> {
  void operator()( ka::range2d_r<double_type> A, ka::range2d_r<double_type> B, ka::range2d_rw<double_type> C )
  {
    size_t N = A.dim(0);
    size_t M = B.dim(0);
    size_t K = C.dim(1);

#if 0    
    /* a call to blas should be more performant here */
    cblas_dgemm(
        CblasRowMajor, CblasNoTrans, CblasNoTrans,
        M, N, K, 1.0, 
        A.ptr(), A.lda(),
        B.ptr(), B.lda(),
        1.0, 
        C.ptr(), C.lda()
    );
#else
    for (size_t i =0; i<N;++i)
      for (size_t j =0; j<M; ++j)
        for (size_t k =0; k<K; ++k)
          C(i,j) += A(i,k)*B(k,j);
#endif
  }
};


// TaskSeqMatProduct gpu implementation

__global__ void mulKernel
(const double_type* a, const double_type* b, double_type* c, unsigned int m)
{
  // compute a * b = c;
  // a, b, c of size m x m
  // ldN the leading dimension

#if 1

  const unsigned int mm = m * m;

  const unsigned int per_thread = mm / blockDim.x;

  unsigned int i = threadIdx.x * per_thread;

  double_type* cpos = c + i;
  double_type* cend = cpos + per_thread;

  if (threadIdx.x == (blockDim.x - 1)) cend = c + mm;

  __syncthreads();

  // foreach c elem
  for (; cpos != cend; ++cpos, ++i)
  {
    const double_type* apos = a + (i / m) * m; // i / m rounded...
    const double_type* bpos = b + i % m;

    // ... res = innerprod(aik, bkj);
    double_type res = 0;
    for (unsigned int k = 0; k < m; ++k, ++apos, bpos += m)
      res += (*apos) * (*bpos);

    // update c
    *cpos += res;
  }

#elif 0

  if ((threadIdx.x == 0) && (threadIdx.y == 0))
  {
    for (unsigned int i = 0; i < m; ++i)
      for (unsigned int j = 0; j < m; ++j)
	for (unsigned int k = 0; k < m; ++k)
	  c[i * m + j] += a[i * m + k] * b[k * m + j];
  }

#endif
}

template<>
struct TaskBodyGPU<TaskSeqMatProduct> {
  void operator()
  (
   ka::gpuStream stream,
   ka::range2d_r<double_type> A,
   ka::range2d_r<double_type> B,
   ka::range2d_rw<double_type> C
  )
  {
#if 0
    printf("mulKernel(%lx, %lx, %lx) %d %d %d\n",
	   (uintptr_t)A.ptr(),
	   (uintptr_t)B.ptr(),
	   (uintptr_t)C.ptr(),
	   C.dim(0), C.dim(1), C.lda());
#endif

    const CUstream custream = (CUstream)stream.stream;

    size_t mm = A.dim(0) * A.dim(0);
    const size_t thread_count = mm < 512 ? mm : 512;

    mulKernel<<<1, dim3(thread_count), 0, custream>>>
      (A.ptr(), B.ptr(), C.ptr(), A.dim(0));
  }
};

struct TaskMatProduct: public ka::Task<3>::Signature<
      ka::R<ka::range2d<double_type> >, /* A */
      ka::R<ka::range2d<double_type> >,  /* B */
      ka::RPWP<ka::range2d<double_type> >   /* C */
>{};

template<>
struct TaskBodyCPU<TaskMatProduct> {
  void operator()( ka::range2d_r<double_type> A, ka::range2d_r<double_type> B, ka::range2d_rpwp<double_type> C )
  {
    size_t M = A.dim(0);
    size_t K = B.dim(0);
    size_t N = B.dim(1);
    int bloc = BLOCSIZE;
    
    for (size_t i=0; i<M; i += bloc)
    {
      ka::rangeindex ri(i, i+bloc);
      for (size_t j=0; j<N; j += bloc)
      {
        ka::rangeindex rj(j, j+bloc);
        for (size_t k=0; k<K; k += bloc)
        {
          ka::rangeindex rk(k, k+bloc);
          ka::Spawn<TaskSeqMatProduct>()(A(ri,rk), B(rk,rj), C(ri,rj));
        }
      }
    }
  }
};

/* Main of the program
*/
struct doit {
  void operator()(int ac, char** av)
  {
    // av[1] = matrix_size
    // av[2] = block_count

    const int matrix_size = atoi(av[1]);
    const int block_count = atoi(av[2]);
    BLOCSIZE = matrix_size / block_count;

    const int n = matrix_size;

    double_type* dA = (double_type*) calloc(n* n, sizeof(double_type));
    double_type* dB = (double_type*) calloc(n* n, sizeof(double_type));
    double_type* dC = (double_type*) calloc(n* n, sizeof(double_type));
    if (0 == dA || 0 == dB || 0 == dC) 
    {
        std::cout << "Fatal Error. Cannot allocate matrices A, B, and C."
            << std::endl;
        return;
    }

    // Populate B and C pseudo-randomly - 
    // The matrices are populated with random numbers in the range (-1.0, +1.0)
    for(int i = 0; i < n * n; ++i) {
        dB[i] = (float) ((i * i) % 1024 - 512) / 512;
    }
    for(int i = 0; i < n * n; ++i) {
        dA[i] = (float) (((i + 1) * i) % 1024 - 512) / 512;
    }
    for(int i = 0; i < n * n; ++i) {
        dC[i] = 0.0;
    }

    ka::array<2,double_type> A(dA, n, n, n);
    ka::array<2,double_type> B(dB, n, n, n);
    ka::array<2,double_type> C(dC, n, n, n);

    // Multiply to get C = A*B 
    double t0 = kaapi_get_elapsedtime();
    ka::Spawn<TaskMatProduct>(ka::SetStaticSched())( A, B, C );
    ka::Sync();

    // dont time memory sync for the benchmarks since
    // it does not reflect the execution pipeline
    double t1 = kaapi_get_elapsedtime();

    // synchronize host memory
    kaapi_memory_synchronize();

    std::cout << t1 - t0; // seconds

    // If n is small, print the results
#if 0
    if (n <= 32) {
      ka::Spawn<TaskPrintMatrix>()( std::string("C"), C );
      ka::Sync();
    }
#endif

    free(dA);
    free(dB);
    free(dC);
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

