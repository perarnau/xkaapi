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

#if defined(STD_CXX0X)  
#include <algorithm>
#elif defined(USE_KASTL)
#include <kastl/algorithm>
#else
#include "kaapi++" // this is the new C++ interface for Kaapi
#endif

#define DEFAULT_MATRIX_SIZE 4

struct FuncApply {
  FuncApply( double* pA, double* pB, double* pC, int nn )
   : A(pA), B(pB), C(pC), n(nn)
  { }
  void operator() (int i)
  {
    int itn = i * n;
    for (int k = 0; k < n; ++k) {
        for (int j = 0; j < n; ++j) {
          int ktn = k * n;
            // Compute A[i,j] in the innner loop.
            A[itn + j] += B[itn + k] * C[ktn + j];
        }
    }
  };
  double* A;
  double* B;
  double* C;
  int n;
};

struct TaskFuncApply: public ka::Task<5>::Signature<ka::RW<double>, ka::R<double>, ka::R<double>, int, int > {};
template<>
struct TaskBodyCPU<TaskFuncApply> {
  void operator()( ka::pointer_rw<double> pA, ka::pointer_r<double> pB, ka::pointer_r<double> pC, int n, int i )
  {
    double* A = &*pA;
    const double* B = &*pB;
    const double* C = &*pC;
    for (int k = 0; k < n; ++k) {
        for (int j = 0; j < n; ++j) {
          int ktn = k * n;
            // Compute A[i,j] in the innner loop.
            A[j] += B[k] * C[ktn + j];
        }
    }
  }
};

// Multiply double precsion square n x n matrices. A = B * C
// Matrices are stored in row major order.
// A is assumed to be initialized.
void matrix_multiply(double* A, double* B, double* C, unsigned int n)
{
    if (n < 1) {
        return;
    }

#if defined(STD_CXX0X)  
    // This is the only Cilk++ keyword used in this program
		// Note the order of the loops and the code motion of the i * n and k * n
		// computation. This gives a 5-10 performance improvment over exchanging
		// the j and k loops.
    kastl::counting_iterator<int> beg(0);
    kastl::counting_iterator<int> end(n);
    kastl::for_each( beg, end, [&](int i)->void {
      int itn = i * n;
      for (unsigned int k = 0; k < n; ++k) {
          for (unsigned int j = 0; j < n; ++j) {
            int ktn = k * n;
              // Compute A[i,j] in the innner loop.
              A[itn + j] += B[itn + k] * C[ktn + j];
          }
      }
    })
#elif defined(USE_KASTL)
    // This is the only Cilk++ keyword used in this program
		// Note the order of the loops and the code motion of the i * n and k * n
		// computation. This gives a 5-10 performance improvment over exchanging
		// the j and k loops.
    kastl::counting_iterator<int> beg(0);
    kastl::counting_iterator<int> end(n);
    FuncApply unop(A,B,C,n);
    kastl::for_each( beg, end, unop );
#else
    for (unsigned int i=0; i<n; ++i)
    {
      int itn = i * n;
      ka::Spawn<TaskFuncApply>() (A+itn, B+itn, C, n, i);
    }
    ka::Sync();   
#endif
    return;
}

void print_matrix(double* M, int nn)
{
    for (int i = 0; i < nn; ++i) {
        for (int j = 0; j < nn; ++j) {
            std::cout << M[i * nn + j] << ",  ";
        }
        std::cout << std::endl;
    }
    return;
}

int main(int argc, char** argv) {
    // Create random input matrices. Override the default size with argv[1]
    // Warning: Matrix indexing is 0 based.
    int nn = DEFAULT_MATRIX_SIZE;
    if (argc > 1) {
        nn = std::atoi(argv[1]);
    }

    std::cout << "Simple algorithm: Mulitply two " << nn << " by " << nn
        << " matrices, coumputing A = B*C" << std::endl;

    double* A = (double*) calloc(nn* nn, sizeof(double));
    double* B = (double*) calloc(nn* nn, sizeof(double));
    double* C = (double*) calloc(nn* nn, sizeof(double));
    if (NULL == A || NULL == B || NULL == C) {
        std::cout << "Fatal Error. Cannot allocate matrices A, B, and C."
            << std::endl;
        return 1;
    }

    // Populate B and C pseudo-randomly - 
    // The matrices are populated with random numbers in the range (-1.0, +1.0)
    for(int i = 0; i < nn * nn; ++i) {
        B[i] = (float) ((i * i) % 1024 - 512) / 512;
    }
    for(int i = 0; i < nn * nn; ++i) {
        C[i] = (float) (((i + 1) * i) % 1024 - 512) / 512;
    }

    // Multiply to get A = B*C 
    double t0 = kaapi_get_elapsedtime();
    matrix_multiply(A, B, C, (unsigned int)nn);
    double t1 = kaapi_get_elapsedtime();
    float par_time = t1-t0;
    std::cout << " Matrix Multiply took " << par_time << " seconds."
        << std::endl;

    // If n is small, print the results
    if (nn <= 8) {
        std::cout << "Matrix A:" << std::endl;
        print_matrix(B, nn);
        std::cout << std::endl << "Matrix B:" << std::endl;
        print_matrix(C, nn);
        std::cout << std::endl << "Matrix C = A * B:" << std::endl;
        print_matrix(A, nn);
    }

    free(A);
    free(B);
    free(C);
    return 0;
}
