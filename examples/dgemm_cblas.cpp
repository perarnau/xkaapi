/****************************************************************************
 * 
 *  Dgem HPCC benchmark in Kaapi...
 *  T. Gautier, based on DGEMM Cilk version
 ***************************************************************************/
#include <cblas.h> 
#include <iostream>
#include "kaapi++.h" /* time function */

struct TaskInit {
  void operator() (double* A, double* B, double* C, long n) 
  { 
    long i; 
    for (i=0; i<n*n; i+=n)
    {
      long j; 
      double* Ai = A+i;
      double* Bi = B+i;
      double* Ci = C+i;
      for (j=0; j<n; j++) { 
        Ai[j] = 3*(i+j);//(double)(random())/RAND_MAX; 
        Bi[j] = 3*(i+j)+1;//(double)(random())/RAND_MAX; 
        Ci[j] = 3*(i+j)+2;//(double)(random())/RAND_MAX; 
      } 
    }
  }
};


/* Main of the program
*/
struct doit {
  void operator()(int argc, char** argv )
  {
    assert(argc==2); 
    long n = atoi(argv[1]);
    double* A;
    double* B;
    double* C; 
    double alpha = 1; 

    double t0,t1;
    long long n3 = ((long long)n)*((long long)n)*((long long)n); 

    A   = new double [n*n]; 
    B   = new double [n*n]; 
    C   = new double [n*n]; 

    t0 = ka::WallTimer::gettime();
    TaskInit()(A, B, C, n); 
    t1 = ka::WallTimer::gettime();
    std::cout << "Init time: " << t1 -t0 << std::endl;

    double beta = 1.0;
    t0 = ka::WallTimer::gettime();
    cblas_dgemm( CblasColMajor, CblasNoTrans, CblasNoTrans, n, n, n, alpha, 
                 A, n, 
                 B, n, beta, 
                 C, n); 
    t1 = ka::WallTimer::gettime();
    printf("%lix%li blas matrix multiply %lld flops in %fs = %fMFLOPS\n", 
           n,n, n3, (t1-t0), 2*n3*1e-6/(t1-t0)); 
  }
};


/* main entry point : Kaapi initialization
*/
int main(int argc, char** argv)
{
//  setenv("VECLIB_MAXIMUM_THREADS", "2", 1);
  /* Start computation by forking the main task */
  doit()(argc, argv); 
    
  return 0;
}
