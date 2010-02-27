/****************************************************************************
 * 
 *  Dgem HPCC benchmark in Kaapi...
 *  T. Gautier, based on DGEMM Cilk version
 ***************************************************************************/
#include <cblas.h> 
#include <iostream>
#include "kaapi++" // this is the new C++ interface for Kaapi


#define BASE 512 

struct TaskAxpy: public ka::Task<8>::Signature< ka::R<double>, 
                                                ka::R<double>,
                                                ka::CW<double>,
                                                int,
                                                int, 
                                                int, 
                                                double,
                                                long
                                              >  {};

/* Specialized by default only on CPU 
   C += alpha * (A*B)
   A = m x k
   B = k x n
   C = m x n
*/
template<>
struct TaskBodyCPU<TaskAxpy> {
  void operator()(
      ka::pointer_r<double> A, 
      ka::pointer_r<double> B, 
      ka::pointer_cw<double> C, 
      int m, int n, int k, 
      double alpha, long columnsep )
  { 
    if (m+n+k<BASE) 
    { 
      double beta=1; 
      cblas_dgemm( CblasColMajor, CblasNoTrans, CblasNoTrans, m, n, k, alpha, 
                   A, columnsep, 
                   B, columnsep, beta, 
                   C, columnsep); 
    } 
    else if (m>=n && m>=k) 
    {
      /* The biggest dimension is m. */ 
      ka::Spawn<TaskAxpy>()(A, B, C, m/2, n, k, alpha, columnsep); 
      ka::Spawn<TaskAxpy>()(A+m/2, B, C+m/2, m-m/2, n, k, alpha, columnsep); 
    } else if (n>=m && n>=k) 
    { 
      /* The biggest dimension is n */ 
      ka::Spawn<TaskAxpy>()(A, B, C, m, n/2, k, alpha, columnsep); 
      ka::Spawn<TaskAxpy>()(A, B+(n/2)*columnsep, C+(n/2)*columnsep, m, n-n/2, k, alpha, columnsep); 
    } else { 
      /* The biggest dimension is k. */ 
      ka::Spawn<TaskAxpy>()(A, B, C, m, n, k/2, alpha, columnsep); 
      // Cilk comment: Need to store into another variable then add them. Or a sync. 
      //sync; 
//      // Kaapi comment: not necessary, here, C has RW access mode -> exclusive access
      ka::Spawn<TaskAxpy>()(A, B, C, m, n, k/2, alpha, columnsep); 
    } 
  }
};

struct TaskInit1: public ka::Task<5>::Signature<ka::W<double>, ka::W<double>, ka::W<double>, long, long> {};

template<>
struct TaskBodyCPU<TaskInit1> {
  void operator() (ka::pointer_w<double> A, ka::pointer_w<double> B, ka::pointer_w<double> C, long i, long n) 
  { 
    long j; 
    for (j=0; j<n; j++) { 
      A[j] = 3*(i+j);//(double)(random())/RAND_MAX; 
      B[j] = 3*(i+j)+1;//(double)(random())/RAND_MAX; 
      C[j] = 3*(i+j)+2;//(double)(random())/RAND_MAX; 
    } 
  } 
};

struct TaskInit: public ka::Task<4>::Signature<ka::W<double>, ka::W<double>, ka::W<double>, long> {};
template<>
struct TaskBodyCPU<TaskInit> {
  void operator() (ka::pointer_w<double> A, ka::pointer_w<double> B, ka::pointer_w<double> C, long n) 
  { 
    long i; 
    for (i=0; i<n*n; i+=n)
      ka::Spawn<TaskInit1>()(A+i, B+i, C+i, i, n); 
  }
};


/* Main of the program
*/
struct doit {
  void operator()(int argc, char** argv )
  {
    assert(argc==2); 
    long n = atoi(argv[1]);
    ka::pointer<double> A;
    ka::pointer<double> B;
    ka::pointer<double> C; 
    double alpha = -1; 

    struct timeval tv0,tv1,tv2; 
    double tdiff_rec, tdiff_mkl; 
    long long n3 = ((long long)n)*((long long)n)*((long long)n); 

    A   = new double [n*n]; 
    B   = new double [n*n]; 
    C   = new double [n*n]; 

    ka::Spawn<TaskInit>()(A, B, C, n); 
    ka::Sync();

    gettimeofday(&tv0,0); 
    ka::Spawn<TaskAxpy>()(A, B, C, n, n, n, alpha, n); 
    ka::Sync();
    gettimeofday(&tv1,0); 
    tdiff_rec = tv1.tv_sec-tv0.tv_sec + 1e-6*(tv1.tv_usec-tv0.tv_usec); 
    printf("%lix%li matrix multiply %lld flops in %fs = %fMFLOPS\n", 
           n,n, n3, tdiff_rec, 2*n3*1e-6/tdiff_rec); 
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
  catch (const ka::Exception& E) {
    ka::logfile() << "Catch : "; E.print(std::cout); std::cout << std::endl;
  }
  catch (...) {
    ka::logfile() << "Catch unknown exception: " << std::endl;
  }
  
  return 0;
}
