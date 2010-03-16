/****************************************************************************
 * 
 *  Dgem HPCC benchmark in Kaapi...
 *  On mac: export VECLIB_MAXIMUM_THREADS=1 to not use multithreaded blas
 *  T. Gautier, based on DGEMM Cilk version
 ***************************************************************************/
#include <cblas.h> 
#include <iostream>
#include "kaapi++" // this is the new C++ interface for Kaapi


#define BASE 1024

struct TaskAxpyBlas: public ka::Task<8>::Signature< 
        ka::R<double>, 
        ka::R<double>,
        ka::RW<double>,
        int,
        int, 
        int, 
        double,
        long
>  {};

template<>
struct TaskBodyCPU<TaskAxpyBlas> {
  void operator()(
      ka::Thread* thread,
      ka::pointer_r<double> A, 
      ka::pointer_r<double> B, 
      ka::pointer_rw<double> C, 
      int m, int n, int k, 
      double alpha, 
      long columnsep )
  { 
    double beta = 1.0;
    cblas_dgemm( CblasColMajor, CblasNoTrans, CblasNoTrans, m, n, k, alpha, 
                 A, columnsep, 
                 B, columnsep, beta, 
                 C, columnsep); 
  }
};

struct TaskAxpy: public ka::Task<8>::Signature< 
      ka::R<double>, 
      ka::R<double>,
      ka::RPWP<double>,
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
      ka::Thread* thread,
      ka::pointer_r<double> A, 
      ka::pointer_r<double> B, 
      ka::pointer_rpwp<double> C, 
      int m, int n, int k, 
      double alpha, 
      long columnsep )
  { 
    if (m+n+k<BASE) 
    { 
      thread->Spawn<TaskAxpyBlas>()(A, B, C, m, n, k, alpha, columnsep); 
    } 
    else if (m>=n && m>=k) 
    {
      /* The biggest dimension is m. */ 
      thread->Spawn<TaskAxpy>()(A, B, C, m/2, n, k, alpha, columnsep); 
      thread->Spawn<TaskAxpy>()(A+m/2, B, C+m/2, m-m/2, n, k, alpha, columnsep); 
    } else if (n>=m && n>=k) 
    { 
      /* The biggest dimension is n */ 
      thread->Spawn<TaskAxpy>()(A, B, C, m, n/2, k, alpha, columnsep); 
      thread->Spawn<TaskAxpy>()(A, B+(n/2)*columnsep, C+(n/2)*columnsep, m, n-n/2, k, alpha, columnsep); 
    } else { 
      /* The biggest dimension is k. */ 
      thread->Spawn<TaskAxpy>()(A, B, C, m, n, k/2, alpha, columnsep); 
      // Cilk comment: Need to store into another variable then add them. Or a sync. 
      ka::Sync();
//      // Kaapi comment: not necessary, here, C has RW access mode -> exclusive access
      thread->Spawn<TaskAxpy>()(A+(k/2)*columnsep, B+k/2, C, m, n, k-k/2, alpha, columnsep);
    } 
  }
};

struct TaskInit1: public ka::Task<5>::Signature<ka::RW<double>, ka::RW<double>, ka::RW<double>, long, long> {};

template<>
struct TaskBodyCPU<TaskInit1> {
  void operator() (ka::pointer_rw<double> A, ka::pointer_rw<double> B, ka::pointer_rw<double> C, long i, long n) 
  { 
    long j; 
    for (j=0; j<n; j++) { 
      A[j] = 3*(i+j);//(double)(random())/RAND_MAX; 
      B[j] = 3*(i+j)+1;//(double)(random())/RAND_MAX; 
      C[j] = 0; //3*(i+j)+2;//(double)(random())/RAND_MAX; 
    } 
  } 
};

struct TaskInit: public ka::Task<4>::Signature<ka::RPWP<double>, ka::RPWP<double>, ka::RPWP<double>, long> {};
template<>
struct TaskBodyCPU<TaskInit> {
  void operator() (ka::Thread* thread, 
                   ka::pointer_rpwp<double> A, 
                   ka::pointer_rpwp<double> B, 
                   ka::pointer_rpwp<double> C, 
                   long n) 
  { 
#if INITPAR
    long i; 
    for (i=0; i<n; ++i)
      thread->Spawn<TaskInit1>()(A+i*n, B+i*n, C+i*n, i*n, n); 
#else
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
        Ci[j] = 0; //3*(i+j)+2;//(double)(random())/RAND_MAX; 
      } 
    }
#endif
  }
};


void print_mat( std::ostream& cout, const char* msg, double* A, int n )
{
  cout << msg << ":=" << std::endl;
  cout << "Matrix([" << std::endl;
  for (int i=0; i<n; ++i)
  { 
    cout << "[";
    for (int j=0; j<n; ++j)
    {
      std::cout << A[i+j*n] << " ";
      if (j != n-1) cout << ",";
    }
    if (i != n-1) cout << "]," << std::endl;
    else cout << "]" << std::endl;
  }
  cout << "]);" << std::endl;
}


/* Main of the program
*/
struct doit {
  void operator()(int argc, char** argv )
  {
    assert(argc>=2); 
    long n = atoi(argv[1]);
    double* mA;
    double* mB;
    double* mC; 
    ka::pointer<double> A;
    ka::pointer<double> B;
    ka::pointer<double> C; 
    double alpha = 1.0; 
    double beta = 0.0; 

    double t0,t1;
    long long n3 = ((long long)n)*((long long)n)*((long long)n); 

    A   = mA = new double [n*n]; 
    B   = mB = new double [n*n]; 
    C   = mC = new double [n*n]; 

    t0 = ka::WallTimer::gettime();
    ka::Spawn<TaskInit>()(A, B, C, n); 
    ka::Sync();
    t1 = ka::WallTimer::gettime();
    std::cout << "Init time: " << t1 -t0 << std::endl;

if (n <16)
{
  print_mat(std::cout, "A", mA, n);
  print_mat(std::cout, "B", mB, n);
  print_mat(std::cout, "C", mC, n);
}

    t0 = ka::WallTimer::gettime();
    ka::Spawn<TaskAxpy>()(A, B, C, n, n, n, alpha, n); 
    ka::Sync();
    t1 = ka::WallTimer::gettime();
    printf("%lix%li matrix multiply %lld flops in %fs = %fMFLOPS\n", 
           n,n, n3, (t1-t0), 2*n3*1e-6/(t1-t0)); 

if (n <16)
{
  print_mat(std::cout, "C1", mC, n);
}

    /* verif */
    t0 = ka::WallTimer::gettime();
    alpha = 1.0;
    beta = -1.0;
    t0 = ka::WallTimer::gettime();
    cblas_dgemm( CblasColMajor, CblasNoTrans, CblasNoTrans, n, n, n, alpha, 
                 A, n, 
                 B, n, beta, 
                 C, n);

if (n <16)
  print_mat(std::cout, "C2", mC, n);
    
    for (int i=0;i<n*n; ++i)
    {
      if (! (C[i] <= 1e-15) ) 
      {
        std::cout << "Verif failed" << std::endl;
        exit(1);
      }
    }
    t1 = ka::WallTimer::gettime();
    std::cout << "Verif ok: " << t1 -t0 << std::endl;
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
