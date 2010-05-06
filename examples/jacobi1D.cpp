#include "kaapi++"
#include <iostream>

#define SHOWTASKS false
#define SHOWPRINT true
#define SHOW_INFO false


float unit = 0;

void work(int n){
  volatile long double x = 5469461.13516;
  for (int i = 0 ; i < n ; i++){
    for (int j = 0 ; j < 10 ; j++){
      x = x*3546876.1684;
      x = x/7864643.1641;
    }
  }
}

// --------------------------------------------------------------------
struct InternalComp: public ka::Task<4>::Signature< int, int, ka::R<int>, ka::RW<int> > {};
template<>
struct TaskBodyCPU<InternalComp> {
  void operator() ( int iter, int i, ka::pointer_r<int> domain_old, ka::pointer_rw<int> domain_new )
  {
    int value1 = *domain_old;
    int value2 = *domain_new;
    *domain_new += *domain_old;
    int value3 = *domain_new;
    if (SHOWTASKS)
    {
      ka::logfile() << "\t*** [Jacobi1D::InternalComp] iter = " << iter
                      << " domain_old[" << i << "] = " << value1
                      << " domain_new[" << i << "] = " << value2
                      << " --> domain_new[" << i << "] = " << value3 << std::endl;      
    }
    work((int)unit);
  }
};


struct ExternalComp: public ka::Task<4>::Signature< int, int, ka::R<int>, ka::RW<int> > {};
template<>
struct TaskBodyCPU<ExternalComp> {
  void operator() ( int iter, int i,  
                    ka::pointer_r<int> domain_old, 
                    ka::pointer_rw<int> domain_new )
  {
    int value1 = *domain_old;
    int value2 = *domain_new;
    *domain_new += 1 + *domain_old;
    int value3 = *domain_new;
    if (SHOWTASKS)
    {
      ka::logfile() << "\t*** [Jacobi1D::ExternalComp] iter = " << iter
                      << " domain_old[" << i << "] = " << value1
                      << " domain_new[" << i << "] = " << value2
                      << " --> domain_new[" << i << "] = " << value3 << std::endl;
    }
    work((int)unit);
  }
};

struct IntegralComp: public ka::Task<4>::Signature< int, int, ka::R<int>, ka::W<int> > {};
template<>
struct TaskBodyCPU<IntegralComp> {
  void operator() ( int iter, int i, ka::pointer_r<int> domain_old, ka::pointer_w<int> domain_new )
  {
    int value1 = *domain_old;
    *domain_new = *domain_old;
    int value2 = *domain_new;
    if (SHOWTASKS)
    {
      ka::logfile() << "\t*** [Jacobi1D::IntegralComp] iter = " << iter
                      << " domain_old[" << i << "] = " << value1
                      << " --> domain_new[" << i << "] = " << value2 << std::endl;
    }
    work((int)unit);
  }
};

struct PrintB: public ka::Task<4>::Signature< int, int, ka::R<int>, ka::R<int> > {};
template<>
struct TaskBodyCPU<PrintB> {
  void operator() ( int iter, int i, ka::pointer_r<int> domain1, ka::pointer_r<int> domain2 )
  {
    int value1 = *domain1;
    int value2 = *domain2;
    ka::logfile() <<  "\t*** [Jacobi1D::PrintB] iter = " << iter
                    << " domain1[" << i << "] = " << value1
                    << " domain2[" << i << "] = " << value2 << std::endl;
  }
};

struct PrintE: public ka::Task<4>::Signature< int, int, ka::R<int>, ka::R<int> > {};
template<>
struct TaskBodyCPU<PrintE> {
  void operator() ( int iter, int i, ka::pointer_r<int> domain1, ka::pointer_r<int> domain2 )
  {    
    int value1 = *domain1;
    int value2 = *domain2;
    ka::logfile() << "\t*** [Jacobi1D::PrintE] iter = " << iter
                    << " domain1[" << i << "] = " << value1
                    << " domain2[" << i << "] = " << value2 << std::endl;      
  }
};

struct Jacobi1D {
  void operator()( int n, int k )
  {
    ka::ThreadGroup threadgroup( 2 );
    ka::auto_pointer<int> domain      = new int[ n ];
    ka::auto_pointer<int> domain_new  = new int[ n ];

    std::cout << "#size=" << n << std::endl;
    std::cout << "#iter=" << k << std::endl;
    int ip1;
    
    threadgroup.begin_partition();
    /* graph description for k iterations */
    for (int step=0; step < k; ++step) 
    {
      if (SHOWPRINT)
      { 
        /* print loop */
        for (int i=0; i < n; ++i) 
        {
          threadgroup[rand()%2]->Spawn<PrintB>()( step, i, domain+i, domain_new+i );
        }
      }
      /* compute loop : internal force */
      for (int i=0; i < n; ++i) 
      {
        threadgroup[rand()%2]->Spawn<InternalComp> () ( step, i, domain+i, domain_new+i );
      }
      /* compute loop : external force */
      for (int i=0; i < n; ++i) 
      {
        if (i ==n-1) ip1 = 0;
        else ip1 = i+1;
        threadgroup.Spawn<ExternalComp> (ka::SetPartition(1))  ( step, i, domain+ip1, domain_new+i );
      }

      /* integral loop */
      for (int i=0; i < n; ++i) 
      {
        threadgroup.Spawn<IntegralComp> (ka::SetPartition(0)) ( step, i, domain_new+i, domain+i );
      }
      if (SHOWPRINT)
      {
        /* print loop */
        for (int i=0; i < n; ++i) 
        {
          threadgroup[rand()%2]->Spawn<PrintE> () ( step, i, domain+i, domain_new+i );
        }
      }
    }
    threadgroup.end_partition();

    threadgroup.execute();
  }
};



/* Main of the program
*/
struct doit {
  void operator()(int argc, char** argv )
  {
    std::cout << "My pid=" << getpid() << std::endl;
    int n = atoi(argv[1]);
    int k = atoi(argv[2]);
    int p = atoi(argv[3]);
    int iter = atoi(argv[4]);

    Jacobi1D()(n,k);
  }
};

int main( int argc, char** argv ) 
{
  float grain = 0;
  if (argc > 5) grain = atof(argv[5]);
  if (grain < 0) grain = 0;
  if (SHOW_INFO) std::cout << "grain: " << grain << std::endl;

  // Calibrate work
  double timer;
  timer = kaapi_get_elapsedtime();
  work(1000000);
  timer = kaapi_get_elapsedtime() - timer;
  if (SHOW_INFO) std::cout << "Calibrate: time = " << timer << std::endl;
  unit = (1000.0/timer); // 1 ms
  if (SHOW_INFO) std::cout << "Calibrate: unit = " << unit << std::endl;
  unit = grain*unit; // 1 ms
  if (SHOW_INFO) std::cout << "Calibrate: grain*unit = " << unit << std::endl;

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
  catch (const ka::InvalidArgumentError& E) {
    ka::logfile() << "Catch invalid arg" << std::endl;
  }
  catch (const ka::BadAlloc& E) {
    ka::logfile() << "Catch bad alloc" << std::endl;
  }
  catch (const ka::Exception& E) {
    ka::logfile() << "Catch : "; E.print(std::cout); std::cout << std::endl;
  }
  catch (...) {
    ka::logfile() << "Catch unknown exception: " << std::endl;
  }
  return 0;    
}
