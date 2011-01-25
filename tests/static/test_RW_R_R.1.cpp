#include "kaapi_impl.h"
#include "kaapi++"
#include <iostream>


// --------------------------------------------------------------------
struct TaskRW: public ka::Task<1>::Signature<ka::RW<int> > {};
template<>
struct TaskBodyCPU<TaskRW> {
  void operator() ( ka::pointer_rw<int> d )
  {
    *d += 100;
    kaapi_processor_t* kproc = kaapi_get_current_processor();
    printf("[%p->%p] :: In TaskRW=(+=100)\n",
      (void*)kproc, (void*)kproc->thread);
    usleep(100);
  }
};

// --------------------------------------------------------------------
struct TaskR: public ka::Task<1>::Signature<ka::R<int> > {};
template<>
struct TaskBodyCPU<TaskR> {
  void operator() ( ka::pointer_r<int> d )
  {
    kaapi_processor_t* kproc = kaapi_get_current_processor();
    printf("[%p->%p] :: In TaskR=(%i)\n",
      (void*)kproc, (void*)kproc->thread, *d);
    usleep(100);
  }
};


/* Main of the program
*/
struct doit {
  void operator()(int argc, char** argv )
  {
    std::cout << "My pid=" << getpid() << std::endl;

    ka::ThreadGroup threadgroup( 2 );
    ka::auto_pointer<int> a      = ka::Alloca<int>(1);
    *a = 1;

    threadgroup.begin_partition();

    threadgroup.Spawn<TaskRW> (ka::SetPartition(0))  ( a );
    threadgroup.Spawn<TaskR>  (ka::SetPartition(1))  ( a );
    threadgroup.Spawn<TaskR>  (ka::SetPartition(0))  ( a );

    threadgroup.end_partition();

    threadgroup.execute();
  }
};


/*
*/
int main( int argc, char** argv ) 
{
  try {
    ka::Community com = ka::System::join_community( argc, argv );
    
    ka::SpawnMain<doit>()(argc, argv); 
          
    com.leave();

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
