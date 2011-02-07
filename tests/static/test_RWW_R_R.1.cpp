#include "kaapi_impl.h"
#include "kaapi++"
#include <iostream>


// --------------------------------------------------------------------
struct TaskRWW: public ka::Task<2>::Signature<ka::RW<int>, ka::W<int> > {};
template<>
struct TaskBodyCPU<TaskRWW> {
  void operator() ( ka::pointer_rw<int> d0, ka::pointer_w<int> d1 )
  {
    *d0 += 1;
    *d1 = 123;
    kaapi_processor_t* kproc = kaapi_get_current_processor();
    printf("[%p->%p] :: In TaskRWW=(1,123)\n",
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
    ka::auto_pointer<int> b      = ka::Alloca<int>(1);
    *a = 0;
    *b = 0;

    threadgroup.begin_partition();

    threadgroup.Spawn<TaskRWW> (ka::SetPartition(0))  ( a,b );
    threadgroup.Spawn<TaskR>  (ka::SetPartition(1))  ( a );
    threadgroup.Spawn<TaskR>  (ka::SetPartition(1))  ( b );

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
  catch (const std::exception& E) {
    ka::logfile() << "Catch : " << E.what() << std::endl;
  }
  catch (...) {
    ka::logfile() << "Catch unknown exception: " << std::endl;
  }
  return 0;    
}
