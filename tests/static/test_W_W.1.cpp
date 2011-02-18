#include "kaapi_impl.h"
#include "kaapi++"
#include <iostream>


// --------------------------------------------------------------------
struct TaskW1: public ka::Task<2>::Signature<ka::W<int>, int > {};
template<>
struct TaskBodyCPU<TaskW1> {
  void operator() ( ka::pointer_w<int> d, int value )
  {
    *d = value;
    kaapi_processor_t* kproc = kaapi_get_current_processor();
    printf("[%p->%p] :: In Task W1=%i\n",(void*)kproc, (void*)kproc->thread, value);
    usleep(100);
  }
};

// --------------------------------------------------------------------
struct TaskW2: public ka::Task<2>::Signature<ka::W<int>, int > {};
template<>
struct TaskBodyCPU<TaskW2> {
  void operator() ( ka::pointer_w<int> d, int value )
  {
    *d = value;
    kaapi_processor_t* kproc = kaapi_get_current_processor();
    printf("[%p->%p] :: In Task W2=%i\n",(void*)kproc, (void*)kproc->thread, value);
    usleep(100);
  }
};


// --------------------------------------------------------------------
struct TaskR1: public ka::Task<1>::Signature<ka::R<int> > {};
template<>
struct TaskBodyCPU<TaskR1> {
  void operator() ( ka::Thread* thread, ka::pointer_r<int> d )
  {
    kaapi_processor_t* kproc = kaapi_get_current_processor();
    printf("[%p->%p] :: In Task R1=%i\n",
	(void*)kproc, (void*)kproc->thread, *d);
    usleep(100);
  }
};


// --------------------------------------------------------------------
struct TaskR2: public ka::Task<1>::Signature<ka::R<int> > {};
template<>
struct TaskBodyCPU<TaskR2> {
  void operator() ( ka::Thread* thread, ka::pointer_r<int> d )
  {
    kaapi_processor_t* kproc = kaapi_get_current_processor();
    printf("[%p->%p] :: In Task R2=%i\n",
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

#warning "BUG: bad dependencies"
    threadgroup.begin_partition();

    threadgroup.Spawn<TaskW1> (ka::SetPartition(0))  ( a, 10 );
    threadgroup.Spawn<TaskW2> (ka::SetPartition(0))  ( a, 20 );
    threadgroup.Spawn<TaskR1> (ka::SetPartition(0))  ( a );
    threadgroup.Spawn<TaskR2> (ka::SetPartition(1))  ( a );

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
