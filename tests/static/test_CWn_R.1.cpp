#include "kaapi++"
#include <iostream>


// --------------------------------------------------------------------
struct TaskCW: public ka::Task<1>::Signature<ka::CW<int> > {};
template<>
struct TaskBodyCPU<TaskCW> {
  void operator() ( ka::pointer_cw<int> d )
  {
    *d += 1;
    std::cout << ka::System::local_gid << "::In Task CW=" << std::endl;
  }
};

// --------------------------------------------------------------------
struct TaskR: public ka::Task<1>::Signature<ka::R<int> > {};
template<>
struct TaskBodyCPU<TaskR> {
  void operator() ( ka::pointer_r<int> d )
  {
    std::cout << ka::System::local_gid << "::In Task R=" << *d << ", @:" << (int*)&*d << std::endl;
  }
};


/* Main of the program
*/
struct doit {
  void operator()(int argc, char** argv )
  {
    int iter = 10;
    std::cout << "My pid=" << getpid() << std::endl;

    ka::ThreadGroup threadgroup( 4 );
    ka::auto_pointer<int> a      = ka::Alloca<int>(1);
    *a = 123;

    threadgroup.begin_partition(KAAPI_THGRP_SAVE_FLAG);

    threadgroup.Spawn<TaskR> (ka::SetPartition(0))  ( a );
    threadgroup.Spawn<TaskCW> (ka::SetPartition(1)) ( a );
    threadgroup.Spawn<TaskCW> (ka::SetPartition(2)) ( a );
    threadgroup.Spawn<TaskCW> (ka::SetPartition(3)) ( a );
    threadgroup.Spawn<TaskCW> (ka::SetPartition(1)) ( a );

    threadgroup.Spawn<TaskR> (ka::SetPartition(0))  ( a );
    threadgroup.Spawn<TaskR> (ka::SetPartition(1))  ( a );
    threadgroup.Spawn<TaskCW> (ka::SetPartition(1)) ( a );
    threadgroup.Spawn<TaskR> (ka::SetPartition(0))  ( a );

    threadgroup.end_partition();

    threadgroup.set_iteration_step( iter );
    
    for (int k=0; k<iter; ++k)
    {
      threadgroup.execute();
      
      printf("\n\n***********\n\n");
    }
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
