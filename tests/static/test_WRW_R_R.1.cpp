#include "kaapi++"
#include <iostream>


// --------------------------------------------------------------------
struct TaskWRW: public ka::Task<2>::Signature<ka::W<int>, ka::RW<int> > {};
template<>
struct TaskBodyCPU<TaskWRW> {
  void operator() ( ka::pointer_w<int> d0, ka::pointer_rw<int> d1 )
  {
    std::cout << "In Task WRW= " << 10 << ", @:" << (int*)d0 << std::endl;
    std::cout << "In Task WRW= +" << 20 << ", @:" << (int*)d1 << std::endl;
    *d0  = 10;
    *d1 += 20;
  }
};


// --------------------------------------------------------------------
struct TaskR: public ka::Task<1>::Signature<ka::R<int> > {};
template<>
struct TaskBodyCPU<TaskR> {
  void operator() ( ka::pointer_r<int> d )
  {
    std::cout << "In Task R=" << *d << ", @:" << (int*)d << std::endl;
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

    threadgroup.begin_partition();

    threadgroup.Spawn<TaskWRW> (ka::SetPartition(0)) ( a,b );
    threadgroup.Spawn<TaskR>  (ka::SetPartition(1))  ( a );
    threadgroup.Spawn<TaskR>  (ka::SetPartition(1))  ( b );

    threadgroup.Spawn<TaskWRW> (ka::SetPartition(1)) ( a,b );
    threadgroup.Spawn<TaskR>  (ka::SetPartition(0))  ( a );
    threadgroup.Spawn<TaskR>  (ka::SetPartition(0))  ( b );

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
