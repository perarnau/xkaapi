#include "kaapi++"
#include <iostream>


// --------------------------------------------------------------------
struct TaskWRW: public ka::Task<2>::Signature<ka::W<int>, ka::RW<int> > {};
template<>
struct TaskBodyCPU<TaskWRW> {
  void operator() ( ka::pointer_w<int> d0, ka::pointer_rw<int> d1 )
  {
  }
};
static ka::RegisterBodyCPU<TaskWRW> dummy_object_TaskWRW;

// --------------------------------------------------------------------
struct TaskR: public ka::Task<1>::Signature<ka::R<int> > {};
template<>
struct TaskBodyCPU<TaskR> {
  void operator() ( ka::pointer_r<int> d )
  {
  }
};
static ka::RegisterBodyCPU<TaskR> dummy_object_TaskR;


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

    threadgroup.print();    

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
