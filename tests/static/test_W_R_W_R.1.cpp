#include "kaapi++"
#include <iostream>


// --------------------------------------------------------------------
struct TaskW1: public ka::Task<2>::Signature<ka::W<int>, int > {};
template<>
struct TaskBodyCPU<TaskW1> {
  void operator() ( ka::pointer_w<int> d, int value )
  {
    std::cout << "In Task W1=" << value << std::endl;
    *d = value;
  }
};

// --------------------------------------------------------------------
struct TaskR1: public ka::Task<1>::Signature<ka::R<int> > {};
template<>
struct TaskBodyCPU<TaskR1> {
  void operator() ( ka::pointer_r<int> d )
  {
    std::cout << "In Task R1=" << *d << std::endl;
  }
};

// --------------------------------------------------------------------
struct TaskW2: public ka::Task<2>::Signature<ka::W<int>, int > {};
template<>
struct TaskBodyCPU<TaskW2> {
  void operator() ( ka::pointer_w<int> d, int value )
  {
    std::cout << "In Task W2=" << value << std::endl;
    *d = value;
  }
};

// --------------------------------------------------------------------
struct TaskR2: public ka::Task<1>::Signature<ka::R<int> > {};
template<>
struct TaskBodyCPU<TaskR2> {
  void operator() ( ka::pointer_r<int> d )
  {
    std::cout << "In Task R2=" << *d << std::endl;
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

    threadgroup.begin_partition();

    threadgroup.Spawn<TaskW1> (ka::SetPartition(0))  ( a, 10 );
    threadgroup.Spawn<TaskR1> (ka::SetPartition(1))  ( a );
    threadgroup.Spawn<TaskW2> (ka::SetPartition(1))  ( a, 20 );
    threadgroup.Spawn<TaskR2> (ka::SetPartition(0))  ( a );

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
