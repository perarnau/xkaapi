#include "kaapi++"
#include <iostream>

enum Direction {
  LEFT,
  RIGHT
};

// --------------------------------------------------------------------
struct TaskInit: public ka::Task<3>::Signature<int, ka::W<double>, double > {};
template<>
struct TaskBodyCPU<TaskInit> {
  void operator() ( int pos, ka::pointer_w<double> D, double v )
  {
    *D = v;
    std::cout << ka::System::local_gid << "::[TaskInit] pos:" << pos << ", D: 1.0" << std::endl;
  }
};


// --------------------------------------------------------------------
struct TaskPrint: public ka::Task<2>::Signature<int, ka::R<double> > {};
template<>
struct TaskBodyCPU<TaskPrint> {
  void operator() ( int pos, ka::pointer_r<double> D )
  {
    std::cout << ka::System::local_gid << "::[TaskPrint] pos:" << pos 
              << ", newV: @:" << &*D << ", v:" << *D
              << std::endl;
  }
};


// --------------------------------------------------------------------
struct TaskUpdate3: public ka::Task<4>::Signature<int, ka::RW<double>, ka::R<double>, ka::R<double> > {};
template<>
struct TaskBodyCPU<TaskUpdate3> {
  void operator() ( int pos, ka::pointer_rw<double> D, ka::pointer_r<double> f1, ka::pointer_r<double> f2 )
  {
    double old = *D;
    *D = *D*0.5 + *f1*0.25 + *f2*0.25;
    std::cout << ka::System::local_gid << "::[TaskUpdate3] pos:" << pos 
              << ", oldv: @:" << &*D << ", v:" << old 
              << ", f1: @" << &*f1 << ", v:" << *f1 
              << ", f2: @" << &*f2 << ", v:" << *f2 << ", newV: @:" << &*D << " v:" << *D
              << std::endl;
  }
};


// --------------------------------------------------------------------
struct TaskExtractF: public ka::Task<3>::Signature<int, ka::W<double>, ka::R<double> > {};
template<>
struct TaskBodyCPU<TaskExtractF> {
  void operator() ( int pos, ka::pointer_w<double> F, ka::pointer_r<double> D )
  {
    double v = *D;
    *F = v;
    std::cout << ka::System::local_gid << "::[TaskExtractF] pos:" << pos << ", F:" << v << std::endl;
  }
};


struct MyBlockCyclicMapping {
  MyBlockCyclicMapping( int size, int bloc )
   : _size(size), _bloc(bloc)
  {}
  int operator()(int nodecount, int tid)
  {
    int gid = (tid / _bloc)%nodecount;
    return gid;
  }
private:
  int _size; 
  int _bloc;
};

/* Main of the program
*/
struct doit {
  void operator()(int argc, char** argv )
  {
    std::cout << "My pid=" << getpid() << std::endl;
    int size = 4;
    int bloc = 1;
//    if (argc >1)
//      size = atoi(argv[1]);
//    if (argc >2)
//      bloc = atoi(argv[2]);

    int n = size*bloc;
    ka::ThreadGroup threadgroup( 4 );
    std::vector<double> D(n);      /* domaine */
    std::vector<double> F(n);

    MyBlockCyclicMapping map(2, 2);
    threadgroup.begin_partition( map );

    for (int i=0; i<size; ++i)
    {
      ka::Spawn<TaskInit> (ka::SetPartition(i))  ( i, &D[i], ((i==0) || (i==size-1)) ? 1.0 : 0.0 );
    }
    threadgroup.end_partition();

    threadgroup.execute();


    threadgroup.begin_partition( map, KAAPI_THGRP_SAVE_FLAG );
    for (int step = 0; step < 2; ++step)
    {
      for (int i=0; i<size; ++i)
      {
        ka::Spawn<TaskExtractF> (ka::SetPartition(i))  ( i, &F[i], &D[i] );
      }
      for (int i=0; i<size; ++i)
      {
        if ((i !=0) && (i !=size-1))
          ka::Spawn<TaskUpdate3>   (ka::SetPartition(i))  ( i, &D[i], &F[i-1], &F[i+1] );
      }
      for (int i=0; i<size; ++i)
      {
        ka::Spawn<TaskPrint> (ka::SetPartition(i))  ( i, &D[i] );
      }
    }
    threadgroup.end_partition();

    threadgroup.set_iteration_step( 2 );
    threadgroup.execute();
    
    printf("\n\n***********\n\n");
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
