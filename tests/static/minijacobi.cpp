#include "kaapi++"
#include <iostream>

enum Direction {
  LEFT,
  RIGHT
};

// --------------------------------------------------------------------
struct TaskUpdate: public ka::Task<3>::Signature<int, ka::R<double>, ka::W<double> > {};
template<>
struct TaskBodyCPU<TaskUpdate> {
  void operator() ( int bloc, ka::pointer_r<double> oD, ka::pointer_w<double> nD )
  {
  }
};

// --------------------------------------------------------------------
struct TaskSwap: public ka::Task<3>::Signature<int, ka::R<double>, ka::W<double> > {};
template<>
struct TaskBodyCPU<TaskSwap> {
  void operator() ( int bloc, ka::pointer_r<double> oD, ka::pointer_w<double> F )
  {
  }
};

// --------------------------------------------------------------------
struct TaskExtractF: public ka::Task<4>::Signature<Direction, int, ka::R<double>, ka::W<double> > {};
template<>
struct TaskBodyCPU<TaskExtractF> {
  void operator() ( Direction d, int bloc, ka::pointer_r<double> oD, ka::pointer_w<double> F )
  {
  }
};

// --------------------------------------------------------------------
struct TaskUpdateFontier: public ka::Task<4>::Signature<Direction, int, ka::RW<double>, ka::R<double> > {};
template<>
struct TaskBodyCPU<TaskUpdateFontier> {
  void operator() ( Direction d, int bloc, ka::pointer_rw<double> nD, ka::pointer_r<double> f )
  {
  }
};


/* Main of the program
*/
struct doit {
  void operator()(int argc, char** argv )
  {
    std::cout << "My pid=" << getpid() << std::endl;
    int size = 4;
    int bloc = 1;
    if (argc >1)
      size = atoi(argv[1]);
    if (argc >2)
      bloc = atoi(argv[2]);

    int n = size*bloc;
    ka::ThreadGroup threadgroup( size );
    std::vector<double> oD(n);      /* bloc cyclic mapping */
    std::vector<double> nD(n);      /* bloc cyclic mapping */
    std::vector<double> Fr[2];
    Fr[0].resize(size-1); 
    Fr[1].resize(size-1); 
    std::vector<double> Fl[2];
    Fl[0].resize(size-1); 
    Fl[1].resize(size-1); 

    threadgroup.begin_partition();

    for (int step = 0; step <1; ++step)
    {
      for (int i=0; i<size; ++i)
      {
        if (i != (size-1)) /* right fontier */
          threadgroup.Spawn<TaskExtractF> (ka::SetPartition(i))  ( RIGHT, bloc, &oD[(i*bloc)], &Fr[step%2][i] );
        if (i != 0) /* left fontier */
          threadgroup.Spawn<TaskExtractF> (ka::SetPartition(i))  ( LEFT, bloc, &oD[(i*bloc)], &Fl[step%2][i-1] );
      }
      for (int i=0; i<size; ++i)
      {
        threadgroup.Spawn<TaskUpdate>   (ka::SetPartition(i))  ( bloc, &oD[(i*bloc)], &nD[(i*bloc)] );
      }
      for (int i=0; i<size; ++i)
      {
        if ( i != size-1)
          threadgroup.Spawn<TaskUpdateFontier>(ka::SetPartition(i))  ( RIGHT, bloc, &nD[(i*bloc)], &Fl[step%2][i] );
        if ( i != 0)
          threadgroup.Spawn<TaskUpdateFontier>(ka::SetPartition(i))  ( LEFT, bloc, &nD[(i*bloc)], &Fr[step%2][i-1] );
      }
      for (int i=0; i<size; ++i)
      {
        threadgroup.Spawn<TaskSwap>(ka::SetPartition(i))  ( bloc, &nD[(i*bloc)], &oD[(i*bloc)] );
      }
    }
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
