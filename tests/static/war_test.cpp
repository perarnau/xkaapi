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
struct TaskUpdateIntern: public ka::Task<2>::Signature<int, ka::RW<double> > {};
template<>
struct TaskBodyCPU<TaskUpdateIntern> {
  void operator() ( int pos, ka::pointer_rw<double> D )
  {
    *D = *D * 0.5;
  }
};


// --------------------------------------------------------------------
struct TaskUpdateExtern: public ka::Task<3>::Signature<int, ka::RW<double>, ka::R<double> > {};
template<>
struct TaskBodyCPU<TaskUpdateExtern> {
  void operator() ( int pos, ka::pointer_rw<double> D, ka::pointer_r<double> f1 )
  {
    *D += *f1 * 0.25;
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
  }
};


// --------------------------------------------------------------------
struct TaskKernel: public ka::Task<4>::Signature<int, 
      ka::RW<ka::range1d<double> >, 
      ka::RW<ka::range1d<double> >,
      ka::RW<ka::range1d<double> > 
> {};
template<>
struct TaskBodyCPU<TaskKernel> {
  void operator() ( int n, ka::range1d_rw<double> D, ka::range1d_rw<double> Fright, ka::range1d_rw<double> Fleft )
  {
    for (int step = 0; step < 10; ++step)
    {
      for (int i=0; i<n; ++i)
      {
        if (i >0)
          ka::Spawn<TaskExtractF>   ()  ( i, &Fleft[i], &D[i] );
        if (i+1 <n)
          ka::Spawn<TaskExtractF>   ()  ( i, &Fright[i], &D[i] );
      }
      for (int i=0; i<n; ++i)
          ka::Spawn<TaskUpdateIntern>   ()    ( i, &D[i] );

      for (int i=0; i<n; ++i)
      {
        if (i>0)
          ka::Spawn<TaskUpdateExtern>() ( i, &D[i], &Fright[i-1] );
        if (i+1 <n)
          ka::Spawn<TaskUpdateExtern>() ( i, &D[i], &Fleft[i+1] );
      }
    }
  }
};


/* Main of the program
*/
struct doit {
  void operator()(int argc, char** argv )
  {
    std::cout << "My pid=" << getpid() << std::endl;
    int n = 100;
    int iter = 1;
    if (argc >1)
      n = atoi(argv[1]);

    std::vector<double> vD(n);      /* domain */
    std::vector<double> vF(n*2);
    
    /* view of vector */
    ka::array<1,double> D( &vD[0], n );
    ka::array<1,double> Fleft ( &vF[0], n );
    ka::array<1,double> Fright( &vF[n], n );
    
    for (int k=0; k<iter; ++k)
    {
      ka::Spawn<TaskKernel>(ka::SetStaticSched()) (n, D, Fleft, Fright);
      ka::Sync();
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
