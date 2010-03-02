//! run as: karun -np 2 --threads 2 ./fibo_apiatha 36 4

/****************************************************************************
 * 
 *  Shared usage sample : fibonnaci
 *
 ***************************************************************************/

#include <iostream>
#include "kaapi++" // this is the new C++ interface for Kaapi

// --------------------------------------------------------------------
extern long fiboseq( const long n );

extern long fiboseq_On(const long n);

long cutoff;
int ct;

/* Print any typed shared
 * this task has read acces on a, it will wait until previous write acces on it are done
*/
static double start_time;
static double stop_time;
template<class T>
struct PrintBody {
  void operator() ( ka::pointer_r<T> a, int niter, const T& ref_value)
  { 
    /*  ka::WallTimer::gettime is a wrapper around gettimeofday(2) */
//    double t1 = ka::WallTimer::gettime();
    double delay = stop_time - start_time;

    /*  ka::System::getRank() prints out the id of the node executing the task */
    ka::logfile() << ka::System::getRank() << ": -----------------------------------------" << std::endl;
    ka::logfile() << ka::System::getRank() << ": res  = " << *a << std::endl;
    ka::logfile() << ka::System::getRank() << ": cutoff = " << cutoff << ", time = " << delay/niter << " s" << std::endl;
    ka::logfile() << ka::System::getRank() << ": -----------------------------------------" << std::endl;
  }
};

/* Description of Update task */
template<class T>
struct TaskPrint : public ka::Task<3>::Signature<ka::R<T>, int, const T&> {};

/* Specialize default / CPU */
template<class T>
struct TaskBodyCPU<TaskPrint<T> > : public TaskPrint<T> { 
  void operator() ( ka::pointer_r<T> a, int niter, const T& ref_value )
  { 
    PrintBody<T>()(a, niter, ref_value); 
  }
};


/* Kaapi Fibo task.
   A Task is a type with respect a given signature. The signature specifies the number of arguments (2),
   and the type and access mode for each parameters.
   Here the first parameter is declared with a write mode. The second is passed by value.
 */
struct TaskFibo : public ka::Task<2>::Signature<ka::W<long>, const long > {};

/* Main of the program
*/
struct doit {

  void do_experiment(unsigned int n, unsigned int iter )
  {
    double t = ka::WallTimer::gettime();
    int ref_value = fiboseq_On(n);
    double delay = ka::WallTimer::gettime() - t;
    ka::logfile() << "[fibo_apiatha] Sequential value for n = " << n << " : " << ref_value 
                    << " (computed in " << delay << " s)" << std::endl;
      
    kaapi_stack_t* stack = kaapi_self_stack();
    kaapi_frame_t frame;
    ka::pointer<long> res = ka::Alloca<long>(1);
    for (cutoff=2; cutoff<3; ++cutoff)
    {
      kaapi_stack_save_frame(stack, &frame);
      ka::Spawn<TaskFibo>()( res, n );
      /* */
      ka::Sync();
      start_time= ka::WallTimer::gettime();
      for (unsigned int i = 0 ; i < iter ; ++i)
      {   
        ka::Spawn<TaskFibo>()( res, n );
      }
      /* */
      ka::Sync();
      stop_time= ka::WallTimer::gettime();
      /* ka::SetLocal ensures that the task is executed locally (cannot be stolen) */
      ka::Spawn<TaskPrint<long> >()(res, iter, ref_value);      
      ka::Sync();
      kaapi_stack_restore_frame(stack, &frame);
    }
  }

  void operator()(int argc, char** argv )
  {
    unsigned int n = 30;
    if (argc > 1) n = atoi(argv[1]);
    unsigned int iter = 1;
    if (argc > 2) iter = atoi(argv[2]);
    cutoff = 2;
    if (argc > 3) cutoff = atoi(argv[3]);
    ct = 0;
    if (argc > 4) ct = atoi(argv[4]);
    
    ka::logfile() << "In main: n = " << n << ", iter = " << iter << ", cutoff = " << cutoff << std::endl;
    do_experiment( n, iter );
  }
};


/* main entry point : Kaapi initialization
*/
int main(int argc, char** argv)
{
  try {
    /* Join the initial group of computation : it is defining
       when launching the program by a1run.
    */
    ka::Community com = ka::System::join_community( argc, argv );
    
    /* Start computation by forking the main task */
    ka::SpawnMain<doit>()(argc, argv); 
    
    /* Leave the community: at return to this call no more athapascan
       tasks or shared could be created.
    */
    com.leave();

    /* */
    ka::System::terminate();
  }
  catch (const ka::InvalidArgumentError& E) {
    ka::logfile() << "Catch invalid arg" << std::endl;
  }
  catch (const ka::BadAlloc& E) {
    ka::logfile() << "Catch bad alloc" << std::endl;
  }
  catch (const ka::Exception& E) {
    ka::logfile() << "Catch : "; E.print(std::cout); std::cout << std::endl;
  }
  catch (...) {
    ka::logfile() << "Catch unknown exception: " << std::endl;
  }
  
  return 0;
}

