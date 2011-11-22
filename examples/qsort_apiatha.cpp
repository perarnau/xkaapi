//! run as: karun -np 2 --threads 2 ./fibo_apiatha 36 4

/****************************************************************************
 * 
 *  Shared usage sample : fibonnaci
 *
 ***************************************************************************/


#include <iostream>
#include "kaapi++" // this is the new C++ interface for Kaapi

// --------------------------------------------------------------------
/* qsort */


/* sort an array...
 */
struct SumBody {
  void operator() ( ka::pointer_w<int> res, 
                    ka::pointer_r<int> a, 
                    ka::pointer_r<int> b) 
  {
    /* write is used to write data to a Shared_w
     * read is used to read data from a Shared_r
     */
    *res = *a + *b;
  }
};
struct TaskSum : public ka::Task<3>::Signature<ka::W<int>, ka::R<int>, ka::R<int> > {};
template<>
struct TaskBodyCPU<TaskSum> : public SumBody { };


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
    start_time= ka::WallTimer::gettime();
    for (unsigned int i = 0 ; i < iter ; ++i)
    {   
      ka::pointer_rpwp<int> res = ka::Alloca<int>(1);
      
      ka::Fork<TaskFibo>(ka::SetLocal)( res, n );

      /* ka::SetLocal ensures that the task is executed locally (cannot be stolen) */
      ka::Fork<TaskPrint<int> >(ka::SetLocal)(res, ref_value);
    }
  }

  void operator()(int argc, char** argv )
  {
    unsigned int n = 30;
    if (argc > 1) n = atoi(argv[1]);
    unsigned int iter = 1;
    if (argc > 2) iter = atoi(argv[2]);
    
    ka::logfile() << "In main: n = " << n << ", iter = " << iter << std::endl;
    do_experiment( n, iter );
  }
};


/* main entry point : Athapascan initialization
*/
#if defined(KAAPI_USE_IPHONEOS)
void* KaapiMainThread::run_main(int argc, char** argv)
#else
int main(int argc, char** argv)
#endif
{
  try {
#if defined(KAAPI_USR_FT)
    FT::set_savehandler( &fibo_userglobal );
#endif

    /* Join the initial group of computation : it is defining
       when launching the program by a1run.
    */
    ka::Community com = ka::System::join_community( argc, argv );
    
    /* Start computation by forking the main task */
    ka::ForkMain<doit>()(argc, argv); 
    
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

