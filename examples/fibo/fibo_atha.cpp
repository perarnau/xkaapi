//! run as: karun -np 2 --threads 2 ./fibo_apiatha 36 4

/****************************************************************************
 * 
 *  Shared usage sample : fibonnaci
 *
 ***************************************************************************/


#include <iostream>
#include "athapascan-1" // this is the header required by athapascan


// --------------------------------------------------------------------
/* Sequential fibo function
 */
int fiboseq(int n)
{ return (n<2 ? n : fiboseq(n-1)+fiboseq(n-2) ); }

int fiboseq_On(int n){
  if(n<2){
    return n;
  }else{

    int fibo=1;
    int fibo_p=1;
    int tmp=0;
    int i=0;
    for( i=0;i<n-2;i++){
      tmp = fibo+fibo_p;
      fibo_p=fibo;
      fibo=tmp;
    }
    return fibo;
  }
}

/* Print any typed shared
 * this task has read acces on a, it will wait until previous write acces on it are done
*/
template<class T>
struct Print {
  void operator() ( a1::Shared_r<T> a, const T& ref_value, double time )
  { 
    /*  a1::System::getRank() prints out the id of the node executing the task */
    ka::logfile() << ka::System::getRank() << ": -----------------------------------------" << std::endl;
    ka::logfile() << ka::System::getRank() << ": Res  = " << a.read() << std::endl;
    ka::logfile() << ka::System::getRank() << ": Time(s): " << time << std::endl;
    ka::logfile() << ka::System::getRank() << ": -----------------------------------------" << std::endl;
  }
};


/* Sum two integers
 * this task reads a and b (read acces mode) and write their sum to res (write access mode)
 * it will wait until previous write to a and b are done
 * once finished, further read of res will be possible
 */
struct Sum {
  void operator() ( a1::Shared_w<int> res, 
                    a1::Shared_r<int> a, 
                    a1::Shared_r<int> b) 
  {
    /* write is used to write data to a Shared_w
     * read is used to read data from a Shared_r
     */
    res.write(a.read()+b.read());
  }
};

/* Athapascan Fibo task
 * - res is the return value, return value are usually put in a Shared_w
 * - n is the order of fibonnaci. It could be a Shared_r, but there are no dependencies to compute on it, so it would be useless
 * - threshold is used to control the grain of the application. The greater it is, the more task will be created, the more parallelism there will be.
 *   a high value of threshold also decreases the performances, beacause of athapascan's overhead, choose it wisely
 */
struct Fibo {
  void operator() ( a1::Shared_w<int> res, int n )
  {  
    if (n < 2) {
      res.write( fiboseq(n) );
    }
    else {
      a1::Shared<int> res1;
      a1::Shared<int> res2;

      /* the Fork keyword is used to spawn new task
       * new tasks are executed in parallel as long as dependencies are respected
       */
      a1::Fork<Fibo>() ( res1, n-1);
      a1::Fork<Fibo>() ( res2, n-2 );

      /* the Sum task depends on res1 and res2 which are written by previous tasks
       * it must wait until thoses tasks are finished
       */
      a1::Fork<Sum>()  ( res, res1, res2 );
    }
  }
};


/* Main of the program
*/
struct doit {

  void do_experiment(unsigned int n, unsigned int iter )
  {
    double t = Util::WallTimer::gettime();
    int ref_value = fiboseq_On(n);
    double delay = Util::WallTimer::gettime() - t;
    Util::logfile() << "[fibo_atha] Sequential value for n = " << n << " : " << ref_value 
                    << " (computed in " << delay << " s)" << std::endl;

    double start_time;
    double stop_time;
    
    a1::Shared<int> res(0);
    for (long cutoff=2; cutoff<3; ++cutoff)
    {
      a1::Fork<Fibo>(a1::SetLocal)( res, n );
      /* */
      a1::Sync();
      start_time= ka::WallTimer::gettime();
      for (unsigned int i = 0 ; i < iter ; ++i)
      {   
        a1::Fork<Fibo>(a1::SetLocal)( res, n );
        /* */
        a1::Sync();
      }
      stop_time= ka::WallTimer::gettime();

      /* a1::SetLocal ensures that the task is executed locally (cannot be stolen) */
      a1::Fork<Print<int> >(a1::SetLocal)(res, ref_value, (stop_time-start_time)/iter );
    }
  }

  void operator()(int argc, char** argv )
  {
    unsigned int n = 30;
    if (argc > 1) n = atoi(argv[1]);
    unsigned int iter = 1;
    if (argc > 2) iter = atoi(argv[2]);
    
    Util::logfile() << "In main: n = " << n << ", iter = " << iter << std::endl;
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
    a1::Community com = a1::System::join_community( argc, argv );
    
    /* Start computation by forking the main task */
    a1::ForkMain<doit>()(argc, argv); 
    a1::Sync();
    
    /* Leave the community: at return to this call no more athapascan
       tasks or shared could be created.
    */
    com.leave();

    /* */
    a1::System::terminate();
  }
  catch (const std::exception& E) {
    Util::logfile() << "Catch : " << E.what() << std::endl;
  }
  catch (...) {
    Util::logfile() << "Catch unknown exception: " << std::endl;
  }
  
  return 0;
}

