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
unsigned long long fiboseq(unsigned long long n)
{ return (n<2 ? n : fiboseq(n-1)+fiboseq(n-2) ); }

unsigned long long fiboseq_On(unsigned long long n){
  if(n<2){
    return n;
  }else{

    unsigned long long fibo=1;
    unsigned long long fibo_p=1;
    unsigned long long tmp=0;
    unsigned long long i=0;
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
  void operator() ( a1::Shared_r<T> a, const T& ref_value, double t )
  { 
    /*  Util::WallTimer::gettime is a wrapper around gettimeofday(2) */
    double delay = Util::WallTimer::gettime() - t;

    /*  a1::System::getRank() prints out the id of the node executing the task */
    Util::logfile() << a1::System::getRank() << ": -----------------------------------------" << std::endl;
    Util::logfile() << a1::System::getRank() << ": res  = " << a.read() << std::endl;
    Util::logfile() << a1::System::getRank() << ": time = " << delay << " s" << std::endl;
    Util::logfile() << a1::System::getRank() << ": -----------------------------------------" << std::endl;
  }
};


/* Sum two integers
 * this task reads a and b (read acces mode) and write their sum to res (write access mode)
 * it will wait until previous write to a and b are done
 * once finished, further read of res will be possible
 */
struct Sum {
  void operator() ( a1::Shared_w<unsigned long long> res, 
                    a1::Shared_r<unsigned long long> a, 
                    a1::Shared_r<unsigned long long> b) 
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
  void operator() ( a1::Shared_w<unsigned long long> res, int n )
  {  
    if (n < 2) {
      res.write( fiboseq(n) );
    }
    else {
      a1::Shared<unsigned long long> res1;
      a1::Shared<unsigned long long> res2;

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
    unsigned long long ref_value = fiboseq_On(n);
    double delay = Util::WallTimer::gettime() - t;
    Util::logfile() << "[fibo_apiatha] Sequential value for n = " << n << " : " << ref_value 
                    << " (computed in " << delay << " s)" << std::endl;
    for (unsigned int i = 0 ; i < iter ; ++i)
    {   
      double time= Util::WallTimer::gettime();

      a1::Shared<unsigned long long> res(0);
      
      a1::Fork<Fibo>(a1::SetLocal)( res, n );

      /* a1::SetLocal ensures that the task is executed locally (cannot be stolen) */
      a1::Fork<Print<unsigned long long> >(a1::SetLocal)(res, ref_value, time);
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
    
for (int i=0; i<100; ++i)
{
    /* Start computation by forking the main task */
    a1::ForkMain<doit>()(argc, argv); 
    a1::Sync();
}
    
    /* Leave the community: at return to this call no more athapascan
       tasks or shared could be created.
    */
    com.leave();

    /* */
    a1::System::terminate();
  }
  catch (const a1::InvalidArgumentError& E) {
    Util::logfile() << "Catch invalid arg" << std::endl;
  }
  catch (const a1::BadAlloc& E) {
    Util::logfile() << "Catch bad alloc" << std::endl;
  }
  catch (const a1::Exception& E) {
    Util::logfile() << "Catch : "; E.print(std::cout); std::cout << std::endl;
  }
  catch (...) {
    Util::logfile() << "Catch unknown exception: " << std::endl;
  }
  
  return 0;
}

