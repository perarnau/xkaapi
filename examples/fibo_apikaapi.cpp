//! run as: karun -np 2 --threads 2 ./fibo_apiatha 36 4

/****************************************************************************
 * 
 *  Shared usage sample : fibonnaci
 *
 ***************************************************************************/


#include <iostream>
#include "kaapi++" // this is the new C++ interface for Kaapi
#include "kaapi_impl.h" // this is the new C++ interface for Kaapi

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
static double start_time;
template<class T>
struct PrintBody {
  void operator() ( ka::pointer_r<T> a, const char* msg, const T& ref_value)
  { 
    /*  ka::WallTimer::gettime is a wrapper around gettimeofday(2) */
    double t1 = ka::WallTimer::gettime();
    double delay = t1 - start_time;
    start_time= t1;

    /*  ka::System::getRank() prints out the id of the node executing the task */
    ka::logfile() << ka::System::getRank() << ": -----------------------------------------" << std::endl;
    ka::logfile() << ka::System::getRank() << ": res  = " << *a << std::endl;
    ka::logfile() << ka::System::getRank() << ": " << msg << ", time = " << delay << " s" << std::endl;
    ka::logfile() << ka::System::getRank() << ": -----------------------------------------" << std::endl;
  }
};

/* Description of Update task */
template<class T>
struct TaskPrint : public ka::Task<3>::Signature<ka::R<T>, const char*, const T&> {};

/* Specialize default / CPU */
template<class T>
struct TaskBodyCPU<TaskPrint<T> > : public TaskPrint<T> { 
  void operator() ( ka::pointer_r<T> a, const char* msg, const T& ref_value )
  { 
    PrintBody<T>()(a, msg, ref_value); 
  }
};

void dobreak()
{
}

/* Sum two integers
 * this task reads a and b (read acces mode) and write their sum to res (write access mode)
 * it will wait until previous write to a and b are done
 * once finished, further read of res will be possible
 */
struct TaskSum : public ka::Task<3>::Signature<ka::W<int>, ka::R<int>, ka::R<int> > {};

template<>
struct TaskBodyCPU<TaskSum> : public TaskSum
{
  void operator() ( ka::pointer_w<int> res, 
                    ka::pointer_r<int> a, 
                    ka::pointer_r<int> b ) 
  {
    /* write is used to write data to a Shared_w
     * read is used to read data from a Shared_r
     */
//    if (res.ptr() == (void*)0x70040) dobreak();
    *res = *a + *b;
  }
};


/* Kaapi Fibo task.
   A Task is a type with respect a given signature. The signature specifies the number of arguments (2),
   and the type and access mode for each parameters.
   Here the first parameter is declared with a write mode. The second is passed by value.
 */
struct TaskFibo : public ka::Task<2>::Signature<ka::W<int>, int > {};


/* Implementation for CPU machine 
*/
template<>
struct TaskBodyCPU<TaskFibo> : public TaskFibo {
  void operator() ( ka::pointer_w<int> res, int n );
};

void TaskBodyCPU<TaskFibo>::operator() ( ka::pointer_w<int> res, int n )
{  
  if (n < 2) {
    *res = fiboseq(n);
  }
  else {
    ka::pointer_rpwp<int> res1 = ka::Alloca<int>(1);
    ka::pointer_rpwp<int> res2 = ka::Alloca<int>(1);

    /* the Spawn keyword is used to spawn new task
     * new tasks are executed in parallel as long as dependencies are respected
     */
    ka::Spawn<TaskFibo>() ( res1, n-1);
    ka::Spawn<TaskFibo>() ( res2, n-2 );

    /* the Sum task depends on res1 and res2 which are written by previous tasks
     * it must wait until thoses tasks are finished
     */
    ka::Spawn<TaskSum>(ka::SetLocal)  ( res, res1, res2 );
  }
}


/* Same as TaskFibo, but because a task may only have one implementation for each machine,
   we create a new task (new type).
*/
struct TaskFiboOpt : public ka::Task<2>::Signature<ka::W<int>, int > {};
template<>
struct TaskBodyCPU<TaskFiboOpt> : public TaskFibo {
  void operator() ( ka::Thread* thread, ka::pointer_w<int> res, int n )
  {  
    if (n < 2) {
      *res = fiboseq(n);
      kaapi_stack_print(1, kaapi_self_stack() );
      exit(1);
    }
    else {
      ka::pointer_rpwp<int> res1 = thread->Alloca<int>(1);
      ka::pointer_rpwp<int> res2 = thread->Alloca<int>(1);

      /* the Spawn keyword is used to spawn new task
       * new tasks are executed in parallel as long as dependencies are respected
       */
      thread->Spawn<TaskFiboOpt>() ( res1, n-1);
      thread->Spawn<TaskFiboOpt>() ( res2, n-2 );

      /* the Sum task depends on res1 and res2 which are written by previous tasks
       * it must wait until thoses tasks are finished
       */
      thread->Spawn<TaskSum>(ka::SetLocal)  ( res, res1, res2 );
    }
  }
};



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
      ka::pointer<int> res = ka::Alloca<int>(1);
      
      ka::Spawn<TaskFiboOpt>(ka::SetLocal)( res, n );

      /* ka::SetLocal ensures that the task is executed locally (cannot be stolen) */
      ka::Spawn<TaskPrint<int> >(ka::SetLocal)(res, "Nop", ref_value);
      
      /* */
      ka::Sync();
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


/* main entry point : Kaapi initialization
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

