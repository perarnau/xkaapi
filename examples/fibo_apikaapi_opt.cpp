//! run as: karun -np 2 --threads 2 ./fibo_apiatha 36 4

/****************************************************************************
 * 
 *  Shared usage sample : fibonnaci
 *
 ***************************************************************************/

#include <iostream>
#include "kaapi++" // this is the new C++ interface for Kaapi


extern long cutoff;

// --------------------------------------------------------------------
/* Sequential fibo function
 */
long fiboseq( const long n ) {
  if( n<2 )
    return n;
  else
    return fiboseq(n-1)+fiboseq(n-2);
}

long fiboseq_On(const long n){
  if(n<2){
    return n;
  }else{

    long fibo=1;
    long fibo_p=1;
    long tmp=0;
    int i=0;
    for( i=0;i<n-2;i++){
      tmp = fibo+fibo_p;
      fibo_p=fibo;
      fibo=tmp;
    }
    return fibo;
  }
}


/* Sum two integers
 * this task reads a and b (read acces mode) and write their sum to res (write access mode)
 * it will wait until previous write to a and b are done
 * once finished, further read of res will be possible
 */
struct TaskSum : public ka::Task<3>::Signature<ka::W<long>, ka::R<long>, ka::R<long> > {};

template<>
struct TaskBodyCPU<TaskSum> : public TaskSum
{
  void operator() ( ka::pointer_w<long> res, 
                    ka::pointer_r<long> a, 
                    ka::pointer_r<long> b ) 
  {
    /* write is used to write data to a Shared_w
     * read is used to read data from a Shared_r
     */
    *res = *a + *b;
  }
};


/* Kaapi Fibo task.
   A Task is a type with respect a given signature. The signature specifies the number of arguments (2),
   and the type and access mode for each parameters.
   Here the first parameter is declared with a write mode. The second is passed by value.
 */
struct TaskFibo : public ka::Task<2>::Signature<ka::W<long>, const long > {};


/* Implementation for CPU machine 
*/
template<>
struct TaskBodyCPU<TaskFibo> : public TaskFibo {
  void operator() ( ka::Thread* thread, ka::pointer_w<long> res, const long n )
  {  
    if (n < 2){ //cutoff) {
      *res = n; //fiboseq(n);
    }
    else {
      ka::pointer_rpwp<long> res1 = thread->Alloca<long>(1);
      ka::pointer_rpwp<long> res2 = thread->Alloca<long>(1);

      /* the Spawn keyword is used to spawn new task
       * new tasks are executed in parallel as long as dependencies are respected
       */
      thread->Spawn<TaskFibo>() ( res1, n-1);
      thread->Spawn<TaskFibo>() ( res2, n-2 );

      /* the Sum task depends on res1 and res2 which are written by previous tasks
       * it must wait until thoses tasks are finished
       */
      thread->Spawn<TaskSum>() ( res, res1, res2 );
    }
  }
};

__attribute__((constructor)) void InitLib()
{
}
