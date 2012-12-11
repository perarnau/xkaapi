/*
** xkaapi
** 
**
** Copyright 2009,2010,2011,2012 INRIA.
**
** Contributors :
**
** thierry.gautier@inrialpes.fr
** 
** This software is a computer program whose purpose is to execute
** multithreaded computation with data flow synchronization between
** threads.
** 
** This software is governed by the CeCILL-C license under French law
** and abiding by the rules of distribution of free software.  You can
** use, modify and/ or redistribute the software under the terms of
** the CeCILL-C license as circulated by CEA, CNRS and INRIA at the
** following URL "http://www.cecill.info".
** 
** As a counterpart to the access to the source code and rights to
** copy, modify and redistribute granted by the license, users are
** provided only with a limited warranty and the software's author,
** the holder of the economic rights, and the successive licensors
** have only limited liability.
** 
** In this respect, the user's attention is drawn to the risks
** associated with loading, using, modifying and/or developing or
** reproducing the software by the user in light of its specific
** status of free software, that may mean that it is complicated to
** manipulate, and that also therefore means that it is reserved for
** developers and experienced professionals having in-depth computer
** knowledge. Users are therefore encouraged to load and test the
** software's suitability as regards their requirements in conditions
** enabling the security of their systems and/or data to be ensured
** and, more generally, to use and operate it in the same conditions
** as regards security.
** 
** The fact that you are presently reading this means that you have
** had knowledge of the CeCILL-C license and that you accept its
** terms.
** 
*/
#include <iostream>
#include "kaapi++" // this is the new C++ interface for Kaapi
#include "test_main.h"
#include "fib_verify.h"

kaapi_atomic_t counter_task;

/* Sum two integers
 * this task reads a and b (read acces mode) and write their sum to res (write access mode)
 * it will wait until previous write to a and b are done
 * once finished, further read of res will be possible
 */
struct TaskSum : public ka::Task<3>::Signature<ka::W<long>, ka::R<long>, ka::R<long> > {};

template<>
struct TaskBodyCPU<TaskSum>
{
  void operator() ( ka::pointer_w<long> r, 
                    ka::pointer_r<long> a, 
                    ka::pointer_r<long> b ) 
  {
    /* write is used to write data to a Shared_w
     * read is used to read data from a Shared_r
     */
    *r = *a + *b;
  }
};


/* Delete heap allocated data
 * this task delete the data allocated during recursive computation of Fibonacci number
 */
struct TaskDelete : public ka::Task<1>::Signature<ka::RW<long> > {};

template<>
struct TaskBodyCPU<TaskDelete>
{
  void operator() ( ka::pointer_rw<long> ptr ) 
  {
    delete ptr;
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
struct TaskBodyCPU<TaskFibo>
{
  void operator() ( ka::pointer_w<long> ptr, const long n )
  {  
    if (n < 2){ 
      *ptr = n; 
      return;
    }
    else {
      long* otr1 = new long;
      long* otr2 = new long;
      ka::pointer<long> ptr1 = otr1;
      ka::pointer<long> ptr2 = otr2;
//printf("New @:%p\n", (long*)otr1); printf("New @:%p\n", (long*)otr2); fflush(stdout);

      /* the Spawn keyword is used to spawn new task
       * new tasks are executed in parallel as long as dependencies are respected
       */
      KAAPI_ATOMIC_INCR(&counter_task);
      ka::Spawn<TaskFibo>() ( ptr1, n-1 );
      KAAPI_ATOMIC_INCR(&counter_task);
      ka::Spawn<TaskFibo>() ( ptr2, n-2 );

      /* the Sum task depends on res1 and res2 which are written by previous tasks
       * it must wait until thoses tasks are finished
       */
      KAAPI_ATOMIC_INCR(&counter_task);
      ka::Spawn<TaskSum>() ( ptr, ptr1, ptr2 );
      
      /* spawn tasks to delete data 
       */
      KAAPI_ATOMIC_INCR(&counter_task);
      ka::Spawn<TaskDelete>()( ptr1 );
      KAAPI_ATOMIC_INCR(&counter_task);
      ka::Spawn<TaskDelete>()( ptr2 );
    }
  }
};


/* Main of the program
*/
void doit::operator()(int argc, char** argv )
{
  unsigned int n = 30;
  if (argc > 1) n = atoi(argv[1]);
  
  double start_time;
  double stop_time;

  long res_value = 0;
  res_value = rand();
  ka::pointer<long> res = &res_value;

  start_time= ka::WallTimer::gettime();
  KAAPI_ATOMIC_INCR(&counter_task);
  ka::Spawn<TaskFibo>()( res, n );
  ka::Sync();
  stop_time= ka::WallTimer::gettime();

  kaapi_assert( res_value == fiboseq_verify(n) );
  
  
  kaapi_perf_idset_t idset;
  kaapi_perf_counter_t internal_counter;
  kaapi_perf_idset_zero( &idset );
  kaapi_perf_idset_add(&idset, KAAPI_PERF_ID_TASKS );
  kaapi_perf_read_counters( &idset, &internal_counter );
  
  std::cout << ": -----------------------------------------" << std::endl;
  std::cout << ": Res  = " << res_value << std::endl;
  std::cout << ": Time(s): " << (stop_time-start_time) << std::endl;
  std::cout << ": #task  : " << KAAPI_ATOMIC_READ(&counter_task) << std::endl;
  std::cout << ": #internal task counter:" << internal_counter << ", " << kaapi_perf_id_to_name(KAAPI_PERF_ID_TASKS) << std::endl;
  std::cout << ": -----------------------------------------" << std::endl;
}

