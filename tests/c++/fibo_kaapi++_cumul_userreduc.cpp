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
#include "fib_verify.h"
#include "test_main.h"


struct MyReductionOperator {
  void operator()( long& result, const long& value)
  { 
    result += value; 
  }
};


/* Kaapi Fibo task.
   A Task is a type with respect a given signature. The signature specifies the number of arguments (2),
   and the type and access mode for each parameters.
   Here the first parameter is declared with a write mode. The second is passed by value.
 */
struct TaskFibo : public ka::Task<2>::Signature<ka::CW<long,MyReductionOperator>, const long > {};


/* Implementation for CPU machine 
*/
template<>
struct TaskBodyCPU<TaskFibo>
{
  /* explicit global reduction: MyReductionOperator */
  void operator() ( ka::pointer_cw<long,MyReductionOperator> res, const long n )
  {  
    if (n < 2){ 
      *res += n; 
      return;
    }
    else {
      /* the Spawn keyword is used to spawn new task
       * new tasks are executed in parallel as long as dependencies are respected
       */
      ka::Spawn<TaskFibo>() ( res, n-1 );
      ka::Spawn<TaskFibo>() ( res, n-2 );
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
  ka::pointer<long> res = &res_value;

  start_time= ka::WallTimer::gettime();
  ka::Spawn<TaskFibo>()( res, n );
  ka::Sync();
  stop_time= ka::WallTimer::gettime();

  kaapi_assert( res_value == fiboseq_verify(n) );
  
  ka::logfile() << ": -----------------------------------------" << std::endl;
  ka::logfile() << ": Res  = " << res_value << std::endl;
  ka::logfile() << ": Time(s): " << (stop_time-start_time) << std::endl;
  ka::logfile() << ": -----------------------------------------" << std::endl;
}
