/*
** xkaapi
** 
**
** Copyright 2009 INRIA.
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

int cutoff;

long long fib_seq (int n)
{
	int x, y;
	if (n < 2) return n;

	x = fib_seq(n - 1);
	y = fib_seq(n - 2);

	return x + y;
}



/* Kaapi Fibo task.
   A Task is a type with respect a given signature. The signature specifies the number of arguments (2),
   and the type and access mode for each parameters.
   Here the first parameter is declared with a write mode. The second is passed by value.
 */
struct TaskFibo : public ka::Task<3>::Signature<ka::W<long long>, int, int > {};


/* Implementation for CPU machine 
*/
template<>
struct TaskBodyCPU<TaskFibo>
{
  void operator() ( ka::pointer_w<long long> ptr, int n, int d )
  {  
    long long x, y;

    if (n < 2) 
    {
      *ptr = n;
      return;
    }

    if ( d < cutoff ) 
    {
      ka::Spawn<TaskFibo>()( &x, n-1, d+1 );
      ka::Spawn<TaskFibo>()( &y, n-2, d+1 );
      ka::Sync();

    } else {
      x = fib_seq(n-1);
      y = fib_seq(n-2);
    }

    *ptr = x + y;
  }
};


/* Main of the program
*/
struct doit {

  void do_experiment(unsigned int n, unsigned int iter )
  {
    double start_time;
    double stop_time;

    long res_value = 0;
    ka::pointer<long> res = &res_value;
    for (int i=0; i<1; ++i)
    {
      res_value = 0;
      ka::Spawn<TaskFibo>()( res, n, 0 );
      /* */
      ka::Sync();
      start_time= ka::WallTimer::gettime();
      for (unsigned int i = 0 ; i < iter ; ++i)
      {   
        res_value = 0;
        ka::Spawn<TaskFibo>()( res, n, 0 );
        /* */
        ka::Sync();
      }
      stop_time= ka::WallTimer::gettime();

      ka::logfile() << ": -----------------------------------------" << std::endl;
      ka::logfile() << ": Res  = " << res_value << std::endl;
      ka::logfile() << ": Time(s): " << (stop_time-start_time)/iter << std::endl;
      ka::logfile() << ": -----------------------------------------" << std::endl;
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
  catch (const std::exception& E) {
    ka::logfile() << "Catch : " << E.what() << std::endl;
  }
  catch (...) {
    ka::logfile() << "Catch unknown exception: " << std::endl;
  }
  
  return 0;
}
