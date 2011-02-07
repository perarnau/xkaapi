/*
** xkaapi
** 
** Created on Tue Mar 31 15:19:14 2009
** Copyright 2009 INRIA.
**
** Contributors :
** thierry.gautier@inrialpes.fr
** fabien.lementec@imag.fr
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
#include "for_each_work.h"

/** Description of the example.

    This example is the same as for_each_0xx.cpp excepted the use of C++ lambda 
    expression to define the splitter.

    Overview of the execution.
    
    What is shown in this example.
    
    Next example(s) to read.
*/

/** Simple thief task that only do sequential computation
*/
template<typename T, typename OP>
struct TaskBodyCPU<TaskThief<T, OP> > {
  void operator() ( ka::pointer_rw<T> beg, ka::pointer_rw<T> end, OP op)
  {
    std::for_each( beg, end, op );
  }
};


/* For each main function */
template<typename T, class OP>
static void for_each( T* ibeg, T* iend, OP op )
{
  /* range to process */
  ka::StealContext* sc;
  Work<T,OP> work(ibeg, iend, op);
  T* beg;
  T* end;

  /* push an adaptive task */
  sc = ka::TaskBeginAdaptive(
        /* flag: concurrent which means concurrence between extrac_seq & splitter executions */
          KAAPI_SC_CONCURRENT 
        /* flag: no preemption which means that not preemption will be available (few ressources) */
        | KAAPI_SC_NOPREEMPTION, 

        /* This lambda implements the splitter.
           In this version, we do not use the method Work::split which is defined here as a lambda.
        */
        [&](                       /* standard arguments for the splitter function */
            ka::StealContext* sc,  /* the steal context */
            int nreq,              /* number of requests */
            ka::Request* req       /* array of request */
           ) -> void
          {
            /* stolen range */
            T* beg_theft;
            T* end_theft;
            size_t size_theft;

            if (!work.extract_par( nreq, beg_theft, end_theft )) return;
            size_theft = (end_theft-beg_theft)/nreq;
            
            /* thief work: create a task */
            for (; nreq>1; --nreq, ++req, beg_theft+=size_theft)
              req->Spawn<TaskThief<T,OP> >(sc)( ka::pointer<T>(beg_theft), ka::pointer<T>(beg_theft+size_theft), op );
            req->Spawn<TaskThief<T,OP> >(sc)( ka::pointer<T>(beg_theft), ka::pointer<T>(end_theft), op );
          }
  );

  /* while there is sequential work to do*/
  while (work.extract_seq(beg, end))
    /* apply w->op foreach item in [pos, end[ */
    std::for_each( beg, end, op );

  /* wait for thieves */
  ka::TaskEndAdaptive(sc);

  /* here: 1/ all thieves have finish their result */
}


/**
*/
void apply_cos( double& v )
{
  v = cos(v);
}

/* My main task */
struct doit {
  void operator()(int argc, char** argv )
  {
    double t0,t1;
    double sum = 0.f;
    size_t size = 100000;
    if (argc >1) size = atoi(argv[1]);
    
    double* array = new double[size];

    for (int iter = 0; iter < 100; ++iter)
    {
      /* initialize, apply, check */
      for (size_t i = 0; i < size; ++i)
        array[i] = 0.f;
        
      t0 = kaapi_get_elapsedns();
      for_each( array, array+size, apply_cos );
      t1 = kaapi_get_elapsedns();
      sum += (t1-t0)/1000; /* ms */

      for (size_t i = 0; i < size; ++i)
        if (array[i] != 1.f)
        {
          std::cout << "invalid @" << i << " == " << array[i] << std::endl;
          break ;
        }
    }

    std::cout << "Done " << sum/100 << " (ms)" << std::endl;
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
