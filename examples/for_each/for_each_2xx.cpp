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

    Overview of the execution.
          The previous example, for_each_1xx.cpp has a main drawback: 
    - if the work load is very unbalanced, then the master thread may
    finish before thieves. Then it has choice 1/ to steal becomes to
    steal other thread or 2/ to preempt the thieves in order to get back
    computations.

    What is shown in this example.
      The purpose of this example is to show how to preempt thief in order
    to reduce. If the for_each algorithm does not required such preempt to
    be efficient, it is necessary for more complex algorithme such as prefix
    computation where the parallelism introduce overhead in the computation.
    
    Next example(s) to read.
*/


/** Same as for_each_1xx with preemption
*/
template<typename T, typename OP>
struct TaskBodyCPU<TaskThief<T, OP> > {
  void operator() ( ka::StealContext* sc, ka::pointer_rw<T> first, ka::pointer_rw<T> last, OP op )
  {
    Work<T,OP> work(first, last, op);
    T* beg;
    T* end;

    /* set the splitter for this task */
    sc->set_splitter(
        /* as for the initial work: set the method to call and the object */
        &ka::WrapperSplitter<Work<T,OP>,&Work<T,OP>::split>,
        &work
    );

    /* while there is sequential work to do*/
    while (work.extract_seq(beg, end))
    {
      /* apply w->op foreach item in [pos, end[ */
      std::for_each( beg, end, op );
      if (sc->is_preempted()) 
      {
        /* copy back the work state to the output state for the master */
        std::pair<T*, T*>& thief_context = sc->arg_preemption<std::pair<T*, T*> >();
        thief_context.first  = work.begin();
        thief_context.second = work.end();
        sc->ack_preemption();
        return;
      }
    }
  }
};


/* For each main function */
template<typename T, class OP>
static void for_each( T* first, T* last, OP op )
{
  /* range to process */
  ka::StealContext* sc;
  Work<T,OP> work(first, last, op);

  /* push an adaptive task */
  sc = ka::TaskBeginAdaptive(
        /* flag: concurrent which means concurrence between extrac_seq & splitter executions */
          KAAPI_SC_CONCURRENT 
        /* flag: no preemption which means that not preemption will be available (few ressources) */
        | KAAPI_SC_NOPREEMPTION, 
        /* use a wrapper to specify the method to used during parallel split */
        &ka::WrapperSplitter<Work<T,OP>,&Work<T,OP>::split>,
        &work
  );

redo_work:  
  /* while there is sequential work to do*/
  while (work.extract_seq(first, last))
    /* apply w->op foreach item in [pos, end[ */
    std::for_each( first, last, op );

  /* end of the sequential computation, preempt my thieves */
  ka::StealContext::thief_iterator beg = sc->begin_thief();
  ka::StealContext::thief_iterator end = sc->end_thief();
  
  while (beg != end)
  {
    if (beg->signal_preempt() == 0) 
    { /* send successed because thief was not finished */
      std::pair<T*, T*>* thief_context = beg->wait_preempt<std::pair<T*, T*> >();
      if (thief_context->first != thief_context->second) 
      {
        first = thief_context->first;
        last  = thief_context->second;
        goto redo_work;
      }
    } /* else send_preempt !=0 which means that this thief is finished */
    ++beg;
  }
  
  /* here : it is garanteed that all thieves and their thieves have finished */

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
  catch (const ka::Exception& E) {
    ka::logfile() << "Catch : " << E.what() << std::endl;
  }
  catch (...) {
    ka::logfile() << "Catch unknown exception: " << std::endl;
  }
  
  return 0;
}
