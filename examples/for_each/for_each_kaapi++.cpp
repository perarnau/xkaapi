/*
** xkaapi
** 
**
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
    
    What is shown in this example.
    
    Next example(s) to read.
*/


/** Simple task that only do sequential computation
*/
template<typename T, typename OP>
struct TaskBodyCPU<TaskWork<T, OP> > {
  void operator() ( ka::pointer_rw<Work<T,OP> > work )
  {
    T* beg; 
    T* end;
    
    /* while there is sequential work to do*/
    while (work->pop(beg, end))
    {
      /* apply w->op foreach item in [pos, end[ */
      std::for_each( beg, end, work->op() );
    }
  }
};

/** Splitter for CPU implementation, call in concurrency with 
    TaskBodyCPU<TaskWork<T, OP> >
*/
template<typename T, typename OP>
struct TaskSplitter<TaskWork<T, OP> > {
  int operator() ( ka::StealContext* sc, 
                   int nreq, 
                   ka::ListRequest::iterator begin, 
                   ka::ListRequest::iterator end,
                   ka::pointer_rw<Work<T,OP> > work
                  ) 
  {
    return work->split(sc, nreq, begin, end );
  }
};



/* For each main function */
template<typename T, class OP>
static void for_each( T* beg, T* end, OP op )
{
  /* range to process */
  Work<T,OP> work(beg, end-beg, op);
  ka::Spawn<TaskWork<T,OP> >()(&work);
  ka::Sync();
}


/**
*/
void apply_cos( double& v )
{
  v += cos(v);
  usleep(100);
}

/* My main task */
struct doit {
  void operator()(int argc, char** argv )
  {
    double t0,t1;
    double sum = 0.f;
    size_t size = 30000;
    if (argc >1) size = atoi(argv[1]);
    
    double* array = new double[size];
baseaddr = array;    

    for (int iter = 0; iter < 1; ++iter)
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
