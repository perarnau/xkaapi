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
#include <stdint.h>
#include <sys/types.h>
#include <cuda.h>
#include "kaapi++.h"
#include "for_each_work.h"

// missing decls
typedef uintptr_t kaapi_mem_addr_t;
extern "C" void kaapi_mem_delete_host_mappings(kaapi_mem_addr_t, size_t);
typedef float double_type;


/** Description of the example.

    Overview of the execution.
    
    What is shown in this example.
    
    Next example(s) to read.
*/

__global__ void add1
(double_type* array, unsigned int first, unsigned int last)
{
  const unsigned int nelems = last - first;
  const unsigned int per_thread = nelems / blockDim.x;
  unsigned int i = first + threadIdx.x * per_thread;

  unsigned int j = last;
  if (threadIdx.x != (blockDim.x - 1)) j = i + per_thread;

  for (; i < j; ++i) ++array[i];
}

/** Simple thief task that only do sequential computation
*/
template<typename T, typename OP>
struct TaskBodyCPU<TaskThief<T, OP> > {
  void operator() ( ka::pointer_rw<T> beg, ka::pointer_rw<T> end, OP op)
  {
    std::for_each( beg, end, op );
  }
};

template<typename T, typename OP>
struct TaskBodyGPU<TaskThief<T, OP> > {
  void operator()
  (ka::gpuStream stream, ka::pointer_rw<T> beg, ka::pointer_rw<T> end, OP op)
  {
    const CUstream custream = (CUstream)stream.stream;

    const size_t size = (size_t)(end - beg);

    printf("cudaTask(0x%lx 0x%lx, %lu)\n",
	   (uintptr_t)custream, beg, size);
    fflush(stdout);

    add1<<<1, 256, 0, custream>>>(beg, 0, size);
  }
};

/* For each main function */
template<typename T, class OP>
static void for_each( T* beg, T* end, OP op )
{
  T* const base = beg;
  const size_t size = (size_t)(end - beg) * sizeof(T*);

  /* range to process */
  ka::StealContext* sc;
  Work<T,OP> work(beg, end, op);

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
  
  /* while there is sequential work to do*/
  while (work.extract_seq(beg, end))
  {
    /* apply w->op foreach item in [pos, end[ */
    std::for_each( beg, end, op );
  }
  
  /* wait for thieves */
  ka::TaskEndAdaptive(sc);
  /* here: 1/ all thieves have finish their result */

  kaapi_mem_delete_host_mappings((kaapi_mem_addr_t)base, size);
}


/**
*/
void apply_add1( double_type& v )
{
  v += 1;
}

/* My main task */
struct doit {
  void operator()(int argc, char** argv )
  {
    double t0,t1;
    double sum = 0.f;
    size_t size = 100000;
    if (argc >1) size = atoi(argv[1]);
    
    double_type* array = new double_type[size];

    for (int iter = 0; iter < 100; ++iter)
    {
      /* initialize, apply, check */
      for (size_t i = 0; i < size; ++i)
        array[i] = 0.f;
        
      t0 = kaapi_get_elapsedns();
      for_each( array, array+size, apply_add1 );
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
