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
#include "kaapi++"
#include <algorithm>
#include <string.h>
#include <math.h>


/** Description of the example.

    Overview of the execution.
    
    What is shown in this example.
    
    Next example(s) to read.
*/


/* task signature */
template<typename T, typename OP>
struct TaskForEachTerminal : public ka::Task<3>::Signature<ka::RW<T>, ka::RW<T>, OP> {};

/* CPU implementation */
template<typename T, typename OP>
struct TaskBodyCPU<TaskForEachTerminal<T, OP> > {
  void operator() ( ka::pointer_rw<T> beg, ka::pointer_rw<T> end, OP op) 
  {
    std::for_each( beg, end, op );
  }
};

/* task signature */
template<typename T, typename OP>
struct TaskForEach : public ka::Task<3>::Signature<ka::RPWP<T>, ka::RPWP<T>, OP> {};

/* CPU implementation */
template<typename T, typename OP>
struct TaskBodyCPU<TaskForEach<T, OP> > {
  void operator() ( ka::pointer_rpwp<T> beg, ka::pointer_rpwp<T> end, OP op) 
  {
#define CONFIG_SEQ_GRAIN 256
    if (end-beg < CONFIG_SEQ_GRAIN)
      ka::Spawn<TaskForEachTerminal<T,OP> >()( beg, end, op );
    else {
      int med = (end-beg)/2;
      ka::Spawn<TaskForEach<T,OP> >()( beg, beg+med, op );
      ka::Spawn<TaskForEach<T,OP> >()( beg+med, end, op );
    }
  }
};


/* For each main function */
template<typename T, class OP>
void for_each( T* beg, T* end, OP op )
{
  ka::Spawn<TaskForEach<T,OP> >()(beg, end, op);

  /* here: wait all thieves have finish their result */
  ka::Sync();
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
