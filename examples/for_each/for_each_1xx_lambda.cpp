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
#include <string.h>
#include <math.h>


/** Description of the example.

    Overview of the execution.
    
    What is shown in this example.
    
    Next example(s) to read.
*/
template<typename T, typename OP>
class Work {
public:
  /* cstor */
  Work(T* array, size_t size, OP op)
  {
    /* initialize work */
    _lock.write(0);
    _op    = op;
    _array = array;
    _beg   = 0;
    _end   = size;
  }

  /* extract sequential work */
  bool extract_seq( T*& beg, T*& end);

  /* extract parallel work */
  bool extract_par( T*& beg, T*& end);
  
protected:
  /* spinlocking */
  void lock()
  {
    while ( (_lock.read() == 1) || !_lock.cas(0, 1))
      ;
  }

  /* unlock */
  void unlock()
  { 
    _lock.write_barrier(0); 
  }

protected:
  ka::atomic_t<32> _lock;
  OP _op;
  T* _array;
  volatile size_t _beg;
  volatile size_t _end;
};



/** Task for the thief
*/
template<typename T, typename OP>
struct TaskThief : public ka::Task<3>::Signature<ka::RW<T>, ka::RW<T>, OP> {};

template<typename T, typename OP>
struct TaskBodyCPU<TaskThief<T, OP> > {
  void operator() ( ka::pointer_rw<T> first, ka::pointer_rw<T> last, OP op )
  {
    T* beg;
    T* end;
    Work<T,OP> work(first, last-first, op);

    /* while there is sequential work to do*/
    while (work.extract_seq(beg, end))
      /* apply w->op foreach item in [pos, end[ */
      std::for_each( beg, end, op );
  }
};




/** seq work extractor 
*/
template<typename T, typename OP>
bool Work<T,OP>::extract_seq(T*& beg, T*& end)
{
  /* extract from range beginning */
#define CONFIG_SEQ_GRAIN 64
  size_t seq_size = CONFIG_SEQ_GRAIN;

  size_t i, j;

  lock();
  i = _beg;
  if (seq_size > (_end - _beg))
    seq_size = _end - _beg;

  j = _beg + seq_size;
  _beg += seq_size;
  unlock();

  if (seq_size == 0)
    return false;

  beg = _array + i;
  end = _array + j;

  return true;
}


template<typename T, typename OP>
bool Work<T,OP>::extract_par(T*& beg_theft, T*& end_theft)
{
  /* concurrent with victim */
  lock();

  const size_t total_size = _end - _beg;

  /* how much per req */
# define CONFIG_PAR_GRAIN 128
  unit_size = 0;
  if (total_size > CONFIG_PAR_GRAIN)
  {
    unit_size = total_size / (nreq + 1);
    if (unit_size == 0)
    {
      nreq = (total_size / CONFIG_PAR_GRAIN) - 1;
      unit_size = CONFIG_PAR_GRAIN;
    }

    /* steal and update victim range */
    const size_t stolen_size = unit_size * nreq;
    beg_theft = _array + _beg - stolen_size;
    end_theft = _array + _end;
    _end -= stolen_size;
  }
  unlock();
}



/* For each main function */
template<typename T, class OP>
static void for_each( T* array, size_t size, OP op )
{
  /* range to process */
  ka::StealContext* sc;
  Work<T,OP> work(array, size, op);
  T* beg;
  T* end;

  /* push an adaptive task */
  sc = ka::TaskBeginAdaptive(
        /* flag: concurrent which means concurrence between extrac_seq & splitter executions */
          KAAPI_SC_CONCURRENT 
        /* flag: no preemption which means that not preemption will be available (few ressources) */
        | KAAPI_SC_NOPREEMPTION, 
        /* this lambda is implements the splitter */
        [&](                       /* standard arguments for the splitter function */
            ka::StealContext* sc,  /* the steal context */
            int nreq,              /* number of requests */
            ka::Request* req       /* array of request */
           )
          {
            /* stolen range */
            T* beg_theft;
            T* end_theft;
            
            if (!extract_par(beg_theft, end_theft)) return;

            for (; nreq >0; --nreq, ++req, end_theft -= unit_size)
              /* thief work: create a task */
              req->Spawn<TaskThief<T,OP> >(sc)( ka::pointer<T>(end_theft-unit_size), ka::pointer<T>(end_theft), _op );
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
void apply_sin( double& v )
{
  v = sin(v);
}


/* My main task */
struct doit {
  void operator()(int argc, char** argv )
  {
    size_t size = 10000;
    if (argc >1) size = atoi(argv[1]);
    
    double* array = new double[size];

    /* initialize, apply, check */
    memset(array, 0, sizeof(array));

    for_each( array, array+size, apply_sin );

    std::cout << "Done" << std::endl;
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
