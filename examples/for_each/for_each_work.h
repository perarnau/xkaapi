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
#ifndef FOR_EACH_WORK_H
#define FOR_EACH_WORK_H

#include "kaapi++"
#include <algorithm>
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
  Work()
   : _array(0)
  {}
  
  /* cstor */
  Work(T* beg, size_t size, OP op)
  {
    /* initialize work */
    _op    = op;
    _array = beg;
    kaapi_atomic_initlock( &_lock );
    kaapi_workqueue_init_with_lock(&_wq, 0, size, &_lock);
  }

  /* */
  ~Work()
  {
    kaapi_atomic_destroylock( &_lock );
  }
  
  /* extract sequential work */
  bool pop( T*& beg, T*& end)
  {
#define CONFIG_SEQ_GRAIN 8
    kaapi_workqueue_index_t b, e;
    if (kaapi_workqueue_pop(&_wq, &b, &e, CONFIG_SEQ_GRAIN)) return false;
    beg = _array+b;
    end = _array+e;
    return true;
  }
  
  /* name of the method should be splitter !!! split work and reply to requests */
  int split (
    ka::StealContext* sc, 
    int nreq, 
    ka::ListRequest::iterator beg,
    ka::ListRequest::iterator end
  );
  
  T* begin() { return _array + kaapi_workqueue_range_begin(&_wq); }
  T* end()   { return _array + kaapi_workqueue_range_end(&_wq); }
  OP& op()   { return _op; }

protected:
  /* extract parallel work for nreq. Return the unit size */
  bool helper_split( int& nreq, T*& beg, T*& end)
  {
    kaapi_atomic_lock( &_lock );
#define CONFIG_PAR_GRAIN 8
    kaapi_workqueue_index_t steal_size, i,j;
    kaapi_workqueue_index_t range_size = kaapi_workqueue_size(&_wq);
    if (range_size <= CONFIG_PAR_GRAIN)
    {
      kaapi_atomic_unlock( &_lock );
      return false;
    }

    steal_size = range_size * nreq / (nreq + 1);
    if (steal_size == 0)
    {
      nreq = (range_size / CONFIG_PAR_GRAIN) - 1;
      steal_size = nreq*CONFIG_PAR_GRAIN;
    }

    /* perform the actual steal. if the range
       changed size in between, redo the steal
     */
    if (kaapi_workqueue_steal(&_wq, &i, &j, steal_size))
    {
      kaapi_atomic_unlock( &_lock );
      return false;
    }
    kaapi_atomic_unlock( &_lock );
//    printf("Steal: [%li, %li)\n", i, j);
//    fflush(stdout);
    beg = _array + i;
    end = _array + j;
    return true;
  }
  
protected:
  OP                _op;
  T*                _array;
  kaapi_workqueue_t _wq;
  kaapi_lock_t      _lock;
};


/** Task for the thief
    CPU implementation: see different implementations
*/
template<typename T, typename OP>
struct TaskWork : public ka::Task<1>::Signature<ka::RW<Work<T,OP> > > {};


/* name of the method should be splitter !!! split work and reply to requests */
template<typename T, typename OP>
int Work<T,OP>::split (
  ka::StealContext* sc, 
  int nreq, 
  ka::ListRequest::iterator beg,
  ka::ListRequest::iterator end
)
{
  /* stolen range */
  T* beg_theft;
  T* end_theft;
  size_t size_theft;

  if (!helper_split( nreq, beg_theft, end_theft )) 
    return 0;
  size_theft = (end_theft-beg_theft)/nreq;

  /* thief work: create a task */
  for (; nreq>1; --nreq, ++beg, beg_theft+=size_theft)
  {
    beg->Spawn<TaskWork<T,OP> >(sc)( 
      new (*beg) Work<T,OP>( beg_theft, size_theft, _op)
    );
    beg->commit();
  }
  beg->Spawn<TaskWork<T,OP> >(sc)(
    new (*beg) Work<T,OP>( beg_theft, end_theft-beg_theft, _op)
  );
  beg->commit();
  ++beg;
  
  return 0;
}

#endif
