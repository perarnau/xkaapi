/*
 ** xkaapi
 ** 
 ** Created on Tue Mar 31 15:19:14 2009
 ** Copyright 2009 INRIA.
 **
 ** Contributors :
 **
 ** thierry.gautier@inrialpes.fr
 ** fabien.lementec@gmail.com / fabien.lementec@imag.fr
 
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
#include "kastl/kastl_workqueue.h"
#include <unistd.h>

namespace kastl {

namespace impl {

/**
*/
void work_queue::lock_pop()
{
  while (!atomic_cas(&_lock, 0, 1))
    ;
}

/**
*/
void work_queue::lock_steal()
{
  while ((atomic_read(&_lock) == 1) && !atomic_cas(&_lock, 0, 1))
    usleep(1);
}

/**
*/
void work_queue::unlock()
{
  atomic_write(&_lock, 0);
}
  
/** */
bool work_queue::slow_pop(range& r, work_queue::size_type size)
{
  /* already done in inlined pop :
      _beg += size;
      mem_synchronize();
    test (_beg > _end) was true.
    The real interval is [_beg-size, _end)
  */
  r.first = _end; /* upper bound of what could be _end (always decreasing) */
  lock_pop();
  if (_beg > _end)
  {
    if (_beg - size >= _end)
    {
      _beg -= size;
      unlock();
      return false;
    }
    r.last = _end;
    size -= _beg - r.last;
    _beg = r.last;
  }
  else r.last = _beg;
  unlock();

  /* */
  r.first = r.last - size;
  if (r.first <0) r.first = 0;

  return true;
}
  
  
/**
*/
bool work_queue::steal(range& r, work_queue::size_type size)
{
  lock_steal();
  _end -= size;
  kaapi_mem_barrier();
//  mem_synchronize();
  if (_end < _beg)
  {
    _end += size;
    unlock();
    return false;
  }
  
  r.first = _end;
  r.last  = r.first + size;
  
  unlock();
  
  return true;
}  


/**
*/
bool work_queue::steal(range& r, work_queue::size_type size_max, work_queue::size_type size_min)
{
  kaapi_assert_debug( size_min <= size_max );
  lock_steal();
  _end -= size_max;
  kaapi_mem_barrier();
  if (_end < _beg)
  {
    _end += size_max - size_min;
    kaapi_mem_barrier();
    if (_beg < _end)
    {
      r.first = _end;
      r.last  = r.first+size_min;
      unlock();
      return true;
    }
    _end += size_min; 
    unlock();
    return false;
  }
  
  r.first = _end;
  r.last  = r.first + size_max;
  
  unlock();
  return true;
}  

} /* impl namespace */

} /* kastl namespace */
