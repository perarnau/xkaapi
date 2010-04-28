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
#include "kastl_workqueue.h"
#include <unistd.h>

namespace kastl {

namespace rts {

/**
*/
template<>
void work_queue_t<64>::lock_pop()
{
  while (!_lock.cas( 0, 1))
    kaapi_slowdown_cpu();
}

/**
*/
template<>
void work_queue_t<64>::lock_steal()
{
  while (!_lock.cas( 0, 1))
    kaapi_slowdown_cpu();
}

/**
*/
template<>
void work_queue_t<64>::unlock()
{
  kaapi_writemem_barrier();
  _lock.write(0);
}
  
/** */
template<>
bool work_queue_t<64>::slow_pop(range_t<64>& r, size_type size)
{
  /* already done in inlined pop :
      _beg += size;
      mem_synchronize();
    test (_beg > _end) was true.
    The real interval is [_beg-size, _end)
  */

  _beg -= size; /* abort transaction */

  kaapi_mem_barrier();

  lock_pop();

  r.first = _beg;

  if ((_beg + size) > _end)
  {
    size = _end - _beg;
    if (size == 0)
    {
      unlock();
      return false;
    }
  }

  _beg += size;

  unlock();

  r.last = _beg;

  return true;
}
  
  
/**
*/
template<>
bool work_queue_t<64>::steal(range_t<64>& r, size_type size)
{
  lock_steal();
  _end -= size;
  kaapi_mem_barrier();
  if (_end < _beg)
  {
    _end += size;
    unlock();
    return false;
  }
  r.first = _end;  
  unlock();
  r.last  = r.first + size;
  return true;
}  


/**
*/
template<>
bool work_queue_t<64>::steal(range_t<64>& r, size_type size_max, size_type size_min)
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
  unlock();
  r.last  = r.first + size_max;
  
  return true;
}  


/**
*/
template<>
bool work_queue_t<64>::steal_unsafe(range_t<64>& r, size_type size)
{
  _end -= size;
  kaapi_mem_barrier();
  if (_end < _beg)
  {
    _end += size;
    return false;
  }
  r.first = _end;  
  r.last  = r.first + size;
  return true;
}  


/**
*/
template<>
bool work_queue_t<64>::steal_unsafe(range_t<64>& r, size_type size_max, size_type size_min)
{
  kaapi_assert_debug( size_min <= size_max );
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
      return true;
    }
    _end += size_min; 
    return false;
  }
  
  r.first = _end;
  r.last  = r.first + size_max;
  
  return true;
}  




} /* impl namespace */

} /* kastl namespace */
