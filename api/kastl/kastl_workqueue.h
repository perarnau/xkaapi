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
#ifndef _KASTL_WORK_QUEUE_H_
#define _KASTL_WORK_QUEUE_H_
#include "kaapi.h"

namespace kastl {
  
namespace impl {

  /* atomics, low level memory routines */

  typedef struct atomic_t
  {
    volatile long _counter;

    inline atomic_t(long value)
      : _counter(value) {}

  } atomic_t;

  static inline long atomic_sub(atomic_t* a, long value)
  {
    return __sync_sub_and_fetch(&a->_counter, value);
  }

  static inline long atomic_add(atomic_t* a, long value)
  {
    return __sync_add_and_fetch(&a->_counter, value);
  }

  static inline int atomic_cas(atomic_t* a, long o, long n)
  {
    return __sync_bool_compare_and_swap(&a->_counter, o, n);
  }

  static inline long atomic_read(atomic_t* a)
  {
    return a->_counter;
  }

  static inline void atomic_write(atomic_t* a, long value)
  {
    a->_counter = value;
  }

  static inline void mem_synchronize()
  {
    /* this is needed */
    __sync_synchronize();
  }

  /** work range
   */
  struct range
  {
    // must be signed int
    typedef long index_type;
    
    index_type first;
    index_type last;
    
    range()
#if defined(KAAPI_DEBUG)    
     : first(0), last(0)
#endif
    { }

    range( index_type f, index_type l )
     : first(f), last(l)
    { }
    
    index_type size() const
    {
      return last - first;
    }

    bool is_empty() const
    {
      return !(first < last);
    }
  };
  
  /** work_queue index
   */
  typedef long volatile work_queue_index_t;
  
  /** work work_queue
   */
  class work_queue {
  public:
    typedef range::index_type size_type;
    typedef range::index_type index_type;
    
    /* default cstor */
    work_queue();
    
    /* set the work_queue */
    void set( const range& );
    
    /* clear the work_queue 
     */
    inline void clear();
    
    /* return true iff the work_queue is empty 
     */
    inline bool is_empty() const;
    
    /* return the size of the work_queue 
     */
    inline size_type size() const;
    
    /* pop a subrange of size at most sz 
     return true in case of success
     */
    bool pop(range&, size_type sz);
    
    /* steal a subrange of size at most sz
       return true in case of success
     */
    bool steal(range&, size_type sz);
    
    /* steal a subrange of size at most sz_max
       return true in case of success
     */
    bool steal(range&, size_type sz_max, size_type sz_min);
    
  private:
    /* */
    bool slow_pop( range&, size_type sz );
    
    /* */
    void lock_pop();

    /* */
    void lock_steal();
    
    /* */
    void unlock();
    
    /* data structure that required to be correctly aligned in order to
     ensure atomicity of read/write.
     Currently only IA32 & x86-64
     */
    work_queue_index_t _beg __attribute__((aligned(64)));
    work_queue_index_t _end __attribute__((aligned(64)));

    atomic_t _lock;
  };
  
  /** */
  inline work_queue::work_queue()
  : _beg(0), _end(0), _lock(0)
  {}
  
  /** */
  inline void work_queue::set( const range& r)
  {
    /* optim: only lock for setting _end
       after having set _beg to _end
     */

    lock_pop();
    _beg = r.first;
    _end = r.last;
    unlock();
  }
  
  /** */
  inline void work_queue::clear()
  {
    _end = 0;
    _beg = 0; 
  }
  
  /** */
  inline work_queue::size_type work_queue::size() const
  {
    return _end - _beg;
  }
  
  /** */
  inline bool work_queue::is_empty() const
  {
    return _end <= _beg;
  }
  
  /** */
  inline bool work_queue::pop(range& r, work_queue::size_type size)
  {
    _beg += size;
    mem_synchronize();
    if (_beg > _end) return slow_pop( r, size );
    
    r.first = _beg - size;
    r.last = _beg;
    
    // check for boundaries
    kaapi_assert_debug( (_beg >=0) && (_beg <= _end) );
    kaapi_assert_debug( (r.first >=0) && (r.first <= r.last) );
    
    return true;
  }

} /* impl namespace */

} /* kastl namespace */

#endif
