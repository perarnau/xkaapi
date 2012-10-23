/*
 ** xkaapi
 ** 
 **
 ** Copyright 2009,2010,2011,2012 INRIA.
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
#include "kastl/kastl_types.h"
#include <limits>


namespace kastl {
  
/* --- most of these class / structure should be put into the C API
   * the work_queue_t structure is more general that its usage in KaSTL
   it could be used inside the C runtime to manage the steal of task/frame.
*/  
namespace rts {

  /** work range_t: two public available integer of 'bits' bits
   */
  template<int bits>
  struct range_t
  {
    // must be signed int
    typedef typename signed_type_that_have_bits<bits>::type index_type;
    typedef typename range_t<bits>::index_type              size_type;
    
    index_type first;
    index_type last;
    
    /* */
    range_t()
#if defined(KAAPI_DEBUG)    
     : first(0), last(0)
#endif
    { }

    /* */
    range_t( index_type f, index_type l )
     : first(f), last(l)
    { }
    
    /* */
    void clear()
    { first = last = 0; }

    /* */
    index_type size() const
    {
      if (first < last) return last - first;
      return 0;
    }

    /* */
    bool is_empty() const
    {
      return !(first < last);
    }
    
    /* shift the range_t to put the first at orig */
    void shift( range_t::index_type orig )
    {
      last  -= first-orig;
      first = orig;
    }
  };

  
  /** work work_queue_t: the main important data structure.
      It steal/pop are managed by a Dijkstra like protocol.
      The threads that want to steal serialize their access
      through a lock.
   */
  template<int bits>
  class work_queue_t {
  public:
    typedef typename range_t<bits>::size_type  size_type;
    typedef typename range_t<bits>::index_type index_type;
    typedef range_t<bits>                      range_type;
    
    /* default cstor */
    work_queue_t();
    
    /* cstor */
    work_queue_t( const range_t<bits>& r );
    
    /* set the work_queue_t */
    void set( const range_t<bits>& );

    /* clear the work_queue_t 
     */
    void clear();
    
    /* return true iff the work_queue_t is empty 
     */
    bool is_empty() const;
    
    /* return the size of the work_queue_t 
     */
    size_type size() const;
    
    /* push a new valid subrange_t at the begin of the queue.
       Only valid of the pushed range_t just before the remainding part of the queue,
       i.e. iff queue.beg == range_t.last.
       If the range_t is valid then returns true else return false.
       The method is concurrent with the steal's methods.
     */
    bool push_front( const range_t<bits>& r );
    
    /* extend the queue from [_beg,_end) to [first, _end)
       Only valid of the first < _beg.
       If its valid then returns true else return false.
       The method is concurrent with the steal's methods.
     */
    bool push_front( const index_type& first );

    /* pop one element
       return true in case of success
     */
    bool pop(index_type&);
    
    /* pop a subrange_t of size at most sz.
       Return true in case of success. 
       The poped range may be of size less than sz, which means that the queue is empty
       and the next pop will return false.
    */
    bool pop(range_t<bits>&, size_type sz_max);

    /* pop a subrange_t of size at most sz.
       Return true in case of success. 
       The poped range may be of size less than sz, which means that the queue is empty
       and the next pop will return false.
    */
    bool pop_safe(range_t<bits>&, size_type sz_max);


    /* push_back a new valid subrange_t at the end of the queue.
       Only valid of the pushed range_t just after the queue,
       i.e. iff queue.end == range_t.first.
       If the range_t is valid then returns true else return false.
       The method is concurrent with the pop's method.
     */
    bool push_back( const range_t<bits>& r );
    
    /* extend the queue from [_beg,_end) to [_beg, last)
       Only valid of the last > _end.
       If its valid then returns true else return false.
       The method is concurrent with the pop's method.
     */
    bool push_back( const index_type& last );
    
    /* steal a subrange_t of size at most sz_max
       The method locks the mutex on the queue to serialize concurrent steal execution and try to steal work. 
       The method is safe with concurrent execution of pop's kind of methods.
       Return true in case of success else false.
     */
    bool steal(range_t<bits>&, size_type sz_max );

    /* steal a subrange_t of size at most sz_max
       The method DOES NOT lock the mutex on the queue to serialize concurrent steal execution before trying to steal work. 
       The method is UNSAFE with concurrent execution of pop's kind of methods. The caller is responsible to
       ensure no concurrency of steal operation on the queue, for instance by locking the queue by a call to the lock_steal() 
       method.
       Return true in case of success else false.
     */
    bool steal_unsafe(range_t<bits>&, size_type sz_max );
    
    /* */
    bool slow_pop( range_t<bits>&, size_type sz );
    
    /* */
    void lock_pop();

    /* */
    void lock_steal();
    
    /* */
    void unlock();
    
    /* */
    index_type begin() const { return _beg; }

    /* */
    index_type end() const { return _end; }

    /* data field required to be correctly aligned in order to ensure atomicity of read/write. 
       Put them on two separate lines of cache (assume == 64bytes) due to different access by 
       concurrent threads.
       Currently only IA32 & x86-64.
       An assertion is put inside the constructor to verify that this field are correctly aligned.
     */
    index_type volatile _beg; /*_beg & _end on two cache lines */
    index_type volatile _end __attribute__((aligned(64))); /* minimal constraints for _end / _beg _lock and _end on same cache line */

    atomic_t<32> _lock __attribute__((aligned));       /* one bit is enough .. */
  };
  
  /** */
  template<int bits>
  inline work_queue_t<bits>::work_queue_t()
  : _beg(0), _end(0), _lock(0)
  {
#if defined(__i386__)||defined(__x86_64)
    kaapi_assert_debug( (((unsigned long)&_beg) & (bits/8-1)) == 0 ); 
    kaapi_assert_debug( (((unsigned long)&_end) & (bits/8-1)) == 0 );
#else
#  warning "May be alignment constraints exit to garantee atomic read write"
#endif
  }
  
  /** */
  template<int bits>
  inline work_queue_t<bits>::work_queue_t( const range_t<bits>& r )
  : _beg(r.first), _end(r.last), _lock(0)
  {
  }

  /** */
  template<int bits>
  inline void work_queue_t<bits>::set( const range_t<bits>& r)
  {
    /* try to order writes to always guarantee that 
       - if the queue not empty before calling the method and 
       if the range_t is not empty
       - then the queue is empty during the execution
       - then the queue is not empty on return
    */
    lock_pop();
    /* no reorder over volatile variable */
    _end = std::numeric_limits<typename work_queue_t<bits>::index_type>::min();
    _beg = r.first;
    _end = r.last;
    unlock();
  }

  /** */
  template<int bits>
  inline void work_queue_t<bits>::clear()
  {
    lock_pop();
    _end = 0;
    _beg = 0; 
    unlock();
  }
  
  /** */
  template<int bits>
  inline typename work_queue_t<bits>::size_type work_queue_t<bits>::size() const
  {
    /* on lit d'abord _end avant _beg afin que le voleur puisse qui a besoin
       en general d'avoir la taille puis avoir la valeur la + ajour possible...
    */
    index_type e = _end;
    kaapi_readmem_barrier();
    index_type b = _beg;
    return b < e ? e-b : 0;
  }
  
  /** */
  template<int bits>
  inline bool work_queue_t<bits>::is_empty() const
  {
    /* inverse ici... critical path optimization ? on veut la valeur la plus
       a jour possible pour la victime (pop)
    */
    index_type b = _beg;
    kaapi_readmem_barrier();
    index_type e = _end;
    return e <= b;
  }

  /** */
  template<int bits>
  inline bool work_queue_t<bits>::push_front( const range_t<bits>& r )
  {
    kaapi_assert_debug( !r.is_empty() ) ;
    if (r.last != _beg ) return false;
    kaapi_writemem_barrier();
    _beg = r.first;
    return true;
  }
  
  /** */
  template<int bits>
  inline bool work_queue_t<bits>::push_front( const index_type& first )
  {
    if (first > _beg ) return false;
    kaapi_writemem_barrier();
    _beg = first;
    return true;
  }


  /** */
  template<int bits>
  inline bool work_queue_t<bits>::pop( index_type& item )
  {
    item = _beg;
    ++_beg;
    /* read of _end after write of _beg */
    kaapi_mem_barrier(); 
    if (_beg > _end) 
    {
      range_t<bits> r;
      bool retval = slow_pop( r, 1 );
      item = r.first;
      return retval;
    }
    
    return true;
  }


  /** */
  template<int bits>
  inline bool work_queue_t<bits>::pop(range_t<bits>& r, size_type size)
  {
    _beg += size;
    /* read of _end after write of _beg */
    kaapi_mem_barrier();
    if (_beg > _end) {
      return slow_pop( r, size );
    }
    
    r.last = _beg;
    r.first = r.last - size;
    
    return true;
  }


  /** */
  template<int bits>
  inline bool work_queue_t<bits>::pop_safe(range_t<bits>& r, size_type size)
  {
    if (_end <=_beg) return false;
    if (_end - _beg < size)
      size = _end - _beg;
    _beg += size;
    r.last = _beg;
    r.first = r.last - size;
    return true;
  }


  /** */
  template<int bits>
  inline bool work_queue_t<bits>::push_back( const range_t<bits>& r )
  {
    kaapi_assert_debug( !r.is_empty() ) ;
    if (r.first != _end ) return false;
    kaapi_writemem_barrier();
    _end = r.last;
    return true;
  }


  /** */
  template<int bits>
  inline bool work_queue_t<bits>::push_back( const index_type& last )
  {
    if (last < _end ) return false;
    kaapi_writemem_barrier();
    _end = last;
    return true;
  }

  /** */
  template<int bits>
  inline void work_queue_t<bits>::lock_pop()
  {
    while (!_lock.cas(0, 1))
      kaapi_slowdown_cpu();
  }

  /** */
  template<int bits>
  inline void work_queue_t<bits>::lock_steal()
  {
    while (!_lock.cas(0, 1))
      kaapi_slowdown_cpu();
  }

  /** */
  template<int bits>
  inline void work_queue_t<bits>::unlock()
  {
    kaapi_writemem_barrier();
    _lock.write(0);
  }
    
  /** */
  template<int bits>
  bool work_queue_t<bits>::slow_pop(range_t<bits>& r, size_type size_max)
  {
    /* already done in inlined pop :
       _beg += size_max;
       mem_synchronize();
       test (_beg > _end) was true.
       The real interval is [_beg-size_max, _end)
    */
    _beg -= size_max; /* abort transaction */
    kaapi_mem_barrier();
    lock_pop();
    
    r.first = _beg;
    
    if ((_beg + size_max) > _end)
    {
      size_max = _end - _beg;
      if (size_max == 0)
      {
        unlock();
        return false;
      }
    }
    
    _beg += size_max;
    
    unlock();
    
    r.last = _beg;
    
    return true;
  }
  
#if 0 /* deprecated */
  /** */
  template<int bits>
  bool work_queue_t<bits>::steal(range_t<bits>& r, size_type size)
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
#endif

  /**
   */
  template<>
  bool work_queue_t<64>::steal(range_t<64>& r, size_type size_max)
  {
    kaapi_assert_debug( 1 <= size_max );
    lock_steal();
    _end -= size_max;
    kaapi_mem_barrier();
    if (_end < _beg)
    {
      _end += size_max - 1;
      kaapi_mem_barrier();
      if (_beg < _end)
      {
        r.first = _end;
        r.last  = r.first+1;
        unlock();
        return true;
      }
      _end += 1; 
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
  template<int bits>
  bool work_queue_t<bits>::steal_unsafe(range_t<bits>& r, size_type size_max )
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

} /* rts namespace */


/**
    Projection
*/
namespace impl {
  typedef rts::atomic_t<64>     atomic;
  typedef rts::range_t<64>      range;
  typedef rts::work_queue_t<64> work_queue;
}

} /* kastl namespace */

#endif
