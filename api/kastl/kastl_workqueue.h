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
#include <limits>


namespace kastl {
  
/* --- most of these class / structure should be put into the C API
   * the work_queue structure is more general that its usage in KaSTL
   it could be used inside the C runtime to manage the steal of task/frame.
*/  
namespace rts {

  /* C++ atomics encapsulation, low level memory routines 
     * only specialized for signed 32 bits and 64 bits integer
  */
  template<int bits>
  struct atomic_t;
  
  template<>
  class atomic_t<32> {
  public:
    atomic_t<32>(kaapi_int32_t value)
    { 
#if defined(__i386__)||defined(__x86_64)
      kaapi_assert_debug( ((unsigned long)&_atom & (32-1)) == 0 ); 
#endif
      KAAPI_ATOMIC_WRITE(&_atom, value);
    }

    kaapi_int32_t read() const 
    { return KAAPI_ATOMIC_READ(&_atom); }

    void write( kaapi_int32_t value ) 
    { KAAPI_ATOMIC_WRITE(&_atom, value); }
    
    bool cas( kaapi_int32_t oldvalue, kaapi_int32_t newvalue )
    { return KAAPI_ATOMIC_CAS( &_atom, oldvalue, newvalue ); }

  protected:
    kaapi_atomic32_t _atom;
  };


  template<>
  class atomic_t<64> {
  public:
    atomic_t<64>(kaapi_int64_t value)
    { 
      KAAPI_ATOMIC_WRITE(&_atom, value);
#if defined(__i386__)||defined(__x86_64)
      kaapi_assert_debug( (unsigned long)this & (64-1) == 0 ); 
#endif
    }

    kaapi_int64_t read() const 
    { return KAAPI_ATOMIC_READ(&_atom); }

    void write( kaapi_int64_t value )
    { KAAPI_ATOMIC_WRITE(&_atom, value); }
    
    bool cas( kaapi_int64_t oldvalue, kaapi_int64_t newvalue )
    { return KAAPI_ATOMIC_CAS( &_atom, oldvalue, newvalue ); }

  protected:
    kaapi_atomic64_t _atom;
  };


  /* Mini trait to get the C++ signed integral type for representing integer 
     on given bits 
  */
  template<int bits> struct signed_type_that_have_bits{ };
  template<> struct signed_type_that_have_bits<8> { typedef  kaapi_int8_t type; };
  template<> struct signed_type_that_have_bits<16>{ typedef  kaapi_int16_t type; };
  template<> struct signed_type_that_have_bits<32>{ typedef  kaapi_int32_t type; };
  template<> struct signed_type_that_have_bits<64>{ typedef  kaapi_int64_t type; };

  template<typename type> struct bits_for_type{ 
    enum { bits = sizeof(type)*8 };
  };
  
  /** work range: two public available integer of 'bits' bits
   */
  template<int bits>
  struct range
  {
    // must be signed int
    typedef typename signed_type_that_have_bits<bits>::type index_type;
    typedef typename range<bits>::index_type                size_type;
    
    index_type first;
    index_type last;
    
    /* */
    range()
#if defined(KAAPI_DEBUG)    
     : first(0), last(0)
#endif
    { }

    /* */
    range( index_type f, index_type l )
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
    
    /* shift the range to put the first at orig */
    void shift( range::index_type orig )
    {
      last  -= first-orig;
      first = orig;
    }
  };

  
  /** work work_queue: the main important data structure.
      It steal/pop are managed by a Disjkstra like protocol.
      The threads that want to steal serialize their access
      through a lock.
   */
  template<int bits>
  class work_queue {
  public:
    typedef typename range<bits>::size_type  size_type;
    typedef typename range<bits>::index_type index_type;
    
    /* default cstor */
    work_queue();
    
    /* set the work_queue */
    void set( const range<bits>& );

    /* clear the work_queue 
     */
    void clear();
    
    /* return true iff the work_queue is empty 
     */
    bool is_empty() const;
    
    /* return the size of the work_queue 
     */
    size_type size() const;
    
    /* push a new valid subrange at the begin of the queue.
       Only valid of the pushed range just before the remainding part of the queue,
       i.e. iff queue.beg == range.last.
       If the range is valid then returns true else return false.
       The method is concurrent with the steal's methods.
     */
    bool push_front( const range<bits>& r );
    
    /* extend the queue from [_beg,_end) to [first, _end)
       Only valid of the first < _beg.
       If its valid then returns true else return false.
       The method is concurrent with the steal's methods.
     */
    bool push_front( const typename range<bits>::index_type& first );

    /* pop one element
       return true in case of success
     */
    bool pop(typename range<bits>::index_type&);
    
    /* pop a subrange of size at most sz 
       return true in case of success
     */
    bool pop(range<bits>&, typename range<bits>::size_type sz);

    /* push_back a new valid subrange at the end of the queue.
       Only valid of the pushed range just after the queue,
       i.e. iff queue.end == range.first.
       If the range is valid then returns true else return false.
       The method is concurrent with the pop's method.
     */
    bool push_back( const range<bits>& r );
    
    /* extend the queue from [_beg,_end) to [_beg, last)
       Only valid of the last > _end.
       If its valid then returns true else return false.
       The method is concurrent with the pop's method.
     */
    bool push_back( const typename range<bits>::index_type& last );

    /* steal one element
       return true in case of success
     */
    bool steal(typename range<bits>::index_type&);
    
    /* steal a subrange of size at most sz
       return true in case of success
     */
    bool steal(range<bits>&, typename range<bits>::size_type sz);
    
    /* steal a subrange of size at most sz_max
       return true in case of success
     */
    bool steal(range<bits>&, typename range<bits>::size_type sz_max, typename range<bits>::size_type sz_min);
    
  private:
    /* */
    bool slow_pop( range<bits>&, typename range<bits>::size_type sz );
    
    /* */
    void lock_pop();

    /* */
    void lock_steal();
    
    /* */
    void unlock();
    
    /* data field required to be correctly aligned in order to ensure atomicity of read/write. 
       Put them on two separate lines of cache (assume == 64bytes) due to different access by 
       concurrent threads.
       Currently only IA32 & x86-64.
       An assertion is put inside the constructor to verify that this field are correctly aligned.
     */
    index_type volatile _beg __attribute__((aligned(64)));
    index_type volatile _end __attribute__((aligned(64)));

    atomic_t<32> _lock __attribute__((aligned(64))); /* one bit is enough .. */
  };
  
  /** */
  template<int bits>
  inline work_queue<bits>::work_queue()
  : _beg(0), _end(0), _lock(0)
  {
#if defined(__i386__)||defined(__x86_64)
    kaapi_assert_debug( (((unsigned long)&_beg) & (bits-1)) == 0 ); 
    kaapi_assert_debug( (((unsigned long)&_end) & (bits-1)) == 0 );
#else
#  warning "May be alignment constraints exit to garantee atomic read write"
#endif
  }
  
  /** */
  template<int bits>
  inline void work_queue<bits>::set( const range<bits>& r)
  {
    /* try to order writes to always guarantee that 
       - if the queue not empty before calling the method and 
       if the range is not empty
       - then the queue is empty during the execution
       - then the queue is not empty on return
    */
    _end = std::numeric_limits<typename work_queue<bits>::index_type>::min();
    kaapi_writemem_barrier();
    _beg = r.first;
    kaapi_writemem_barrier();
    _end = r.last;
  }

  /** */
  template<int bits>
  inline void work_queue<bits>::clear()
  {
    _end = 0;
    _beg = 0; 
  }
  
  /** */
  template<int bits>
  inline typename work_queue<bits>::size_type work_queue<bits>::size() const
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
  inline bool work_queue<bits>::is_empty() const
  {
    /* inverse ici... critical path optimization ? on veut la valeur la plus
       à jour possible pour la victime (pop)
    */
    index_type b = _beg;
    kaapi_readmem_barrier();
    index_type e = _end;
    return e <= b;
  }

  /** */
  template<int bits>
  inline bool work_queue<bits>::push_front( const range<bits>& r )
  {
    kaapi_assert_debug( !r.is_empty() ) ;
    if (r.last != _beg ) return false;
    kaapi_writemem_barrier();
    _beg = r.first;
    return true;
  }
  
  /** */
  template<int bits>
  inline bool work_queue<bits>::push_front( const work_queue<bits>::index_type& first )
  {
    if (first > _beg ) return false;
    kaapi_writemem_barrier();
    _beg = first;
    return true;
  }


  /** */
  template<int bits>
  inline bool work_queue<bits>::pop( typename range<bits>::index_type& item )
  {
    item = _beg;
    ++_beg;
    /* read of _end after write of _beg */
    kaapi_mem_barrier(); 
    if (_beg > _end) {
      range<bits> r;
      bool retval = slow_pop( r, 1 );
      item = r.first;
      return retval;
    }
    
    return true;
  }


  /** */
  template<int bits>
  inline bool work_queue<bits>::pop(range<bits>& r, typename range<bits>::size_type size)
  {
    r.first = _beg;
    _beg += size;
    /* read of _end after write of _beg */
    kaapi_mem_barrier();
    if (_beg > _end) return slow_pop( r, size );
    
    r.last = r.first + size;
    
    return true;
  }


  /** */
  template<int bits>
  inline bool work_queue<bits>::push_back( const range<bits>& r )
  {
    kaapi_assert_debug( !r.is_empty() ) ;
    if (r.first != _end ) return false;
    kaapi_writemem_barrier();
    _end = r.last;
    return true;
  }


  /** */
  template<int bits>
  inline bool work_queue<bits>::push_back( const typename range<bits>::index_type& last )
  {
    if (last < _end ) return false;
    kaapi_writemem_barrier();
    _end = last;
    return true;
  }


  /** */
  template<int bits>
  inline bool work_queue<bits>::steal(typename range<bits>::index_type& item)
  {
    range<bits> r;
    bool retval = steal(r, 1);
    if (retval) item = r.first;
    return retval;
  }

} /* impl namespace */


/**
    Projection
*/
namespace impl {
  typedef rts::atomic_t<64>   atomic;
  typedef rts::range<64>      range;
  typedef rts::work_queue<64> work_queue;
}

} /* kastl namespace */

#endif
