/*
 ** xkaapi
 ** 
 ** Created on Tue Mar 31 15:19:14 2009
 ** Copyright 2009 INRIA.
 **
 ** Contributors :
 **
 ** thierry.gautier@inrialpes.fr
 
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
#ifndef _KASTL_FIFO_QUEUE_H_
#define _KASTL_FIFO_QUEUE_H_
#include "kastl_workqueue.h"


namespace kastl {
  
/* --- most of these class / structure should be put into the C API
   * the work_queue structure is more general that its usage in KaSTL
   it could be used inside the C runtime to manage the steal of task/frame.
*/  
namespace rts {

  /* mini trait to give the best type to represent 'capacity' items
  */
  template<bool inf_256, bool inf_65536, bool inf_4294967296>
  struct type_that_is_less {};

  template<>
  struct type_that_is_less<true,true,true> {
    typedef unsigned char type;
  };
  template<>
  struct type_that_is_less<false,true,true> {
    typedef unsigned short type;
  };
  template<>
  struct type_that_is_less<false,false,true> {
    typedef unsigned int type;
  };
  template<unsigned int capacity>
  struct type_that_represents_capacity
  { 
    typedef typename type_that_is_less<
         (capacity<=255U),
         (capacity<=65535U),
         (capacity<=4294967295U) 
    >::type type;
  };

} // namespace rts

namespace impl {

  /** fifo_queue: a main important data structure.
      through a lock.
   */
  template<typename T,int capacity=1>
  class fifo_queue {
  public:
    typedef typename rts::type_that_represents_capacity<capacity>::type size_type;
    typedef typename rts::type_that_represents_capacity<capacity>::type index_type;

    /* default cstor */
    fifo_queue();
    
    /* clear the work_queue 
     */
    void clear();
    
    /* return the size of the work_queue 
     */
    size_type size() const;
    
    /* enqueue entry method */
    bool enqueue(T* data);
    
    /* dequeue */
    bool dequeue(T*& data);

  private:
    /* data field required to be correctly aligned in order to ensure atomicity of read/write. 
       Put them on two separate lines of cache (assume == 64bytes) due to different access by 
       concurrent threads.
       Currently only IA32 & x86-64.
       An assertion is put inside the constructor to verify that this field are correctly aligned.
     */
    index_type volatile _head __attribute__((aligned(64)));
    T* volatile         _buffer[capacity];
    index_type volatile _tail __attribute__((aligned(64)));
  };
  
  /** */
  template<typename T,int capacity>
  inline fifo_queue<T,capacity>::fifo_queue()
  : _head(0), _tail(0)
  {
#if defined(__i386__)||defined(__x86_64)
    kaapi_assert_debug( (unsigned long)&_head & (rts::bits_for_type<index_type>::bits-1) == 0 ); 
    kaapi_assert_debug( (unsigned long)&_tail & (rts::bits_for_type<index_type>::bits-1) == 0 );
#else
#  warning "May be alignment constraints exit to garantee atomic read write"
#endif
  }
  
  /** */
  template<typename T,int capacity>
  inline void fifo_queue<T,capacity>::clear()
  {
    _head = 0;
    _tail = 0; 
  }
  
  /** */
  template<typename T,int capacity>
  inline typename fifo_queue<T,capacity>::size_type fifo_queue<T,capacity>::size() const
  {
    /* on lit d'abord _end avant _beg afin que le voleur puisse qui a besoin
       en general d'avoir la taille puis avoir la valeur la + ajour possible...
    */
    index_type e = _head;
    kaapi_readmem_barrier();
    index_type b = _tail;
    return b < e ? e-b : 0;
  }
  
  /** */
  template<typename T,int capacity>
  inline bool fifo_queue<T,capacity>::enqueue(T* data) 
  {
    if (0 != _buffer[_head])
        return false;

    _buffer[_head] = data;
    kaapi_writemem_barrier();
    _head = (_head+1 < capacity ? _head+1 : 0);
    return true;
  }
    
  /** */
  template<typename T,int capacity>
  inline bool fifo_queue<T,capacity>::dequeue(T*& data) 
  {
    data = _buffer[_tail];
    if (0 == data)
        return false;
    _buffer[_tail] = 0;
    kaapi_writemem_barrier();
    _tail = (_tail+1 < capacity ? _tail+1 : 0);
    return 0;
  }

} /* impl namespace */

} /* kastl namespace */

#endif
