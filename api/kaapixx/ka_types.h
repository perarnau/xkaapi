/*
** xkaapi
** 
** Created on Tue Mar 31 15:19:14 2009
** Copyright 2009 INRIA.
**
** Contributors :
**
** thierry.gautier@inrialpes.fr
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
#ifndef _KAAPI_XX__TYPES_H_
#define _KAAPI_XX__TYPES_H_

#include "kaapi.h"
#include "ka_error.h"
#include "ka_debug.h"

/** \namespace ka
    \brief The atha namespace contains definitions to port the library onto an operating system
*/
namespace ka {

// -------------------------------------------------------------------------
/** Compatibility with kaapi type kaapi_globalid_t
*/
typedef kaapi_globalid_t GlobalId;

  /* C++ atomics encapsulation, low level memory routines 
     * only specialized for signed 32 bits and 64 bits integer
  */
  template<int bits>
  struct atomic_t;
  
  template<>
  class atomic_t<32> {
  public:
    atomic_t<32>(int32_t value =0)
    { 
#if defined(__i386__)||defined(__x86_64)
      kaapi_assert_debug( ((unsigned long)&_atom & (32/8-1)) == 0 ); 
#endif
      KAAPI_ATOMIC_WRITE(&_atom, value);
    }

    int32_t read() const 
    { return KAAPI_ATOMIC_READ(&_atom); }

    void write( int32_t value ) 
    { KAAPI_ATOMIC_WRITE(&_atom, value); }
    
    void write_barrier( int32_t value ) 
    { KAAPI_ATOMIC_WRITE_BARRIER(&_atom, value); }
    
    bool cas( int32_t oldvalue, int32_t newvalue )
    { return KAAPI_ATOMIC_CAS( &_atom, oldvalue, newvalue ); }

    int32_t incr( )
    { return KAAPI_ATOMIC_INCR( &_atom ); }

    int32_t sub( int32_t v )
    { return KAAPI_ATOMIC_SUB( &_atom, v ); }

    int32_t fetch_and_or( int32_t mask )
    { return KAAPI_ATOMIC_OR_ORIG( &_atom, mask ); }

    int32_t fetch_and_and( int32_t mask )
    { return KAAPI_ATOMIC_AND_ORIG( &_atom, mask ); }

    int32_t fetch_and_xor( int32_t mask )
    { return KAAPI_ATOMIC_XOR_ORIG( &_atom, mask ); }

    int32_t or_and_fetch( int32_t mask )
    { return KAAPI_ATOMIC_OR( &_atom, mask ); }

    int32_t and_and_fetch( int32_t mask )
    { return KAAPI_ATOMIC_AND( &_atom, mask ); }

    int32_t xor_and_fetch( int32_t mask )
    { return KAAPI_ATOMIC_XOR( &_atom, mask ); }

  protected:
    kaapi_atomic32_t _atom;
  };


  template<>
  class atomic_t<64> {
  public:
    atomic_t<64>(int64_t value =0)
    { 
      KAAPI_ATOMIC_WRITE(&_atom, value);
#if defined(__i386__)||defined(__x86_64)
      kaapi_assert_debug( ((unsigned long)this & (64/8-1)) == 0 ); 
#endif
    }

    int64_t read() const 
    { return KAAPI_ATOMIC_READ(&_atom); }

    void write( int64_t value )
    { KAAPI_ATOMIC_WRITE(&_atom, value); }
    
    void write_barrier( int64_t value ) 
    { KAAPI_ATOMIC_WRITE_BARRIER(&_atom, value); }
        
    bool cas( int64_t oldvalue, int64_t newvalue )
    { return KAAPI_ATOMIC_CAS64( &_atom, oldvalue, newvalue ); }

    int64_t incr( )
    { return KAAPI_ATOMIC_INCR64( &_atom ); }

    int64_t sub( int64_t v )
    { return KAAPI_ATOMIC_SUB64( &_atom, v ); }

    int64_t fetch_and_or( int64_t mask )
    { return KAAPI_ATOMIC_OR64_ORIG( &_atom, mask ); }

    int64_t fetch_and_and( int64_t mask )
    { return KAAPI_ATOMIC_AND64_ORIG( &_atom, mask ); }

    int64_t fetch_and_xor( int64_t mask )
    { return KAAPI_ATOMIC_XOR64_ORIG( &_atom, mask ); }

    int64_t or_and_fetch( int64_t mask )
    { return KAAPI_ATOMIC_OR64( &_atom, mask ); }

    int64_t and_and_fetch( int64_t mask )
    { return KAAPI_ATOMIC_AND64( &_atom, mask ); }

    int64_t xor_and_fetch( int64_t mask )
    { return KAAPI_ATOMIC_XOR64( &_atom, mask ); }

  protected:
    kaapi_atomic64_t _atom;
  };
  
}
#endif
