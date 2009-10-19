/*
** kaapi_atomic.h
** ckaapi
** 
** Created on Tue Mar 31 15:20:42 2009
** Copyright 2009 INRIA.
**
** Contributors :
**
** christophe.laferriere@imag.fr
** thierry.gautier@imag.fr
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
#ifndef _KAAPI_ATOMIC_H
#define _KAAPI_ATOMIC_H 1

#include "kaapi_config.h"
#include "kaapi_private_structure.h"

/* ============================= Atomic Function ============================ */

#if (((__GNUC__ == 4) && (__GNUC_MINOR__ >= 1)) || (__GNUC__ > 4) \
|| defined(__INTEL_COMPILER))
// Note: ICC seems to also support these builtins functions
#  if defined(__INTEL_COMPILER)
#    warning Using ICC. Please, check if icc really support atomic operations
/* ia64 impl using compare and exchange */
//#    define KAAPI_CAS(_a, _o, _n) _InterlockedCompareExchange(_a, _n, _o )
#  endif

#  define KAAPI_ATOMIC_CAS(a, o, n) \
__sync_bool_compare_and_swap( &((a)->_counter), o, n) 

#  define KAAPI_ATOMIC_INCR(a) \
__sync_add_and_fetch( &((a)->_counter), 1 ) 

#  define KAAPI_ATOMIC_DECR(a) \
__sync_sub_and_fetch( &((a)->_counter), 1 ) 

#  define KAAPI_ATOMIC_SUB(a, value) \
__sync_sub_and_fetch( &((a)->_counter), value ) 

#  define KAAPI_ATOMIC_READ(a) \
((a)->_counter)

#  define KAAPI_ATOMIC_WRITE(a, value) \
(a)->_counter = value

#elif defined(KAAPI_USE_APPLE) // if gcc version on Apple is less than 4.1

#  include <libkern/OSAtomic.h>

#  define KAAPI_ATOMIC_CAS(a, o, n) \
OSAtomicCompareAndSwap32Barrier( o, n, &((a)->_counter)) 

#  define KAAPI_ATOMIC_INCR(a) \
OSAtomicIncrement32Barrier( &((a)->_counter) ) 

#  define KAAPI_ATOMIC_DECR(a) \
OSAtomicDecrement32Barrier(&((a)->_counter) ) 

#  define KAAPI_ATOMIC_SUB(a, value) \
OSAtomicAdd32Barrier( -value, &((a)->_counter) ) 

#  define KAAPI_ATOMIC_READ(a) \
((a)->_counter)

#  define KAAPI_ATOMIC_WRITE(a, value) \
(a)->_counter = value

#else
#  error "Please add support for atomic operations on this system/architecture"
#endif // GCC > 4.1

/* ========================================================================== */

/* ============================= Memory Barrier ============================= */

#if defined(KAAPI_USE_APPLE)
#  include <libkern/OSAtomic.h>

static inline void kaapi_writemem_barrier()  
{
  OSMemoryBarrier();
  // Compiler fence to keep operations from
  __asm__ __volatile__("" : : : "memory" );
}

static inline void kaapi_readmem_barrier()  
{
  OSMemoryBarrier();
  // Compiler fence to keep operations from
  __asm__ __volatile__("" : : : "memory" );
}

#elif defined(KAAPI_USE_LINUX)

#  define kaapi_writemem_barrier() \
__sync_synchronize()

#  define kaapi_readmem_barrier() \
__sync_synchronize()

#else
#  error "Undefined barrier"
#endif // KAAPI_USE_APPLE


/* ========================================================================== */
/** Termination dectecting barrier
*/

/**
*/
#define kaapi_barrier_td_init( kpb, value ) \
  KAAPI_ATOMIC_WRITE( kpb, value )

/**
*/
#define kaapi_barrier_td_destroy( kpb )
  
/**
*/
#define kaapi_barrier_td_setactive( kpb, b ) \
  if (b) { KAAPI_ATOMIC_INCR( kpb ); } \
  else KAAPI_ATOMIC_DECR( kpb )

/**
*/
#define kaapi_barrier_td_isterminated( kpb ) \
  (KAAPI_ATOMIC_READ(kpb ) == 0)

#endif // _KAAPI_ATOMIC_H
