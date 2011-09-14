/*
** xkaapi
** 
** Created on Tue Mar 31 15:19:09 2009
** Copyright 2009 INRIA.
**
** Contributors :
**
** thierry.gautier@inrialpes.fr
** fabien.lementec@gmail.com / fabien.lementec@imag.fr
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
#ifndef _KAAPI_ATOMIC_H_
#define _KAAPI_ATOMIC_H_ 1

#if defined(__cplusplus)
extern "C" {
#endif

#include <stdint.h>

/** Atomic type
*/
typedef struct kaapi_atomic32_t {
  volatile uint32_t _counter;
} kaapi_atomic32_t;
typedef kaapi_atomic32_t kaapi_atomic_t;


typedef struct kaapi_atomic64_t {
  volatile uint64_t _counter;
} kaapi_atomic64_t;

/* ========================= Low level memory barrier, inline for perf... so ============================= */
/** Implementation note
    - all functions or macros without _ORIG return the new value after apply the operation.
    - all functions or macros with ORIG return the old value before applying the operation.
*/
#if defined(KAAPI_DEBUG)
static inline int __kaapi_isaligned(const volatile void* a, size_t byte)
{
  kaapi_assert( (((uintptr_t)a) & ((unsigned long)byte - 1)) == 0 );
  return 1;
}
#  define __KAAPI_ISALIGNED_ATOMIC(a,instruction)\
      (__kaapi_isaligned( &(a)->_counter, sizeof((a)->_counter)) ? (instruction) : 0)
#else
static inline int __kaapi_isaligned(const volatile void* a, size_t byte)
{
  if ((((uintptr_t)a) & ((unsigned long)byte - 1)) == 0 ) return 1;
  return 0;
}
#  define __KAAPI_ISALIGNED_ATOMIC(a,instruction)\
      (instruction)
#endif

#define KAAPI_ATOMIC_READ(a) \
  __KAAPI_ISALIGNED_ATOMIC(a, (a)->_counter)

#define KAAPI_ATOMIC_WRITE(a, value) \
  __KAAPI_ISALIGNED_ATOMIC(a, (a)->_counter = value)

#define KAAPI_ATOMIC_WRITE_BARRIER(a, value) \
    __KAAPI_ISALIGNED_ATOMIC(a, (kaapi_mem_barrier(), (a)->_counter = value))

//BEFORE:    __KAAPI_ISALIGNED_ATOMIC(a, (kaapi_writemem_barrier(), (a)->_counter = value))

#if (((__GNUC__ == 4) && (__GNUC_MINOR__ >= 1)) || (__GNUC__ > 4) \
|| defined(__INTEL_COMPILER))
/* Note: ICC seems to also support these builtins functions */
#  if defined(__INTEL_COMPILER)
#    warning Using ICC. Please, check if icc really support atomic operations
/* ia64 impl using compare and exchange */
/*#    define KAAPI_CAS(_a, _o, _n) _InterlockedCompareExchange(_a, _n, _o ) */
#  endif

#  define KAAPI_ATOMIC_CAS(a, o, n) \
    __KAAPI_ISALIGNED_ATOMIC(a, __sync_bool_compare_and_swap( &((a)->_counter), o, n))

/* functions which return new value (NV) */
#  define KAAPI_ATOMIC_INCR(a) \
    __KAAPI_ISALIGNED_ATOMIC(a, __sync_add_and_fetch( &((a)->_counter), 1 ))

#  define KAAPI_ATOMIC_DECR(a) \
    __KAAPI_ISALIGNED_ATOMIC(a, __sync_sub_and_fetch( &((a)->_counter), 1 ))

#  define KAAPI_ATOMIC_ADD(a, value) \
    __KAAPI_ISALIGNED_ATOMIC(a, __sync_add_and_fetch( &((a)->_counter), value ))

#  define KAAPI_ATOMIC_SUB(a, value) \
    __KAAPI_ISALIGNED_ATOMIC(a, __sync_sub_and_fetch( &((a)->_counter), value ))

#  define KAAPI_ATOMIC_AND(a, o) \
    __KAAPI_ISALIGNED_ATOMIC(a, __sync_and_and_fetch( &((a)->_counter), o ))

#  define KAAPI_ATOMIC_OR(a, o) \
    __KAAPI_ISALIGNED_ATOMIC(a, __sync_or_and_fetch( &((a)->_counter), o ))

#  define KAAPI_ATOMIC_XOR(a, o) \
    __KAAPI_ISALIGNED_ATOMIC(a, __sync_xor_and_fetch( &((a)->_counter), o ))

/* linux functions which return old value */
#  define KAAPI_ATOMIC_AND_ORIG(a, o) \
    __KAAPI_ISALIGNED_ATOMIC(a, __sync_fetch_and_and( &((a)->_counter), o ))

#  define KAAPI_ATOMIC_OR_ORIG(a, o) \
    __KAAPI_ISALIGNED_ATOMIC(a, __sync_fetch_and_or( &((a)->_counter), o ))

#  define KAAPI_ATOMIC_XOR_ORIG(a, o) \
    __KAAPI_ISALIGNED_ATOMIC(a, __sync_fetch_and_xor( &((a)->_counter), o ))

/* linux 64 bit versions */
#  define KAAPI_ATOMIC_CAS64(a, o, n) \
    __KAAPI_ISALIGNED_ATOMIC(a, __sync_bool_compare_and_swap( &((a)->_counter), o, n))

/* linux functions which return new value (NV) */
#  define KAAPI_ATOMIC_INCR64(a) \
    __KAAPI_ISALIGNED_ATOMIC(a, __sync_add_and_fetch( &((a)->_counter), 1 ) )

#  define KAAPI_ATOMIC_DECR64(a) \
    __KAAPI_ISALIGNED_ATOMIC(a, __sync_sub_and_fetch( &((a)->_counter), 1 ) )

#  define KAAPI_ATOMIC_ADD64(a, value) \
    __KAAPI_ISALIGNED_ATOMIC(a, __sync_add_and_fetch( &((a)->_counter), value ) )

#  define KAAPI_ATOMIC_SUB64(a, value) \
    __KAAPI_ISALIGNED_ATOMIC(a, __sync_sub_and_fetch( &((a)->_counter), value ) )

#  define KAAPI_ATOMIC_AND64(a, o) \
    __KAAPI_ISALIGNED_ATOMIC(a, __sync_and_and_fetch( &((a)->_counter), o ))

#  define KAAPI_ATOMIC_OR64(a, o) \
    __KAAPI_ISALIGNED_ATOMIC(a, __sync_or_and_fetch( &((a)->_counter), o ))

#  define KAAPI_ATOMIC_XOR64(a, o) \
    __KAAPI_ISALIGNED_ATOMIC(a, __sync_xor_and_fetch( &((a)->_counter), o ))

/* functions which return old value */
#  define KAAPI_ATOMIC_AND64_ORIG(a, o) \
    __KAAPI_ISALIGNED_ATOMIC(a, __sync_fetch_and_and( &((a)->_counter), o ))

#  define KAAPI_ATOMIC_OR64_ORIG(a, o) \
    __KAAPI_ISALIGNED_ATOMIC(a, __sync_fetch_and_or( &((a)->_counter), o ))

#  define KAAPI_ATOMIC_XOR64_ORIG(a, o) \
    __KAAPI_ISALIGNED_ATOMIC(a, __sync_fetch_and_xor( &((a)->_counter), o ))


#elif defined(__APPLE__) /* if gcc version on Apple is less than 4.1 */

#  include <libkern/OSAtomic.h>

#  define KAAPI_ATOMIC_CAS(a, o, n) \
    OSAtomicCompareAndSwap32( o, n, &((a)->_counter)) 

/* functions which return new value (NV) */
#  define KAAPI_ATOMIC_INCR(a) \
    OSAtomicIncrement32Barrier( &((a)->_counter) ) 

#  define KAAPI_ATOMIC_DECR32(a) \
    OSAtomicDecrement32Barrier(&((a)->_counter) ) 

#  define KAAPI_ATOMIC_ADD(a, value) \
    OSAtomicAdd32Barrier( value, &((a)->_counter) ) 

#  define KAAPI_ATOMIC_SUB(a, value) \
    OSAtomicAdd32Barrier( -value, &((a)->_counter) ) 

#  define KAAPI_ATOMIC_AND(a, o) \
    OSAtomicAnd32Barrier( o, &((a)->_counter) )

#  define KAAPI_ATOMIC_OR(a, o) \
    OSAtomicOr32Barrier( o, &((a)->_counter) )

#  define KAAPI_ATOMIC_XOR(a, o) \
    OSAtomicXor32Barrier( o, &((a)->_counter) )

/* functions which return old value */
#  define KAAPI_ATOMIC_AND_ORIG(a, o) \
    OSAtomicAnd32OrigBarrier( o, &((a)->_counter) )

#  define KAAPI_ATOMIC_OR_ORIG(a, o) \
    OSAtomicOr32OrigBarrier( o, &((a)->_counter) )

#  define KAAPI_ATOMIC_XOR_ORIG(a, o) \
    OSAtomicXor32OrigBarrier( o, &((a)->_counter) )

/* 64 bit versions */
#  define KAAPI_ATOMIC_CAS64(a, o, n) \
    OSAtomicCompareAndSwap64( o, n, &((a)->_counter)) 

/* functions which return new value (NV) */
#  define KAAPI_ATOMIC_INCR64(a) \
    OSAtomicIncrement64Barrier( &((a)->_counter) ) 

#  define KAAPI_ATOMIC_DECR64(a) \
    OSAtomicDecrement64Barrier(&((a)->_counter) ) 

#  define KAAPI_ATOMIC_ADD64(a, value) \
    OSAtomicAdd64Barrier( value, &((a)->_counter) ) 

#  define KAAPI_ATOMIC_SUB64(a, value) \
    OSAtomicAdd64Barrier( -value, &((a)->_counter) ) 

#  define KAAPI_ATOMIC_AND64(a, o) \
    OSAtomicAnd64Barrier( o, &((a)->_counter) )

#  define KAAPI_ATOMIC_OR64(a, o) \
    OSAtomicOr64Barrier( o, &((a)->_counter) )

#  define KAAPI_ATOMIC_XOR64(a, o) \
    OSAtomicXor64Barrier( o, &((a)->_counter) )

/* functions which return old value */
#  define KAAPI_ATOMIC_AND64_ORIG(a, o) \
    OSAtomicAnd64OrigBarrier( o, &((a)->_counter) )

#  define KAAPI_ATOMIC_OR64_ORIG(a, o) \
    OSAtomicOr64OrigBarrier( o, &((a)->_counter) )

#  define KAAPI_ATOMIC_XOR64_ORIG(a, o) \
    OSAtomicXor64OrigBarrier( o, &((a)->_counter) )
#else
#  error "Please add support for atomic operations on this system/architecture"
#endif /* GCC > 4.1 */


#if defined(__i386__)||defined(__x86_64)
#  define kaapi_slowdown_cpu() \
      do { __asm__ __volatile__("pause\n\t"); } while (0)
#else
#  define kaapi_slowdown_cpu()
#endif



#if defined(__APPLE__)
#  include <libkern/OSAtomic.h>
static inline void kaapi_writemem_barrier()  
{
#  ifdef __ppc__
  OSMemoryBarrier();
#  elif defined(__x86_64) || defined(__i386__)
  /* not need sfence on X86 archi: write are ordered */
  __asm__ __volatile__ ("":::"memory");
#  else
#    error "bad configuration"
#  endif
}

static inline void kaapi_readmem_barrier()  
{
#  ifdef __ppc__
  OSMemoryBarrier();
#  elif defined(__x86_64) || defined(__i386__)
  /* not need lfence on X86 archi: read are ordered */
  __asm__ __volatile__ ("lfence":::"memory");
#  else
#    error "bad configuration"
#  endif
}

/* should be both read & write barrier */
static inline void kaapi_mem_barrier()  
{
#  ifdef __ppc__
  OSMemoryBarrier();
#  elif defined(__x86_64) || defined(__i386__)
  /* not need lfence on X86 archi: read are ordered */
  __asm__ __volatile__ ("mfence":::"memory");
#  else
#    error "bad configuration"
#  endif
}

#elif defined(__linux__)

static inline void kaapi_writemem_barrier()  
{
#  if defined(__x86_64) || defined(__i386__)
  /* not need sfence on X86 archi: write are ordered */
  __asm__ __volatile__ ("":::"memory");
#  elif defined(__GNUC__)
  __sync_synchronize();
#  else
#  error "Compiler not supported"
/* xlC ->__lwsync() / bultin */  
#  endif
}

static inline void kaapi_readmem_barrier()  
{
#  if defined(__x86_64) || defined(__i386__)
  /* not need lfence on X86 archi: read are ordered */
  __asm__ __volatile__ ("":::"memory");
#  else
  __sync_synchronize();
#  endif
}

/* should be both read & write barrier */
static inline void kaapi_mem_barrier()  
{
  __sync_synchronize();
}

#elif defined(_WIN32)
static inline void kaapi_writemem_barrier()  
{
  /* Compiler fence to keep operations from */
  /* not need sfence on X86 archi: write are ordered */
  __asm__ __volatile__ ("":::"memory");
}

static inline void kaapi_readmem_barrier()  
{
  /* Compiler fence to keep operations from */
  /* not need lfence on X86 archi: read are ordered */
  __asm__ __volatile__ ("":::"memory");
}

/* should be both read & write barrier */
static inline void kaapi_mem_barrier()  
{
   LONG Barrier = 0;
   __asm__ __volatile__("xchgl %%eax,%0 "
     :"=r" (Barrier));

  /* Compiler fence to keep operations from */
  __asm__ __volatile__("" : : : "memory" );
}


#else
#  error "Undefined barrier"
#endif



/** Note on scheduler lock:
  KAAPI_SCHED_LOCK_CAS -> lock state == 1 iff lock is taken, else 0
  KAAPI_SCHED_LOCK_CAS not defined: see 
    Sewell, P., Sarkar, S., Owens, S., Nardelli, F. Z., and Myreen, M. O. 2010. 
    x86-TSO: a rigorous and usable programmer's model for x86 multiprocessors. 
    Commun. ACM 53, 7 (Jul. 2010), 89-97. 
    DOI= http://doi.acm.org/10.1145/1785414.1785443
*/
static inline int kaapi_atomic_initlock( kaapi_atomic_t* lock )
{
#if defined(KAAPI_SCHED_LOCK_CAS)
  KAAPI_ATOMIC_WRITE(lock,0);
#else
  KAAPI_ATOMIC_WRITE(lock,1);
#endif
  return 0;
}

static inline int kaapi_atomic_trylock( kaapi_atomic_t* lock )
{
#if defined(KAAPI_SCHED_LOCK_CAS)
  int ok;
  /* implicit barrier in KAAPI_ATOMIC_CAS if lock is taken */
  ok = (KAAPI_ATOMIC_READ(lock) ==0) && KAAPI_ATOMIC_CAS(lock, 0, 1);
  kaapi_assert_debug( !ok || (ok && KAAPI_ATOMIC_READ(lock) == 1) );
  return ok;
#else
  if (KAAPI_ATOMIC_DECR(lock) ==0) 
  {
    return 1;
  }
  return 0;
#endif
}

/** 
*/
static inline int kaapi_atomic_lock( kaapi_atomic_t* lock )
{
#if defined(KAAPI_SCHED_LOCK_CAS)
  int ok;
  do {
    ok = (KAAPI_ATOMIC_READ(lock) ==0) && KAAPI_ATOMIC_CAS(lock, 0, 1);
    if (ok) break;
    kaapi_slowdown_cpu();
  } while (1);
  /* implicit barrier in KAAPI_ATOMIC_CAS */
  kaapi_assert_debug( KAAPI_ATOMIC_READ(lock) != 0 );
  return 0;
#else
acquire:
  if (KAAPI_ATOMIC_DECR(lock) ==0) return 1;
  while (KAAPI_ATOMIC_READ(lock) <=0)
  {
    kaapi_slowdown_cpu();
  }
  goto acquire;
#endif
}


/**
*/
static inline int kaapi_atomic_lock_spin( kaapi_atomic_t* lock, int spincount )
{
#if defined(KAAPI_SCHED_LOCK_CAS)
  int ok;
  do {
    ok = (KAAPI_ATOMIC_READ(lock) ==0) && KAAPI_ATOMIC_CAS(lock, 0, 1);
    if (ok) break;
    kaapi_slowdown_cpu();
  } while (1);
  /* implicit barrier in KAAPI_ATOMIC_CAS */
  kaapi_assert_debug( KAAPI_ATOMIC_READ(lock) != 0 );
#else
  int i;
  if (KAAPI_ATOMIC_DECR(lock) ==0) return 1;
  for (i=0; (KAAPI_ATOMIC_READ(lock) <=0) && (i<spincount); ++i)
    kaapi_slowdown_cpu();
  if (KAAPI_ATOMIC_DECR(lock) ==0) return 1;
#endif
  return 0;
}


/**
*/
static inline int kaapi_atomic_unlock( kaapi_atomic_t* lock )
{
#if defined(KAAPI_SCHED_LOCK_CAS)
  kaapi_assert_debug( (unsigned)KAAPI_ATOMIC_READ(lock) == (unsigned)(1) );
  /* implicit barrier in KAAPI_ATOMIC_WRITE_BARRIER */
  KAAPI_ATOMIC_WRITE_BARRIER(lock, 0);
#else
  KAAPI_ATOMIC_WRITE_BARRIER(lock, 1);
#endif
  return 0;
}

static inline void kaapi_atomic_waitlock(kaapi_atomic_t* lock)
{
  /* wait until reaches the unlocked state */
#if defined(KAAPI_SCHED_LOCK_CAS)
  while (KAAPI_ATOMIC_READ(lock))
#else
  while (KAAPI_ATOMIC_READ(lock) == 0)
#endif
    kaapi_slowdown_cpu();
}

static inline int kaapi_atomic_islocked( kaapi_atomic_t* lock )
{
#if defined(KAAPI_SCHED_LOCK_CAS)
  return KAAPI_ATOMIC_READ(lock) != 0;
#else
  return KAAPI_ATOMIC_READ(lock) != 1;
#endif
}

#if defined(__cplusplus)
}
#endif

#endif /* */
