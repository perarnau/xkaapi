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

#if !defined(__SIZEOF_POINTER__)
#  if defined(__LP64__) || defined(__x86_64__)
#    define __SIZEOF_POINTER__ 8
#  elif defined(__i386__) || defined(__ppc__)
#    define __SIZEOF_POINTER__ 4
#  else
#    error KAAPI needs __SIZEOF_* macros. Use a recent version of gcc
#  endif
#endif

#if ((__SIZEOF_POINTER__ != 4) && (__SIZEOF_POINTER__ != 8)) 
#  error KAAPI cannot be compiled on this architecture due to strange size for __SIZEOF_POINTER__
#endif

#include "kaapi_error.h"
#include "kaapi_defs.h"

#if defined(__cplusplus)
extern "C" {
#endif

  
/* ========================= Atomic type ============================= */
/** Atomic type
*/
typedef struct kaapi_atomic8_t {
  volatile int8_t _counter;
} kaapi_atomic8_t;

typedef struct kaapi_atomic16_t {
  volatile int16_t _counter;
} kaapi_atomic16_t;

typedef struct kaapi_atomic32_t {
  volatile int32_t _counter;
} kaapi_atomic32_t;
typedef kaapi_atomic32_t kaapi_atomic_t;

typedef struct kaapi_atomic64_t {
  volatile int64_t _counter;
} kaapi_atomic64_t;

typedef struct kaapi_atomicptr_t {
  volatile intptr_t _counter;
} kaapi_atomicptr_t;

/* ========================= Low level memory barrier, inline for perf... so ============================= */
/** Implementation note
    - all functions or macros without _ORIG return the new value after apply the operation.
    - all functions or macros with ORIG return the old value before applying the operation.
    - this macros are based on GCC bultin functions.
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
    __KAAPI_ISALIGNED_ATOMIC(a, (kaapi_writemem_barrier(), (a)->_counter = value))

//BEFORE:    __KAAPI_ISALIGNED_ATOMIC(a, (kaapi_writemem_barrier(), (a)->_counter = value))

#if (((__GNUC__ == 4) && (__GNUC_MINOR__ >= 1)) || (__GNUC__ > 4) \
|| defined(__INTEL_COMPILER))
/* Note: ICC seems to also support these builtins functions */
#  if defined(__INTEL_COMPILER)
#    warning Using ICC. Please, check if icc really support atomic operations
/* ia64 impl using compare and exchange */
/*#    define KAAPI_CAS(_a, _o, _n) _InterlockedCompareExchange(_a, _n, _o ) */
#  endif

/* kcomputer additional definitions */
#if (defined(__sparc_v9__) && (defined(__fcc_version) || defined(__FCC_VERSION)))
#include "arch/kaapi_kcomputerdefs.h"
#endif

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

/* linux 8 bit versions 
*/
#  define KAAPI_ATOMIC_CAS8(a, o, n)   KAAPI_ATOMIC_CAS(a, o, n)
#  define KAAPI_ATOMIC_INCR8(a)        KAAPI_ATOMIC_INCR(a)
#  define KAAPI_ATOMIC_DECR8(a)        KAAPI_ATOMIC_DECR(a)
#  define KAAPI_ATOMIC_ADD8(a, value)  KAAPI_ATOMIC_ADD(a, value)
#  define KAAPI_ATOMIC_SUB8(a, value)  KAAPI_ATOMIC_SUB(a, value)
#  define KAAPI_ATOMIC_AND8(a, o)      KAAPI_ATOMIC_AND(a, o)
#  define KAAPI_ATOMIC_OR8(a, o)       KAAPI_ATOMIC_OR(a,o)
#  define KAAPI_ATOMIC_XOR8(a, o)      KAAPI_ATOMIC_XOR(a,o)
#  define KAAPI_ATOMIC_AND8_ORIG(a, o) KAAPI_ATOMIC_AND_ORIG(a,o)
#  define KAAPI_ATOMIC_OR8_ORIG(a, o)  KAAPI_ATOMIC_OR_ORIG(a, o)
#  define KAAPI_ATOMIC_XOR8_ORIG(a, o) KAAPI_ATOMIC_XOR_ORIG(a, o)

/* linux 64 bit versions 
*/
#  define KAAPI_ATOMIC_CAS64(a, o, n)   KAAPI_ATOMIC_CAS(a, o, n)
#  define KAAPI_ATOMIC_INCR64(a)        KAAPI_ATOMIC_INCR(a)
#  define KAAPI_ATOMIC_DECR64(a)        KAAPI_ATOMIC_DECR(a)
#  define KAAPI_ATOMIC_ADD64(a, value)  KAAPI_ATOMIC_ADD(a, value)
#  define KAAPI_ATOMIC_SUB64(a, value)  KAAPI_ATOMIC_SUB(a, value)
#  define KAAPI_ATOMIC_AND64(a, o)      KAAPI_ATOMIC_AND(a, o)
#  define KAAPI_ATOMIC_OR64(a, o)       KAAPI_ATOMIC_OR(a,o)
#  define KAAPI_ATOMIC_XOR64(a, o)      KAAPI_ATOMIC_XOR(a,o)
#  define KAAPI_ATOMIC_AND64_ORIG(a, o) KAAPI_ATOMIC_AND_ORIG(a,o)
#  define KAAPI_ATOMIC_OR64_ORIG(a, o)  KAAPI_ATOMIC_OR_ORIG(a, o)
#  define KAAPI_ATOMIC_XOR64_ORIG(a, o) KAAPI_ATOMIC_XOR_ORIG(a, o)


#elif defined(__APPLE__) /* if gcc version on Apple is less than 4.1 */
#  warning "ON THIS ARCHITECTURE, PLEASE USE MORE RECENT GCC COMPILER (>=4.1)"
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
#  if defined(__x86_64) || defined(__i386__)
  /* not need sfence on X86 archi: write are ordered __asm__ __volatile__ ("sfence":::"memory"); */
  __asm__ __volatile__ ("":::"memory");
#  else
  OSMemoryBarrier();
#  endif
}

static inline void kaapi_readmem_barrier()  
{
#  if defined(__x86_64) || defined(__i386__)
  __asm__ __volatile__ ("":::"memory");
//  __asm__ __volatile__ ("lfence":::"memory");
#  else
  OSMemoryBarrier();
#  endif
}

/* should be both read & write barrier */
static inline void kaapi_mem_barrier()  
{
#  if defined(__x86_64) || defined(__i386__)
  /** Mac OS 10.6.8 with gcc 4.2.1 has a buggy __sync_synchronize(); 
      gcc-4.4.4 pass the test with sync_synchronize
  */
  __asm__ __volatile__ ("mfence":::"memory");
#  else
  OSMemoryBarrier();
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
#  elif defined(__GNUC__)
  __sync_synchronize();
#  else
#  error "Compiler not supported"
/* xlC ->__lwsync() / bultin */  
#  endif
}

/* should be both read & write barrier */
static inline void kaapi_mem_barrier()  
{
#  if defined(__x86_64) || defined(__i386__)
  /* not need lfence on X86 archi: read are ordered */
  __asm__ __volatile__ ("mfence":::"memory");
#  elif defined(__GNUC__)
  __sync_synchronize();
#  else
#  error "Compiler not supported"
/* bultin ?? */  
#  endif
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

//#define KAAPI_SCHED_LOCK_PTHREAD 1
//#define KAAPI_SCHED_LOCK_CAS 1
#if defined(KAAPI_DEBUG)
extern unsigned int kaapi_get_self_kid(void);
#endif

#if defined(KAAPI_SCHED_LOCK_PTHREAD)
#include <pthread.h>
typedef struct kaapi_lock_t {
  pthread_mutex_t _mutex;
} kaapi_lock_t;

static inline int kaapi_atomic_initlock( kaapi_lock_t* lock )
{
  return pthread_mutex_init(&lock->_mutex,0);
}

static inline int kaapi_atomic_destroylock( kaapi_lock_t* lock )
{
  return pthread_mutex_destroy(&lock->_mutex);
}

static inline int kaapi_atomic_trylock( kaapi_lock_t* lock )
{
  return 0 == pthread_mutex_trylock( &lock->_mutex );
}

static inline int kaapi_atomic_lock( kaapi_lock_t* lock )
{
  return pthread_mutex_lock(&lock->_mutex);
}

static inline int kaapi_atomic_unlock( kaapi_lock_t* lock )
{
  return pthread_mutex_unlock(&lock->_mutex);
}

static inline void kaapi_atomic_waitlock( kaapi_lock_t* lock)
{
  /* wait until reaches the unlocked state */
  while (!kaapi_atomic_trylock(lock))
    kaapi_slowdown_cpu();
  kaapi_atomic_unlock( lock );
}

static inline int kaapi_atomic_assertlocked( kaapi_lock_t* lock)
{
  return 1;
}

#elif defined(KAAPI_SCHED_LOCK_CAS)

typedef struct kaapi_lock_t {
  volatile int32_t  _counter;
  volatile int32_t  _sync;     /* used for fastes waitlock synchronization */
#if defined(KAAPI_DEBUG)  
  volatile uint32_t _owner;
  volatile uint32_t _unlocker;
  volatile uint32_t _magic;
#endif
} kaapi_lock_t;

#if defined(KAAPI_DEBUG)  
#  define KAAPI_LOCK_INITIALIZER { 1, 0, -1U, -1U, 123123123U }
#else
#  define KAAPI_LOCK_INITIALIZER { 0, 0 }
#endif

static inline int kaapi_atomic_initlock( kaapi_lock_t* lock )
{
  kaapi_assert_debug( lock->_magic != 123123123U);
  KAAPI_DEBUG_INST(lock->_magic = 123123123U;)
  KAAPI_DEBUG_INST(lock->_owner = -1U;)
  KAAPI_DEBUG_INST(lock->_unlocker = -1U;)
  KAAPI_DEBUG_INST(lock->_sync = 0;)
  KAAPI_ATOMIC_WRITE_BARRIER(lock,0);
  return 0;
}

static inline int kaapi_atomic_destroylock( kaapi_lock_t* lock )
{
  kaapi_assert_debug( lock->_magic == 123123123U);
  kaapi_assert_debug(lock->_owner == -1U);
  kaapi_assert_debug(lock->_unlocker != -1U);
  kaapi_assert_debug( KAAPI_ATOMIC_READ(lock) == 0 );
  KAAPI_DEBUG_INST(lock->_magic = 0101010101U;)
  return 0;
}

static inline int kaapi_atomic_trylock( kaapi_lock_t* lock )
{
  kaapi_assert_debug( lock->_magic == 123123123U);
  /* implicit barrier in KAAPI_ATOMIC_CAS if lock is taken */
  if ((KAAPI_ATOMIC_READ(lock) ==0) && KAAPI_ATOMIC_CAS(lock, 0, 1))
  {
    KAAPI_DEBUG_INST(lock->_owner = kaapi_get_self_kid();)
    KAAPI_DEBUG_INST(lock->_unlocker = -1U;)
    return 1;  
  }
  return 0;
}

static inline int kaapi_atomic_lock( kaapi_lock_t* lock )
{
  int ok;
  kaapi_assert_debug( lock->_magic == 123123123U);
  do {
    ok = (KAAPI_ATOMIC_READ(lock) ==0) && KAAPI_ATOMIC_CAS(lock, 0, 1);
    if (ok) break;
    kaapi_assert_debug( lock->_magic == 123123123U);
    kaapi_slowdown_cpu();
  } while (1);
  
  /* implicit barrier in KAAPI_ATOMIC_CAS */
  KAAPI_DEBUG_INST(lock->_owner = kaapi_get_self_kid();)
  KAAPI_DEBUG_INST(lock->_unlocker = -1U;)
  kaapi_assert_debug( KAAPI_ATOMIC_READ(lock) != 0 );
  return 0;
}

static inline int kaapi_atomic_unlock( kaapi_lock_t* lock )
{
  kaapi_assert_debug( lock->_magic == 123123123U);
  kaapi_assert_debug( lock->_unlocker == -1U);
  kaapi_assert_debug( lock->_owner == kaapi_get_self_kid() );
  kaapi_assert_debug( KAAPI_ATOMIC_READ(lock) == 1);
  KAAPI_DEBUG_INST(lock->_unlocker = lock->_owner;)
  KAAPI_DEBUG_INST(lock->_owner = -1U;)
  
  lock->_sync = 0;
  KAAPI_ATOMIC_WRITE_BARRIER(lock, 0);
  return 0;
}

static inline void kaapi_atomic_waitlock( kaapi_lock_t* lock)
{
  kaapi_assert_debug( lock->_magic == 123123123U);

  /* barrier to made visible local modification */
  kaapi_writemem_barrier();

  /* wait until reaches the unlocked state */
  lock->_sync = 1;
  while ((lock->_sync !=0) && (KAAPI_ATOMIC_READ(lock) !=0))
    kaapi_slowdown_cpu();
}

static inline int kaapi_atomic_assertlocked( kaapi_lock_t* lock)
{
  kaapi_assert_debug( lock->_magic == 123123123U);
  return KAAPI_ATOMIC_READ(lock) !=0;
}

#else // elif defined(KAAPI_SCHED_LOCK_CAS)

/** Note on scheduler lock:
  KAAPI_SCHED_LOCK_CAS -> lock state == 1 iff lock is taken, else 0
  KAAPI_SCHED_LOCK_CAS not defined: see 
    Sewell, P., Sarkar, S., Owens, S., Nardelli, F. Z., and Myreen, M. O. 2010. 
    x86-TSO: a rigorous and usable programmer's model for x86 multiprocessors. 
    Commun. ACM 53, 7 (Jul. 2010), 89-97. 
    DOI= http://doi.acm.org/10.1145/1785414.1785443
*/
typedef struct kaapi_lock_t {
  volatile int32_t  _counter;
  volatile int32_t  _sync;     /* used for fastes waitlock synchronization */
#if defined(KAAPI_DEBUG)  
  volatile uint32_t _owner;
  volatile uint32_t _unlocker;
  volatile uint32_t _magic;
#endif
} kaapi_lock_t;

#if defined(KAAPI_DEBUG)  
#  define KAAPI_LOCK_INITIALIZER { 1, 0, -1U, -1U, 123123123U }
#else
#  define KAAPI_LOCK_INITIALIZER { 1, 0 }
#endif

static inline int kaapi_atomic_initlock( kaapi_lock_t* lock )
{
  kaapi_assert_debug( lock->_magic != 123123123U);
  KAAPI_DEBUG_INST(lock->_magic = 123123123U;)
  KAAPI_DEBUG_INST(lock->_owner = -1U;)
  KAAPI_DEBUG_INST(lock->_unlocker = -1U;)
  lock->_sync = 0;
  KAAPI_ATOMIC_WRITE_BARRIER(lock,1);
  return 0;
}

static inline int kaapi_atomic_destroylock( kaapi_lock_t* lock )
{
  kaapi_assert_debug( KAAPI_ATOMIC_READ(lock) == 1 );
  kaapi_assert_debug( lock->_magic == 123123123U);
  kaapi_assert_debug( lock->_owner == -1U);
  KAAPI_DEBUG_INST(lock->_magic = 0101010101U;)
  return 0;
}

static inline int kaapi_atomic_trylock( kaapi_lock_t* lock )
{
  kaapi_assert_debug( lock->_magic == 123123123U);
  if ((KAAPI_ATOMIC_READ(lock) ==1) && (KAAPI_ATOMIC_DECR(lock) ==0))
  {
    KAAPI_DEBUG_INST(lock->_owner = kaapi_get_self_kid();)
    KAAPI_DEBUG_INST(lock->_unlocker = -1U;)
    return 1;
  }
  return 0;
}

static inline int kaapi_atomic_lock( kaapi_lock_t* lock )
{
  kaapi_assert_debug( lock->_magic == 123123123U);
acquire:
  if (KAAPI_ATOMIC_DECR(lock) ==0) 
  {
    KAAPI_DEBUG_INST(lock->_owner = kaapi_get_self_kid();)
    KAAPI_DEBUG_INST(lock->_unlocker = -1U;)
    return 0;
  }
  while (KAAPI_ATOMIC_READ(lock) <=0)
  {
    kaapi_assert_debug( lock->_magic == 123123123U);
    kaapi_slowdown_cpu();
  }
  goto acquire;
}


static inline int kaapi_atomic_unlock( kaapi_lock_t* lock )
{
  kaapi_assert_debug( lock->_magic == 123123123U);
  kaapi_assert_debug( lock->_unlocker == -1U);
  kaapi_assert_debug( lock->_owner == kaapi_get_self_kid() );
  kaapi_assert_debug( KAAPI_ATOMIC_READ(lock) <= 0);
  KAAPI_DEBUG_INST(lock->_unlocker = lock->_owner;)
  KAAPI_DEBUG_INST(lock->_owner = -1U;)

  if (lock->_sync !=0) 
    lock->_sync = 0;
  kaapi_mem_barrier();
  KAAPI_ATOMIC_WRITE(lock, 1);
  return 0;
}

static inline void kaapi_atomic_waitlock( kaapi_lock_t* lock)
{
  kaapi_assert_debug( lock->_magic == 123123123U);
  
  lock->_sync = 1;

  /* barrier to made visible local modification before reading lock->_sync */
  kaapi_mem_barrier();

  /* wait until reaches the unlocked state */
  while ((lock->_sync !=0) && (KAAPI_ATOMIC_READ(lock) <=0))
    kaapi_slowdown_cpu();
}

static inline int kaapi_atomic_assertlocked( kaapi_lock_t* lock)
{
  kaapi_assert_debug( lock->_magic == 123123123U);
  return KAAPI_ATOMIC_READ(lock) <=0;
}

#endif  // #else // if defined(KAAPI_SCHED_LOCK_CAS)



#if (__SIZEOF_POINTER__ == 4)
#  define KAAPI_ATOMIC_CASPTR(a, o, n) \
    KAAPI_ATOMIC_CAS( (kaapi_atomic_t*)a, (uint32_t)o, (uint32_t)n )
#  define KAAPI_ATOMIC_ORPTR_ORIG(a, v) \
    KAAPI_ATOMIC_OR_ORIG( (kaapi_atomic_t*)a, (uint32_t)v)
#  define KAAPI_ATOMIC_ANDPTR_ORIG(a, v) \
    KAAPI_ATOMIC_AND_ORIG( (kaapi_atomic_t*)a, (uint32_t)v)
#  define KAAPI_ATOMIC_WRITEPTR_BARRIER(a, v) \
    KAAPI_ATOMIC_WRITE_BARRIER( (kaapi_atomic_t*)a, (uint32_t)v)

#elif (__SIZEOF_POINTER__ == 8)
#  define KAAPI_ATOMIC_CASPTR(a, o, n) \
    KAAPI_ATOMIC_CAS64( (kaapi_atomic64_t*)(a), (uint64_t)o, (uint64_t)n )
#  define KAAPI_ATOMIC_ORPTR_ORIG(a, v) \
    KAAPI_ATOMIC_OR64_ORIG( (kaapi_atomic64_t*)(a), (uint64_t)v)
#  define KAAPI_ATOMIC_ANDPTR_ORIG(a, v) \
    KAAPI_ATOMIC_AND64_ORIG( (kaapi_atomic64_t*)(a), (uint64_t)v)
#  define KAAPI_ATOMIC_WRITEPTR_BARRIER(a, v) \
    KAAPI_ATOMIC_WRITE_BARRIER( (kaapi_atomic64_t*)a, (uint64_t)v)

#else
#  error "No implementation for pointer to function with size greather than 8 bytes. Please contact the authors."
#endif


#ifdef __cplusplus
}
#endif

#endif
