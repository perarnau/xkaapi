/*
** kaapi.h.in
** ckaapi
** 
** Created on Tue Mar 31 15:19:14 2009
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
#ifndef _KAAPI_H
#define _KAAPI_H 1

#if defined(__cplusplus)
extern "C" {
#endif

#ifdef __APPLE__
#  include <time.h>
#endif

#include <sched.h> // for sched_param type.
#include "kaapi_type.h"

/* ============================== Cpu_set Mess ============================== */

// TODO : DUPLICATE!!!
#ifndef HAVE_CPU_SET_T
#define HAVE_CPU_SET_T
#ifndef __cpu_set_t_defined
#include <strings.h> /* for bzero in cpu set */

/* Size definition for CPU sets.  */
#  define __CPU_SETSIZE	64
#  define __NCPUBITS	(8 * sizeof (__cpu_mask))
  
/* Type for array elements in 'cpu_set_t'.  */
typedef unsigned long int __cpu_mask;
  
/* Data structure to describe CPU mask.  */
typedef struct {
  __cpu_mask __bits[__CPU_SETSIZE / __NCPUBITS];
} cpu_set_t;
  
#  define CPU_ZERO( pcpuset ) bzero( pcpuset, sizeof(cpu_set_t) )
  
#  define CPU_SET( cpu, pcpuset ) (pcpuset)->__bits[ cpu / __NCPUBITS] |= \
(1UL << (cpu -(cpu / __NCPUBITS)*__NCPUBITS))

#  define CPU_CLR( cpu, pcpuset ) (pcpuset)->__bits[ cpu / __NCPUBITS] &= \
~(1UL << (cpu -(cpu / __NCPUBITS)*__NCPUBITS))

#  define CPU_ISSET( cpu, pcpuset ) (pcpuset)->__bits[ cpu / __NCPUBITS] & \
(1UL << (cpu -(cpu / __NCPUBITS)*__NCPUBITS))

#endif
#endif

/* ========================================================================== */



/* ================================ Constants =============================== */

/* Define the number of key destructor iterations. */
#define KAAPI_DESTRUCTOR_ITERATIONS 4


/* Define the minimum stack size. */
#define KAAPI_STACK_MIN 8192


/* Define the number of keys. */
#define KAAPI_KEYS_MAX 512


/* Define the cache line size. */
#define KAAPI_CACHE_LINE 64

/* ========================================================================== */


/* =============================== Initializer ============================== */

#define KAAPI_MUTEX_INITIALIZER {{ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 88, 84, 85, 77, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 32, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}}

#define KAAPI_COND_INITIALIZER {{ 88, 84, 85, 77, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 32, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}}


/* ========================================================================== */


/* ================================== Types ================================= */

/* ========================================================================== */
/** Mutex interface
 */

typedef enum kaapi_mutex_type_t {
  KAAPI_MUTEX_NORMAL    =0,
  KAAPI_MUTEX_RECURSIVE =1
}kaapi_mutex_type_t;

/* ========================================================================== */
/** Thread descriptor
 */

typedef enum kaapi_scope_t {
  KAAPI_SYSTEM_SCOPE  = 0,
  KAAPI_PROCESS_SCOPE = 1
} kaapi_scope_t;
  
typedef struct kaapi_thread_descr_t* kaapi_t;
    
/* ========================================================================== */
/** Dataspecific data structures and functions
 */

typedef unsigned long kaapi_key_t;
  
/* ========================================================================== */
/** Once data structure: same structure as atomic_t
 */

#define KAAPI_ONCE_INIT {{ 0, 0, 0, 0}}

  
/* ========================================================================== */



/* ============================= POSIX Interface ============================ */

/* ========================================================================== */
/** Mutex interface
 */

int kaapi_mutexattr_destroy (kaapi_mutexattr_t *);

int kaapi_mutexattr_init (kaapi_mutexattr_t *);

int kaapi_mutexattr_gettype (const kaapi_mutexattr_t *__restrict, int *__restrict);

int kaapi_mutexattr_settype (kaapi_mutexattr_t *, int);

int kaapi_mutex_destroy (kaapi_mutex_t *);

int kaapi_mutex_init (kaapi_mutex_t *__restrict, const kaapi_mutexattr_t *__restrict);

int kaapi_mutex_lock (kaapi_mutex_t *);

int kaapi_mutex_trylock (kaapi_mutex_t *);

int kaapi_mutex_unlock (kaapi_mutex_t *);

/* ========================================================================== */
/** Condition interface 
 */

int kaapi_condattr_init (kaapi_condattr_t *);

int kaapi_condattr_destroy (kaapi_condattr_t *);

int kaapi_cond_broadcast (kaapi_cond_t *);

int kaapi_cond_destroy (kaapi_cond_t *);

int kaapi_cond_init (kaapi_cond_t *__restrict, const kaapi_condattr_t *__restrict);

int kaapi_cond_signal (kaapi_cond_t *);

int kaapi_cond_timedwait (kaapi_cond_t *__restrict, kaapi_mutex_t *__restrict,
                          const struct timespec *__restrict);

int kaapi_cond_wait (kaapi_cond_t *__restrict, kaapi_mutex_t *__restrict);

/* ========================================================================== */
/** Dataspecific interface
 */

int kaapi_key_create (kaapi_key_t *, void (*)(void*));

int kaapi_key_delete (kaapi_key_t);

void *kaapi_getspecific (kaapi_key_t);

int kaapi_setspecific (kaapi_key_t, const void *);

/* ========================================================================== */
/** Thread interface
 */

int kaapi_create (kaapi_t *__restrict, const kaapi_attr_t *__restrict,
                  void *(*)(void*), void *__restrict);

int kaapi_detach (kaapi_t);

int kaapi_equal (kaapi_t, kaapi_t);

int kaapi_exit (void *);

int kaapi_join (kaapi_t, void **);

int kaapi_once (kaapi_once_t *, void (*)(void));

kaapi_t kaapi_self (void);

/* Not in POSIX spec anymore : */
void kaapi_yield (void);

int kaapi_attr_destroy (kaapi_attr_t *);

int kaapi_attr_getinheritsched (const kaapi_attr_t *__restrict, int *__restrict);

int kaapi_attr_getschedparam (const kaapi_attr_t *__restrict,
                              struct sched_param *__restrict);

int kaapi_attr_getschedpolicy (const kaapi_attr_t *__restrict, int *__restrict);

int kaapi_attr_getscope (const kaapi_attr_t *__restrict, int *__restrict);

int kaapi_attr_getstack (const kaapi_attr_t *__restrict, void **__restrict,
                         size_t *__restrict);

int kaapi_attr_getstacksize (const kaapi_attr_t *__restrict, size_t *__restrict);

/* Not in POSIX spec anymore : */
int kaapi_attr_getstackaddr (const kaapi_attr_t *__restrict, void **__restrict);

int kaapi_attr_getdetachstate (const kaapi_attr_t *, int *);

/* Not in POSIX : */
int kaapi_attr_getaffinity (kaapi_attr_t *, size_t, cpu_set_t *);

int kaapi_attr_init (kaapi_attr_t *);

int kaapi_attr_setinheritsched (kaapi_attr_t *, int);

int kaapi_attr_setschedparam (kaapi_attr_t *__restrict,
                              const struct sched_param *__restrict);

int kaapi_attr_setschedpolicy (kaapi_attr_t *, int);

int kaapi_attr_setscope (kaapi_attr_t *, int);

int kaapi_attr_setstack (kaapi_attr_t *, void *, size_t);

int kaapi_attr_setstacksize (kaapi_attr_t *, size_t);

/* Not in POSIX spec anymore : */
int kaapi_attr_setstackaddr (kaapi_attr_t *, void *);

int kaapi_attr_setdetachstate (kaapi_attr_t *, int);

/* Not in POSIX : */
int kaapi_attr_setaffinity (kaapi_attr_t *, size_t, const cpu_set_t *);

/* ========================================================================== */

/** Get the workstealing concurrency number, i.e. the number of kernel
 activities to execute the user level thread. If kaapi_setconcurrency was no
 called before then return 0, else return the number set by
 kaapi_setconcurrency.
 */
int kaapi_getconcurrency (void);

/** Set the workstealing concurrency number, i.e. the number of kernel
 activities to execute the user level thread.
 */
int kaapi_setconcurrency (int concurrency);

#ifdef __cplusplus
}
#endif

#endif // _KAAPI_H
