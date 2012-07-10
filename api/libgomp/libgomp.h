/*
** xkaapi
** 
**
** Copyright 2009 INRIA.
**
** Contributors :
**
** thierry.gautier@inrialpes.fr
** francois.broquedis@imag.fr
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
#ifndef _KAAPI_LIBGOMP_
#define _KAAPI_LIBGOMP_

#include <stdbool.h>
#include <stdio.h>
#include <stdint.h>

#if defined(__linux__)
#include <omp.h>
#else
/* function from the runtime support of OMP: need for mac... */
typedef enum omp_sched_t {
    omp_sched_static = 1,
    omp_sched_dynamic = 2,
    omp_sched_guided = 3,
    omp_sched_auto = 4
} omp_sched_t;

//typedef kaapi_atomic8_t omp_lock_t;
//typedef kaapi_atomic32_t omp_nest_lock_t;
#endif

#include "kaapi_impl.h"
#include "kaapic_impl.h"

#ifdef HAVE_VISIBILITY_HIDDEN
# pragma GCC visibility push(hidden)
#endif
/* all things defined in this visibility section are private to our library */

/* barrier.c */

#define BAR_CYCLES 3
#define CACHE_LINE_SIZE 64

typedef struct komp_barrier {
  kaapi_atomic_t __attribute__ ((aligned (CACHE_LINE_SIZE))) cycle;
  unsigned int __attribute__ ((aligned (CACHE_LINE_SIZE))) nthreads;
  char __attribute__ ((aligned (CACHE_LINE_SIZE))) count[BAR_CYCLES * CACHE_LINE_SIZE];
} komp_barrier_t; 

struct kompctxt_t;
struct komp_workshare_t;
typedef kaapic_global_work_t komp_globalworkshare_t;


/* init.c */
void komp_barrier_init (struct komp_barrier *barrier, unsigned int num);
void komp_barrier_destroy (struct komp_barrier *barrier);
void komp_barrier_wait (struct kompctxt_t* ctxt, struct komp_barrier *barrier);

extern unsigned long komp_env_nthreads;

/* each task owns its icv instance.
   The following fields are inherited at task creation time:
   - [read and report OpenMP 3.1 spec definition]
*/
typedef struct komp_icv_t {
  int                thread_id;       /* thread id */
  int                next_numthreads; /* number of thread for the next // region */
  int                nested_level;    /* nested level of // region */
  int                nested_parallel; /* !=0 iff nest allowed */
  int                dynamic_numthreads;  /* number of threads the runtime can dynamically
                                             adjust the next parallel region to. */
  omp_sched_t        run_sched;
  int                chunk_size;
#if defined(KAAPI_USE_FOREACH_WITH_DATADISTRIBUTION)
  kaapic_foreach_attr_t attr;         /* attribut for the next foreach loop */
#endif
} komp_icv_t;

/* Team information : common to all threads that shared a same parallel region.
   - lock is in case of concurrency on global team access update
   - barrier is the barrier that is used by all tasks in the parallel region
   - sing_state is the state to implement #pragma omp single construct
   - numthreads is the number of threads in the parallel region. Fix at the 
   start of the parallel region.
   - the global workshare is initialized in the workshare construct (parallel loop)
   
   The team information data created by a the master thread that encounters 
   a #pragma omp parallel construct. Parallel region has a lexical scope 
   that allows to store the team info data into the Kaapi's stack 
   of the runing Kaapi thread.
*/
typedef struct komp_teaminfo_t {
  komp_barrier_t                   barrier;
  int volatile                     current_ordered_index;
  void*  volatile                  single_data;  /* 0 or the & of copy_end */
  unsigned int volatile            section_state;
  unsigned int volatile            ordered_state;  
  int                              numthreads;
  komp_globalworkshare_t* volatile gwork;      /* last foreach loop context */
  unsigned long                    serial;      /* serial number of workshare */
 } komp_teaminfo_t;


/* Workshare structure: it defines loop work for all threads in a team
   - each thread owns its proper state defined by a komp_workshare_t data
   - the Kaapi iterates over [0,N) with increment=1, start and incr is used
   to convert [0,N) to [start,end) + incr ginve in for loop workshare construct.
   - the serial number is used to defined which is the current workshare.
*/
typedef struct komp_workshare_t {
  kaapic_local_work_t*         lwork;  /* last foreach loop context */
  union {
    struct {
      long                     start;  /* start index of the Kaapi/GOMP slice*/
      long                     incr;   /* scaling factor between Kaapi/GOMP */
    } li;
    struct {
      unsigned long long       start;  /* start index of the Kaapi/GOMP slice*/
      unsigned long long       incr;   /* scaling factor between Kaapi/GOMP */
      bool                     up;     /* upward / downward count */
    } ull;
  } rep;
  int                          cur_start;
  int                          cur_end;
  unsigned long                serial; /* serial number of workshare construct */
} komp_workshare_t;


/** Pre-historic TLS  support for OMP */
typedef struct kompctxt_t {
  komp_workshare_t*   workshare;     /* local view of team workshare   */
  komp_teaminfo_t*    teaminfo;      /* global team information */
  komp_icv_t          icv;           /* current icv data */
  int                 inside_single;
  struct kompctxt_t*  save_ctxt;     /* to restore on pop */
} kompctxt_t;

/* omp_max_active_levels:
   OpenMP 3.1: 
   "This routine has the described effect only when called from 
   the sequential part of the program. When called from within 
   an explicit parallel region, the effect of this routine is 
   implementation defined."
 */
extern int omp_max_active_levels;


/** Initial context with teaminformation */
typedef struct kompctxt_first_t {
  kompctxt_t      ctxt;
  komp_teaminfo_t teaminfo; /* sequential thread */
} kompctxt_first_t;


static inline kompctxt_t* komp_get_ctxtkproc( kaapi_processor_t* kproc )
{ 
  if (kproc->libkomp_tls == 0)
  {
    kompctxt_first_t* first = (kompctxt_first_t*)malloc(sizeof(kompctxt_first_t));
    first->ctxt.workshare               = 0;
    first->ctxt.teaminfo                = &first->teaminfo;
    first->ctxt.icv.thread_id           = 0;
    first->ctxt.icv.next_numthreads     = kaapi_getconcurrency();
    first->ctxt.icv.nested_level        = 0;
    first->ctxt.icv.nested_parallel     = 0;
    first->ctxt.icv.dynamic_numthreads  = 0; /* Not sure of this initial value, next_numthreads may
                                     					  be more appropriate here... */
    first->ctxt.icv.run_sched           = omp_sched_dynamic;
    first->ctxt.icv.chunk_size          = 0; /* default */
#if defined(KAAPI_USE_FOREACH_WITH_DATADISTRIBUTION)
    kaapic_foreach_attr_init( &first->ctxt.icv.attr );
#endif
    komp_barrier_init (&first->teaminfo.barrier, 1);
    first->teaminfo.ordered_state       = 0;
    first->teaminfo.single_data = 0;
    first->teaminfo.numthreads  = 1;
    first->teaminfo.gwork       = 0;
    first->teaminfo.serial      = 0;
    kproc->libkomp_tls            = &first->ctxt;
    return &first->ctxt;
  }
  return (kompctxt_t*)kproc->libkomp_tls;
}

static inline kompctxt_t* komp_get_ctxt()
{
  return komp_get_ctxtkproc(kaapi_get_current_processor());
}

/* */
extern komp_teaminfo_t*  komp_init_parallel_start (
  kaapi_processor_t* kproc,
  unsigned num_threads
);

/* */
extern void komp_parallel_start (
  void (*fn) (void *), 
  void *data, 
  unsigned num_threads
);


/* going back to the previous visibility, ie "default" */
#ifdef HAVE_VISIBILITY_HIDDEN
# pragma GCC visibility pop
#endif


#include "libgomp_g.h"

#ifdef HAVE_VERSION_SYMBOL
extern void komp_init_lock_30 (omp_lock_t *) __GOMP_NOTHROW;
extern void komp_destroy_lock_30 (omp_lock_t *) __GOMP_NOTHROW;
extern void komp_set_lock_30 (omp_lock_t *) __GOMP_NOTHROW;
extern void komp_unset_lock_30 (omp_lock_t *) __GOMP_NOTHROW;
extern int  komp_test_lock_30 (omp_lock_t *) __GOMP_NOTHROW;
extern void komp_init_nest_lock_30 (omp_nest_lock_t *) __GOMP_NOTHROW;
extern void komp_destroy_nest_lock_30 (omp_nest_lock_t *) __GOMP_NOTHROW;
extern void komp_set_nest_lock_30 (omp_nest_lock_t *) __GOMP_NOTHROW;
extern void komp_unset_nest_lock_30 (omp_nest_lock_t *) __GOMP_NOTHROW;
extern int  komp_test_nest_lock_30 (omp_nest_lock_t *) __GOMP_NOTHROW;

#if 0 /* not yet implemented */
extern void komp_init_lock_25 (omp_lock_25_t *) __GOMP_NOTHROW;
extern void komp_destroy_lock_25 (omp_lock_25_t *) __GOMP_NOTHROW;
extern void komp_set_lock_25 (omp_lock_25_t *) __GOMP_NOTHROW;
extern void komp_unset_lock_25 (omp_lock_25_t *) __GOMP_NOTHROW;
extern int  komp_test_lock_25 (omp_lock_25_t *) __GOMP_NOTHROW;
extern void komp_init_nest_lock_25 (omp_nest_lock_25_t *) __GOMP_NOTHROW;
extern void komp_destroy_nest_lock_25 (omp_nest_lock_25_t *) __GOMP_NOTHROW;
extern void komp_set_nest_lock_25 (omp_nest_lock_25_t *) __GOMP_NOTHROW;
extern void komp_unset_nest_lock_25 (omp_nest_lock_25_t *) __GOMP_NOTHROW;
extern int  komp_test_nest_lock_25 (omp_nest_lock_25_t *) __GOMP_NOTHROW;
#endif

# define strong_alias(fn, al) \
  extern __typeof (fn) al __attribute__ ((alias (#fn)));
# define komp_lock_symver30(fn) \
  __asm (".symver k" #fn "_30, " #fn "@@OMP_3.0");
# define komp_lock_symver(fn) \
  komp_lock_symver30(fn) \
  __asm (".symver k" #fn "_25, " #fn "@OMP_1.0");

#else
# define komp_init_lock_30 omp_init_lock
# define komp_destroy_lock_30 omp_destroy_lock
# define komp_set_lock_30 omp_set_lock
# define komp_unset_lock_30 omp_unset_lock
# define komp_test_lock_30 omp_test_lock
# define komp_init_nest_lock_30 omp_init_nest_lock
# define komp_destroy_nest_lock_30 omp_destroy_nest_lock
# define komp_set_nest_lock_30 omp_set_nest_lock
# define komp_unset_nest_lock_30 omp_unset_nest_lock
# define komp_test_nest_lock_30 omp_test_nest_lock

#define strong_alias(fn, al)
#define komp_lock_symver30(fn)
#define komp_lock_symver(fn)

#endif


__attribute__((weak))
extern void komp_set_datadistribution_bloccyclic( unsigned long long size, unsigned int length );

extern void omp_set_num_threads (int);
extern int omp_get_num_threads (void);
extern int omp_get_thread_num (void);
extern int omp_get_max_threads (void);
extern int omp_get_num_procs (void);
extern int omp_in_parallel (void);
extern void omp_set_dynamic (int);
extern int omp_get_dynamic (void);
extern void omp_set_nested (int);
extern int omp_get_nested (void);
extern void omp_set_schedule (omp_sched_t, int);
extern void omp_get_schedule (omp_sched_t *, int *);
extern int omp_get_thread_limit (void);
extern void omp_set_max_active_levels (int);
extern int omp_get_max_active_levels (void);
extern int omp_get_level (void);
extern int omp_get_ancestor_thread_num (int);
extern int omp_get_team_size (int);
extern int omp_get_active_level (void);
extern int omp_in_final (void);
extern double omp_get_wtime (void);
extern double omp_get_wtick (void);

#endif // #ifndef _KAAPI_LIBGOMP_
