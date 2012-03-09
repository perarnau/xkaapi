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

#include <omp.h>

#include "kaapi_impl.h"
#include "kaapic_impl.h"

#ifdef HAVE_VISIBILITY_HIDDEN
# pragma GCC visibility push(hidden)
#endif
/* all things defined in this visibility section are private to our library */

/* barrier.c */

#define BAR_CYCLES 3
#define CACHE_LINE_SIZE 64

typedef struct gomp_barrier {
  kaapi_atomic_t __attribute__ ((aligned (CACHE_LINE_SIZE))) cycle;
  unsigned int __attribute__ ((aligned (CACHE_LINE_SIZE))) nthreads;
  char __attribute__ ((aligned (CACHE_LINE_SIZE))) count[BAR_CYCLES * CACHE_LINE_SIZE];
} komp_barrier_t; 


struct PerTeamLocalStorage;
struct WorkShareRep;

/* Team information : common to all threads that shared a same parallel region 
   - localinfo is set to 0 at creation time
   - localinfo[tid] = &workshare of the thread tid is set in the trampoline task
   used to initialize task contexte.
   - the workshare is initialized in the workshare construct (parallel loop)
*/
typedef struct GlobalTeamInformation {
  kaapi_lock_t                 lock;       /* 1 iff work is init */
  int                          numthreads;
  kaapi_atomic_t               single_state;
  komp_barrier_t               barrier;
  kaapic_global_work_t*        volatile gwork;      /* last foreach loop context */
} kaapi_libkomp_teaminfo_t;


/* Workshare structure
*/
typedef struct WorkShareRep {
  kaapic_local_work_t*         lwork;      /* last foreach loop context */
  long                         incr;       /* scaling factor between Kaapi/GOMP slice*/
  int                          workload;   /* workload */
} kaapi_libkompworkshared_t;


/** Pre-historic TLS  support for OMP */
typedef struct PerTeamLocalStorage {
  kaapi_libkompworkshared_t    workshare;     /* team workshare */
  kaapi_libkomp_teaminfo_t*    teaminfo;      /* team information */
  
  int                          threadid;      /* current thread identifier */
  int                          numthreads;    /* number of threads in the parallel region */
  int                          nextnumthreads;/* number of threads for the next // region */
  int                          inside_single;
} kaapi_libkompctxt_t ;


static inline kaapi_libkompctxt_t* komp_get_ctxtkproc( kaapi_processor_t* kproc )
{ 
  if (kproc->libgomp_tls == 0)
  {
    kaapi_libkompctxt_t* ctxt = (kaapi_libkompctxt_t*)malloc(sizeof(kaapi_libkompctxt_t));
    ctxt->workshare.lwork = 0;
    ctxt->teaminfo        = 0;
    ctxt->threadid        = 0;
    ctxt->numthreads      = 1;
    ctxt->nextnumthreads  = kaapi_getconcurrency();
    kproc->libgomp_tls    = ctxt;
    return ctxt;
  }
  return (kaapi_libkompctxt_t*)kproc->libgomp_tls;
}

static inline kaapi_libkompctxt_t* komp_get_ctxt()
{
  return komp_get_ctxtkproc(kaapi_get_current_processor());
}

extern
kaapi_libkomp_teaminfo_t*  komp_init_parallel_start (
  kaapi_processor_t* kproc,
  unsigned num_threads
);

enum omp_task_kind
{
  GOMP_TASK_IMPLICIT,
  GOMP_TASK_IFFALSE,
  GOMP_TASK_WAITING,
  GOMP_TASK_TIED
};


/* init.c */

void gomp_barrier_init (struct gomp_barrier *barrier, unsigned int num);
void gomp_barrier_destroy (struct gomp_barrier *barrier);
void gomp_barrier_wait (struct gomp_barrier *barrier);


/* going back to the previous visibility, ie "default" */
#ifdef HAVE_VISIBILITY_HIDDEN
# pragma GCC visibility pop
#endif

#include "libgomp_g.h"

#ifdef HAVE_VERSION_SYMBOL
extern void gomp_init_lock_30 (omp_lock_t *) __GOMP_NOTHROW;
extern void gomp_destroy_lock_30 (omp_lock_t *) __GOMP_NOTHROW;
extern void gomp_set_lock_30 (omp_lock_t *) __GOMP_NOTHROW;
extern void gomp_unset_lock_30 (omp_lock_t *) __GOMP_NOTHROW;
extern int gomp_test_lock_30 (omp_lock_t *) __GOMP_NOTHROW;
extern void gomp_init_nest_lock_30 (omp_nest_lock_t *) __GOMP_NOTHROW;
extern void gomp_destroy_nest_lock_30 (omp_nest_lock_t *) __GOMP_NOTHROW;
extern void gomp_set_nest_lock_30 (omp_nest_lock_t *) __GOMP_NOTHROW;
extern void gomp_unset_nest_lock_30 (omp_nest_lock_t *) __GOMP_NOTHROW;
extern int gomp_test_nest_lock_30 (omp_nest_lock_t *) __GOMP_NOTHROW;

#if 0 /* not yet implemented */
extern void gomp_init_lock_25 (omp_lock_25_t *) __GOMP_NOTHROW;
extern void gomp_destroy_lock_25 (omp_lock_25_t *) __GOMP_NOTHROW;
extern void gomp_set_lock_25 (omp_lock_25_t *) __GOMP_NOTHROW;
extern void gomp_unset_lock_25 (omp_lock_25_t *) __GOMP_NOTHROW;
extern int gomp_test_lock_25 (omp_lock_25_t *) __GOMP_NOTHROW;
extern void gomp_init_nest_lock_25 (omp_nest_lock_25_t *) __GOMP_NOTHROW;
extern void gomp_destroy_nest_lock_25 (omp_nest_lock_25_t *) __GOMP_NOTHROW;
extern void gomp_set_nest_lock_25 (omp_nest_lock_25_t *) __GOMP_NOTHROW;
extern void gomp_unset_nest_lock_25 (omp_nest_lock_25_t *) __GOMP_NOTHROW;
extern int gomp_test_nest_lock_25 (omp_nest_lock_25_t *) __GOMP_NOTHROW;
#endif

# define strong_alias(fn, al) \
  extern __typeof (fn) al __attribute__ ((alias (#fn)));
# define komp_lock_symver30(fn) \
  __asm (".symver g" #fn "_30, " #fn "@@OMP_3.0");
# define komp_lock_symver(fn) \
  komp_lock_symver30(fn) \
  __asm (".symver g" #fn "_25, " #fn "@OMP_1.0");

#else
# define gomp_init_lock_30 omp_init_lock
# define gomp_destroy_lock_30 omp_destroy_lock
# define gomp_set_lock_30 omp_set_lock
# define gomp_unset_lock_30 omp_unset_lock
# define gomp_test_lock_30 omp_test_lock
# define gomp_init_nest_lock_30 omp_init_nest_lock
# define gomp_destroy_nest_lock_30 omp_destroy_nest_lock
# define gomp_set_nest_lock_30 omp_set_nest_lock
# define gomp_unset_nest_lock_30 omp_unset_nest_lock
# define gomp_test_nest_lock_30 omp_test_nest_lock
#endif

#endif // #ifndef _KAAPI_LIBGOMP_
