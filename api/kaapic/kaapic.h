/*
 ** xkaapi
 ** 
 ** Copyright 2009 INRIA.
 **
 ** Contributors :
 ** thierry.gautier@inrialpes.fr
 ** fabien.lementec@imag.fr
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
#ifndef KAAPIC_H_INCLUDED
# define KAAPIC_H_INCLUDED

#include <stddef.h>
#include <stdint.h>
#include "kaapi.h"
#include "kaapi_cpuset.h"

#if defined(__cplusplus)
extern "C" {
#endif

/* This is the portable C interface the kaapi runtime */
enum kaapic_type
{
  KAAPIC_TYPE_CHAR = 0,
  KAAPIC_TYPE_INT,
  KAAPIC_TYPE_REAL,
  KAAPIC_TYPE_FLOAT=KAAPIC_TYPE_REAL,
  KAAPIC_TYPE_DOUBLE,
  KAAPIC_TYPE_PTR,
  KAAPIC_TYPE_ID /* for FORTRAN only */
};

enum kaapic_mode
{
  KAAPIC_MODE_R = 0,
  KAAPIC_MODE_W,
  KAAPIC_MODE_RW,
  KAAPIC_MODE_CW,
  KAAPIC_MODE_V
};

enum kaapic_init_flag {
  KAAPIC_START_ALL = 0,
  KAAPIC_START_ONLY_MAIN = 1
};

/** kaapic_init INIT initializes the runtime. 
  It must be called once per pro- gram before using any of 
  the other routines. 
  If successful, there must be a corresponding 
  kaapic_finalize at the end of the program.

  \param flags: if not zero, start only the main thread 
  to avoid disturbing the execution until tasks are actually scheduled. 
  The other threads are suspended waiting for a parallel 
  region to be entered (refer to KAAPIF BEGIN PARALLEL)
*/
extern int kaapic_init(int32_t flags);

/** Finalizes the Kaapi runtime.
*/
extern int kaapic_finalize(void);

/** Returns the current time in micro-second.
*/
extern double kaapic_get_time(void);

/** Returns the number of parallel thread available to the X-Kaapi runtime
*/
extern int32_t kaapic_get_concurrency(void);

/* Returns the current thread identifier. 
   Note it should only be called in the context of a X-Kaapi thread, else
   -1 is returned.
*/
extern int32_t kaapic_get_thread_num(void);


/* Internal type, not documented to the user
*/
typedef enum {
  KAAPIC_DATADIST_VOID       = 0,
  KAAPIC_DATADIST_BLOCCYCLIC = 1
} _kaapic_foreach_attr_datadist_type_t;

/* Internal type, not documented to the user
*/
typedef struct _kaapic_foreach_attr_datadist_t {
  _kaapic_foreach_attr_datadist_type_t type;      /* discriminant for the distribution */
  union {
    struct { /* information about bloc-cyclic distribution of data */
      unsigned long long size;
      unsigned int  length; 
    } bloccyclic;
  } dist;
} _kaapic_foreach_attr_datadist_t;

/*
*/
typedef struct kaapic_foreach_attr_t {
  union {
    struct {
      long                 s_grain;
      long                 p_grain;  
    } li;
    struct { /* ull case */
      unsigned long long   s_grain;
      unsigned long long   p_grain;  
    } ull;
  }                    rep;
  _kaapic_foreach_attr_datadist_t datadist;
  unsigned int         nthreads;  /* number of threads for initial splitting */
  int                  policy;    /* choose the policy for splitting */
  kaapi_cpuset_t       threadset; /* thread set used for initial distribution i = kid */
} kaapic_foreach_attr_t;
  
/*
*/
extern int kaapic_foreach_attr_init(kaapic_foreach_attr_t* attr);

/*
*/
extern int kaapic_foreach_attr_set_grains(
  kaapic_foreach_attr_t* attr, 
  long s_grain,
  long p_grain
);

/*
*/
extern int kaapic_foreach_attr_set_grains_ull(
  kaapic_foreach_attr_t* attr, 
  unsigned long long s_grain,
  unsigned long long p_grain
);

extern int kaapic_foreach_attr_set_threads(
  kaapic_foreach_attr_t* attr, 
  unsigned int nthreads
);

/* Specify the distribution of initial range [0,N) over the NUMA node
*/
extern int kaapic_foreach_attr_set_bloccyclic_datadistribution(
  kaapic_foreach_attr_t* attr, 
  unsigned long long blocsize,
  unsigned int cyclelength
);


/*
*/
static inline int kaapic_foreach_attr_destroy(kaapic_foreach_attr_t* attr)
{ return 0; }

/* See documentation
*/
extern void kaapic_foreach( 
  int32_t first, 
  int32_t last, 
  const kaapic_foreach_attr_t* attr,
  int32_t nparam, 
  /* void (*body)(int32_t, int32_t, int32_t, ...), */
  ...
);


/* See documentation
*/
extern void kaapic_foreach_ull( 
  unsigned long long first, 
  unsigned long long last, 
  const kaapic_foreach_attr_t* attr,
  int32_t nparam, 
  /* void (*body)(unsigned long long, unsigned long long, int32_t, ...), */
  ...
);


/* See documentation
*/
extern void kaapic_foreach_with_format(
  int32_t first, 
  int32_t last, 
  const kaapic_foreach_attr_t* attr,
  int32_t nparam, 
  /* void (*body)(int32_t, int32_t, int32_t, ...), */
  ...
);

extern void kaapic_foreach_with_format_ull(
  unsigned long long first, 
  unsigned long long last, 
  const kaapic_foreach_attr_t* attr,
  int32_t nparam, 
  /* void (*body)(unsigned long long, unsigned long long, int32_t, ...), */
  ...
);

/* Allocate a data with the same scope as a task, ie. in the Kaapi'thread stack
*/
extern void* kaapic_alloca( size_t sz );


/*
*/
typedef struct kaapic_spawn_attr_t {
  uint32_t             kid;     /* -1 if any kprocessor */
} kaapic_spawn_attr_t;
  
/*
*/
extern int kaapic_spawn_attr_init(kaapic_spawn_attr_t* attr);

/*
*/
static inline int kaapic_spawn_attr_destroy(kaapic_spawn_attr_t* attr)
{ return 0; }

/* Specify that task should be push into the queue of the kid kprocessor.
   If kid is bigger than the number of active kprocessor then the kid is
   considered as modulo the number of active kprocessor.
*/
extern int kaapic_spawn_attr_set_kproc(
  kaapic_spawn_attr_t* attr, 
  int kid
);


/* Create a task that may steal.
   See documentation for restriction on by value passing rule.
   The attribut attr may be null and correspond to the default attribut.
   \param body : the task body. 
   \param nargs: the argument count. 
   \param ...: a list of tuple { MODE, VALUE/@, COUNT, TYPE  } tuple list.
   \retval 0 in case of success, else an error code
*/
extern int kaapic_spawn(const kaapic_spawn_attr_t* attr, int32_t nargs, ...);

/*
*/
extern void kaapic_sync(void);


/* Flag for parallel regions
*/
enum {
  KAAPIC_FLAG_STATIC_SCHED = 0x1,
  KAAPIC_FLAG_END_NOSYNC   = 0x2,  /* no implicit sync */
  KAAPIC_FLAG_DEFAULT      = 0x0
};

/* Start parallel region with flag.
   \retval 0 in case of success, else an error code
*/
extern int kaapic_begin_parallel(int flags);

/* End parallel region with flag.
   If last parallel_begin has STATIC_SCHED flag, then
   the STATIC_SCHED_FLAG must also be passed to end_parallel.
   \retval 0 in case of success, else an error code
*/
extern int kaapic_end_parallel(int flags);

#if defined(__cplusplus)
}
#endif

#endif /* KAAPIC_H_INCLUDED */
