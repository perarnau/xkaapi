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

#if defined(__cplusplus)
extern "C" {
#endif

/* This is the portable C interface the kaapi runtime */
enum kaapic_type
{
  KAAPIC_TYPE_CHAR = 0,
  KAAPIC_TYPE_INT,
  KAAPIC_TYPE_REAL,
  KAAPIC_TYPE_DOUBLE,
  KAAPIC_TYPE_PTR
};

enum kaapic_mode
{
  KAAPIC_MODE_R = 0,
  KAAPIC_MODE_W,
  KAAPIC_MODE_RW,
  KAAPIC_MODE_V
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

typedef struct kaapic_foreach_attr_t {
  uint32_t s_grain;
  uint32_t p_grain;  
} kaapic_foreach_attr_t;
  
extern int kaapic_foreach_attr_init(kaapic_foreach_attr_t* attr);
extern int kaapic_foreach_attr_set_grains(
  kaapic_foreach_attr_t* attr, 
  uint32_t s_grain,
  uint32_t p_grain
);
static inline int kaapic_foreach_attr_destroy(kaapic_foreach_attr_t* attr)
{ return 0; }

/* See documentation
*/
extern void kaapic_foreach( 
  int32_t beg, 
  int32_t last, 
  kaapic_foreach_attr_t* attr,
  int32_t nparam, 
  /* void (*body)(int32_t, int32_t, int32_t, ...), */
  ...
);


/* See documentation
*/
extern void kaapic_foreach_with_format(
  int32_t beg, 
  int32_t last, 
  kaapic_foreach_attr_t* attr,
  int32_t nparam, 
  /* void (*body)(int32_t, int32_t, int32_t, ...), */
  ...
);

/* Create a task that may be steal.
   See documentation for restriction on by value passing rule.
  \param body : the task body. 
  \param nargs: the argument count. 
  \param ...: a list of tuple { VALUE, COUNT, TYPE, MODE } tuple list.
  \retval 0 in case of success, else an error code
*/
extern int kaapic_spawn(int32_t nargs, ...);

/*
*/
extern void kaapic_sync(void);

/*
*/
extern void kaapic_begin_parallel(void);

/*
*/
extern void kaapic_end_parallel(int32_t flags);

#if defined(__cplusplus)
}
#endif

#endif /* KAAPIC_H_INCLUDED */
