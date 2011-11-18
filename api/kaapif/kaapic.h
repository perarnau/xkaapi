/*
 ** xkaapi
 ** 
 ** Created on Tue Mar 31 15:19:14 2009
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


#include <stdint.h>
#include "kaapif.h"


/* c interface wrappers. depends on kaapif. */

typedef kaapif_foreach_fn_t kaapic_foreach_fn_t;
typedef kaapif_spawn_fn_t kaapic_spawn_fn_t;

static inline void kaapic_init(int32_t flags)
{
  kaapif_init_(&flags);
}

static inline void kaapic_finalize(void)
{
  kaapif_finalize_();
}

static inline double kaapic_get_time(void)
{
  return kaapif_get_time_();
}

static inline int32_t kaapic_get_concurrency(void)
{
  return kaapif_get_concurrency_();
}

static inline int32_t kaapic_get_thread_num(void)
{
  return kaapif_get_thread_num_();
}

static inline void kaapic_set_max_tid(int32_t tid)
{
  kaapif_set_max_tid_(&tid);
}

static inline int32_t kaapic_get_max_tid(void)
{
  return kaapif_get_max_tid_();
}

static inline void kaapic_set_grains(int32_t par_grain, int32_t seq_grain)
{
  kaapif_set_grains_(&par_grain, &seq_grain);
}

#define kaapic_foreach(__f, __i, __j, __n, ...)		\
do {							\
  int32_t ___i = __i;					\
  int32_t ___j = __j;					\
  int32_t ___n = __n;					\
  kaapif_foreach_					\
    ((kaapic_foreach_fn_t)__f, &___i, &___j, &___n, __VA_ARGS__); \
} while (0)

#define kaapic_foreach_with_format(__f, __i, __j, __n, ...)\
do {							\
  int32_t ___i = __i;					\
  int32_t ___j = __j;					\
  int32_t ___n = __n;					\
  kaapif_foreach_with_format_				\
    ((kaapic_foreach_fn_t)__f, &___i, &___j, &___n, __VA_ARGS__); \
} while (0)

#define kaapic_spawn(__f, __n, ...)			\
do {							\
  int32_t ___n = &__n;					\
  kaapif_spawn_						\
    ((kaapic_spawn_fn_t)__f, &___n, __VA_ARGS__);	\
} while (0)


static inline void kaapic_sched_sync(void)
{
  kaapif_sched_sync_();
}

static inline void kaapic_begin_parallel(void)
{
  kaapif_begin_parallel_();
}

static inline void kaapic_end_parallel(int32_t flags)
{
  kaapif_end_parallel_(&flags);
}


#endif /* KAAPIC_H_INCLUDED */
