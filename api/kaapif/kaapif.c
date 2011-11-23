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
#include "kaapi.h"
#include "kaapif.h"
#include "kaapic_impl.h"


extern void _kaapif_register_task_format(void);

#if CONFIG_MAX_TID
static int xxx_max_tid;
#endif
static int xxx_seq_grain;
static int xxx_par_grain;

#define FATAL()						\
do {							\
  printf("fatal error @ %s::%d\n", __FILE__, __LINE__);	\
  kaapi_abort();						\
} while(0)

static kaapi_atomic_t kaapif_initcalled = { 0 };

int kaapif_init_(int32_t* flags)
{
  if (KAAPI_ATOMIC_INCR(&kaapif_initcalled) != 1)
    return 0;

  int err = kaapic_init(*flags);
  if (err !=0) return err;

  /* default work info */
#if CONFIG_MAX_TID
  xxx_max_tid = (unsigned int)(kaapi_getconcurrency() - 1);
#endif
  xxx_seq_grain = 16;
  xxx_par_grain = 2 * xxx_seq_grain;

  return 0;
}


int kaapif_finalize_(void)
{
  if (KAAPI_ATOMIC_DECR(&kaapif_initcalled) != 0)
    return 0;
  return kaapic_finalize();
}


double kaapif_get_time_(void)
{
  return kaapi_get_elapsedtime();
}


int32_t kaapif_get_concurrency_(void)
{
  return (int32_t)kaapi_getconcurrency();
}


int32_t kaapif_get_thread_num_(void)
{
  return (int32_t)kaapi_get_self_kid();
}


void kaapif_set_max_tid_(int32_t* tid)
{
#if CONFIG_MAX_TID
  xxx_max_tid = *tid;
#else
  /* unused */
  *tid = *tid;
#endif
}


void kaapif_set_grains_(int32_t* par_grain, int32_t* seq_grain)
{
  xxx_par_grain = *par_grain;
  xxx_seq_grain = *seq_grain;
}


int32_t kaapif_get_max_tid_(void)
{
#if CONFIG_MAX_TID
  return xxx_max_tid;
#else
  return 0;
#endif
}


void kaapif_sched_sync_(void)
{
  kaapi_sched_sync();
}


int kaapif_begin_parallel_(void)
{
  kaapi_begin_parallel(KAAPI_SCHEDFLAG_DEFAULT);
  return 0;
}


int kaapif_end_parallel_(int32_t* flags)
{
  kaapi_end_parallel(*flags);
  return 0;
}
