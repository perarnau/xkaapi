/*
 ** xkaapi
 ** 
 ** Copyright 2009,2010,2011,2012 INRIA.
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
#include "kaapi_impl.h"
#include "kaapic_impl.h"


kaapic_foreach_attr_t kaapic_default_attr;
unsigned int kaapic_do_parallel;

static kaapi_atomic_t kaapic_initcalled = { 0 };

int kaapic_init(int32_t flags)
{
  int err;
  if (KAAPI_ATOMIC_INCR(&kaapic_initcalled) != 1)
    return 0;

  if (flags ==0)
  {
    kaapic_do_parallel = 0;
    err = kaapi_init(1, 0, 0);
  }
  else /* only the main thread is started */
  {
    kaapic_do_parallel = 1;
    err = kaapi_init(0, 0, 0);
  }

  if (err !=0) return err;

  kaapic_foreach_attr_init( &kaapic_default_attr );
    
  if (getenv("KAAPI_SEQ_GRAIN") !=0)
    kaapic_default_attr.rep.li.s_grain = atoi(getenv("KAAPI_SEQ_GRAIN"));
  else
    kaapic_default_attr.rep.li.s_grain = 1;
  if (getenv("KAAPI_PAR_GRAIN") !=0)
    kaapic_default_attr.rep.li.p_grain = atoi(getenv("KAAPI_PAR_GRAIN"));
  else
    kaapic_default_attr.rep.li.p_grain = 1;

  _kaapic_register_task_format();
  return 0;
}

int kaapic_finalize(void)
{
  if (KAAPI_ATOMIC_DECR(&kaapic_initcalled) == 0)
  {
    kaapic_foreach_attr_destroy(&kaapic_default_attr);
    return kaapi_finalize();
  }

  return 0;
}

double kaapic_get_time(void)
{
  return kaapi_get_elapsedtime();
}

int32_t kaapic_get_concurrency(void)
{
  return (int32_t)kaapi_getconcurrency();
}

int32_t kaapic_get_thread_num(void)
{
  return (int32_t)kaapi_get_self_kid();
}

void kaapic_sync(void)
{
  kaapi_sched_sync();
}


int kaapic_begin_parallel(int flags)
{
  int schedflag = KAAPI_SCHEDFLAG_DEFAULT; /* for runtime */
  if (flags & KAAPIC_FLAG_STATIC_SCHED)
    schedflag |= KAAPI_SCHEDFLAG_STATIC;
  if (flags & KAAPIC_FLAG_END_NOSYNC)
    schedflag |= KAAPI_SCHEDFLAG_NOWAIT;

  kaapi_push_frame(&kaapi_self_thread_context()->stack );
  kaapi_begin_parallel(schedflag);

  return 0;
}

int kaapic_end_parallel(int flags)
{
  int schedflag = KAAPI_SCHEDFLAG_DEFAULT; /* for runtime */
  if (flags & KAAPIC_FLAG_STATIC_SCHED)
    schedflag |= KAAPI_SCHEDFLAG_STATIC;
  if (flags & KAAPIC_FLAG_END_NOSYNC)
    schedflag |= KAAPI_SCHEDFLAG_NOWAIT;

  kaapi_end_parallel(schedflag);
  kaapi_pop_frame(&kaapi_self_thread_context()->stack );
  
  return 0;
}


/* temporary */

#include "kaapi_impl.h"

static kaapi_frame_t saved_fp;

void kaapic_save_frame(void)
{
  kaapi_thread_context_t* const thread = kaapi_self_thread_context();

  saved_fp = *(kaapi_frame_t*)thread->stack.sfp;
  thread->stack.sfp[1] = saved_fp;
  kaapi_writemem_barrier();
  ++thread->stack.sfp;
}

void kaapic_restore_frame(void)
{
  kaapi_thread_context_t* const thread = kaapi_self_thread_context();

  kaapi_sched_lock(&thread->stack.lock);
  thread->stack.sfp->tasklist = 0;
  --thread->stack.sfp;
  *thread->stack.sfp = saved_fp;
  kaapi_sched_unlock(&thread->stack.lock);
}
