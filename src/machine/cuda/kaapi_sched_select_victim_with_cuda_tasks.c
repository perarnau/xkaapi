/*
 ** kaapi_sched_select_victim_with_cuda_tasks.c
 ** 
 **
 ** Copyright 2009,2010,2011,2012 INRIA.
 **
 ** Contributors :
 **
 ** thierry.gautier@inrialpes.fr
 ** Joao.Lima@imag.fr
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

#include <stdlib.h>

#include "kaapi_impl.h"
#include "kaapi_cuda.h"
#include "../common/kaapi_procinfo.h"

#if 0
static inline int
kaapi_cuda_sched_has_task(const kaapi_processor_t * kproc)
{
  kaapi_thread_context_t *const thread = kproc->thread;
  kaapi_frame_t *top_frame;
  kaapi_tasklist_t *tasklist;
  kaapi_readytasklist_t *rtl;
  kaapi_onereadytasklist_t *onertl;
  kaapi_taskdescr_t *td;

  if (NULL == thread)
    return 0;

  for (top_frame = thread->stack.stackframe;
       (top_frame <= thread->stack.sfp); ++top_frame) {
    tasklist = top_frame->tasklist;
    if (tasklist == 0)
      continue;
    rtl = &tasklist->rtl;
    onertl = &rtl->prl[KAAPI_TASKLIST_MAX_PRIORITY];
    if (onertl->next == -1)
      continue;
    td = onertl->base[1 + onertl->next];
    if (0 != td->fmt->entrypoint[KAAPI_PROC_TYPE_CUDA]) {
#if 0
      fprintf(stdout,
	      "%s: OK frame %p td=%p(counter=%d,wc=%d,name=%s)\n",
	      __FUNCTION__, top_frame, td, td->counter, td->wc,
	      td->fmt->name);
      fflush(stdout);
#endif
      return 1;
    }
#if 0
    else {
      fprintf(stdout,
	      "%s: NOGPU frame %p td=%p(counter=%d,wc=%d,name=%s)\n",
	      __FUNCTION__, top_frame, td, td->counter, td->wc,
	      td->fmt->name);
      fflush(stdout);
    }
#endif
  }
  return 0;
}
#endif

int kaapi_sched_select_victim_with_cuda_tasks
    (kaapi_processor_t * kproc,
     kaapi_victim_t * victim, kaapi_selecvictim_flag_t flag)
#if 1				/* disable worksealing */
{
  return kaapi_sched_select_victim_rand(kproc, victim, flag);
}
#else
{
  switch (flag) {
  case KAAPI_SELECT_VICTIM:
    {
      int nbproc, victimid;

      /* select a victim */
      if (kproc->fnc_selecarg[0] == 0)
	kproc->fnc_selecarg[0] = (uintptr_t) (long) rand();

    redo_select:
      nbproc = kaapi_count_kprocessors;
      if (nbproc <= 1)
	return EINVAL;
      victimid = rand_r((unsigned int *) &kproc->fnc_selecarg) % nbproc;

      /* Get the k-processor */
      victim->kproc = kaapi_all_kprocessors[victimid];
      if (victim->kproc == 0)
	goto redo_select;
      if (!kaapi_cuda_sched_has_task(victim->kproc))
	return EINVAL;
      break;
    }

  default:
    {
      break;
    }
  }

  return 0;
}
#endif
