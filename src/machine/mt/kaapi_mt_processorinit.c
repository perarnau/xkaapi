/*
** kaapi_mt_processorinit.c
** xkaapi
** 
**
** Copyright 2009,2010,2011,2012 INRIA.
**
** Contributors :
**
** thierry.gautier@inrialpes.fr
** fabien.lementec@gmail.com 
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
#include "../common/kaapi_procinfo.h"

#if defined(KAAPI_USE_CUDA)
#include "machine/cuda/kaapi_cuda_impl.h"
#endif

#if defined(KAAPI_USE_CUDA)

/* todo: move somewhere else */
extern int kaapi_sched_select_victim_with_cuda_tasks
(kaapi_processor_t*, kaapi_victim_t*,  kaapi_selecvictim_flag_t);

#endif


/*
*/
int kaapi_processor_init( kaapi_processor_t* kproc, 
                          const struct kaapi_procinfo_t* kpi,
                          size_t stacksize
                        )
{
  int i;
  kaapi_thread_context_t* ctxt;

  kproc->thread       = 0;  
  kproc->kid          = kpi->kid;
  kproc->proc_type    = kpi->proc_type;
  kproc->kpi          = kpi;
 
  /* init hierarchy information */
  kproc->numa_nodeid = ~0;
  kproc->hlevel.depth = 0;
  for (i=0; i<ENCORE_UNE_MACRO_DETAILLEE; ++i)
  {
    kproc->hlevel.levels[i].nkids = 0;
    kproc->hlevel.levels[i].nsize = 0;
    kproc->hlevel.levels[i].kids = 0;
    kproc->hlevel.levels[i].set = 0;
  }
  
  kaapi_sched_initlock( &kproc->lock );
  
  kproc->mailbox.head   = 0;
  kproc->mailbox.tail   = 0;
  kproc->rtl_remote     = (kaapi_readytasklist_t*)malloc( sizeof (kaapi_readytasklist_t) );
  kaapi_readytasklist_init( kproc->rtl_remote );

  kproc->rtl     = (kaapi_readytasklist_t*)malloc( sizeof (kaapi_readytasklist_t) );
  kaapi_readytasklist_init( kproc->rtl );

  
  kproc->isidle         = 1;
  kaapi_wsqueuectxt_init( &kproc->lsuspend );

  kaapi_lfree_clear( kproc );

  kproc->seed = kproc->kid;
  kproc->fnc_selecarg[0] = 0;
  kproc->fnc_selecarg[1] = 0;
  kproc->fnc_selecarg[2] = 0;
  kproc->fnc_selecarg[3] = 0;
#if defined(KAAPI_USE_CUDA)
  if( kproc->proc_type == KAAPI_PROC_TYPE_CUDA )
	kproc->fnc_select = kaapi_sched_select_victim_with_cuda_tasks;
  else
#endif
  kproc->fnc_select      = kaapi_default_param.wsselect;

  /* not that, as all other fields, the processor_init is called
     before threads start to execute.
  */
  kproc->emitsteal       = kaapi_default_param.emitsteal;
  kaapi_assert( 0 == kaapi_default_param.emitsteal_initctxt(kproc) );
  
#if defined(KAAPI_DEBUG)
  kproc->req_version = 0;
  kproc->reply_version = 0;
  kproc->compute_version =0;
#endif

  kaapi_assert(0 == pthread_mutex_init(&kproc->suspend_lock, 0) );

  /* */
  kproc->eventbuffer     = 0;

#if defined(KAAPI_USE_PERFCOUNTER)
  kproc->serial          = 0;
  kproc->lastcounter     = 0;
#endif
  
  /* workload */
  kaapi_processor_set_workload(kproc, 0);

  /* seed */
  kproc->seed_data = rand();

  kaapi_processor_computetopo( kproc );

  ctxt = (kaapi_thread_context_t*)kaapi_context_alloc( kproc, stacksize );
  kaapi_assert(ctxt !=0);

  /* set new context to the kprocessor */
  kaapi_setcontext(kproc, ctxt);
  
  memset(&kproc->data_specific, 0, 16*sizeof(void*));
  memset(&kproc->size_specific, 0, 16*sizeof(size_t));
  
  kproc->libkomp_tls = 0;
  
  kaapi_address_space_id_t kasid = kaapi_memory_address_space_create(kaapi_network_get_current_globalid(), kpi->proc_type, 0x100000000UL);
  kaapi_memory_map_create(kproc->kid, kasid);
  
#if defined(KAAPI_USE_CUDA)
  /* initialize cuda processor */
  if (kpi->proc_type == KAAPI_PROC_TYPE_CUDA) {
    kproc->cuda_proc = kaapi_cuda_proc_alloc();
    if (kaapi_cuda_proc_initialize(kproc->cuda_proc, kpi->proc_index))
      return -1;
  }
#endif
  
  return 0;
}



int kaapi_processor_destroy(kaapi_processor_t* kproc)
{
  for (int i=0; i<16; ++i)
  {
    if (kproc->data_specific[i] !=0)
    {
#if defined(KAAPI_USE_CUDA)
      if (kproc->proc_type == KAAPI_PROC_TYPE_CUDA)
        kaapi_cuda_mem_free(kaapi_make_localpointer(kproc->data_specific[i]));
      else
#endif
        free(kproc->data_specific[i]);
    }
    kproc->data_specific[i] = 0;
    kproc->size_specific[i] = 0;
  }

  if (kproc->rtl_remote !=0) 
    free(kproc->rtl_remote);
  kproc->rtl_remote = 0;
  
  if (kproc->rtl !=0) 
    free(kproc->rtl);
  kproc->rtl = 0;

  kaapi_assert(0 == pthread_mutex_destroy(&kproc->suspend_lock) );
  return 0;
}
