/*
** kaapi_init.c
** xkaapi
** 
** Created on Tue Mar 31 15:19:03 2009
** Copyright 2009 INRIA.
**
** Contributors :
**
** christophe.laferriere@imag.fr
** thierry.gautier@inrialpes.fr
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
#include <inttypes.h> 
#include "kaapi_impl.h"



/*
*/
kaapi_uint32_t volatile kaapi_count_kprocessors = 0;

/*
*/
kaapi_processor_t** kaapi_all_kprocessors = 0;


/*
*/
pthread_key_t kaapi_current_processor_key;

/*
*/
#if defined(KAAPI_HAVE_COMPILER_TLS_SUPPORT)
__thread kaapi_thread_t** kaapi_current_thread_key;
__thread kaapi_threadgroup_t kaapi_current_threadgroup_key;
#endif

/*
*/
//pthread_key_t c;

/* 
*/
kaapi_atomic_t kaapi_term_barrier = { 0 };

/* 
*/
volatile int kaapi_isterm = 0;


/** 
*/
#if !defined(KAAPI_HAVE_COMPILER_TLS_SUPPORT)
kaapi_thread_t* kaapi_self_thread(void)
{
  return (kaapi_thread_t*)_kaapi_self_thread()->sfp;
}
kaapi_threadgroup_t kaapi_self_threadgroup(void)
{
  return _kaapi_self_thread()->thgrp;
}
void kaapi_set_threadgroup(kaapi_threadgroup_t thgrp)
{
  _kaapi_self_thread()->thgrp = thgrp;
}
#endif

/**
*/
int kaapi_mt_init(void)
{
  static int iscalled = 0;
  if (iscalled !=0) return 0;
  iscalled = 1;
  
  kaapi_isterm = 0;
  kaapi_thread_context_t* thread;
  kaapi_task_t*   task;
  const char*     version __attribute__((unused)) = get_kaapi_version();
  
  /* initialize the kprocessor key */
  kaapi_assert( 0 == pthread_key_create( &kaapi_current_processor_key, 0 ) );

#if defined(KAAPI_HAVE_COMPILER_TLS_SUPPORT)
  kaapi_current_thread_key = 0;
  kaapi_current_threadgroup_key = 0;
#endif
    
  /* setup topology information */
  kaapi_setup_topology();

#if defined(KAAPI_USE_PERFCOUNTER)
  /* call prior setconcurrency */
  kaapi_perf_global_init();
  /* kaapi_perf_thread_init(); */
#endif

  /* set the kprocessor AFTER topology !!! */
  kaapi_assert_m( 0 == kaapi_setconcurrency(), "kaapi_setconcurrency" );
  
/*** TODO BEG: this code should but outside machine specific init*/
  /* push dummy task in exec mode */
  thread = _kaapi_self_thread();
  task = kaapi_thread_toptask(kaapi_threadcontext2thread(thread));
  kaapi_task_init( task, kaapi_taskstartup_body, 0 );
  kaapi_task_setbody( task, kaapi_exec_body);
  kaapi_thread_pushtask(kaapi_threadcontext2thread(thread));

  /* push the current frame that correspond to the execution of the startup task */
  thread->sfp[1].pc      = thread->sfp->sp;
  thread->sfp[1].sp      = thread->sfp->sp;
  thread->sfp[1].sp_data = thread->sfp->sp_data;
  ++thread->sfp;

  /* WARNING strong impact on execution, see kaapi_sched_sync */
/*  kaapi_stack2threadcontext(stack)->frame_sp = stack->sp; */
/*** END */

  /* dump output information */
#if defined(KAAPI_USE_PERFCOUNTER)
  printf("[KAAPI::INIT] use #physical cpu:%u, start time:%15f\n", kaapi_default_param.cpucount,kaapi_get_elapsedtime());
  fflush( stdout );
#endif
  
  kaapi_default_param.startuptime = kaapi_get_elapsedns();
  
  return 1;
}


/**
*/
int kaapi_mt_finalize(void)
{
  int i;
#if defined(KAAPI_USE_PERFCOUNTER)
  kaapi_uint64_t cnt_tasks;
  kaapi_uint64_t cnt_stealreqok;
  kaapi_uint64_t cnt_stealreq;
  kaapi_uint64_t cnt_stealop;
  kaapi_uint64_t cnt_suspend;
  double t_sched;
  double t_preempt;
  double t_1;
#endif

  static int iscalled = 0;
  if (iscalled !=0) return 0;
  iscalled = 1;

#if defined(KAAPI_USE_PERFCOUNTER)
  /*  */
  kaapi_perf_thread_stop(kaapi_all_kprocessors[0]);
#endif
  
#if defined(KAAPI_USE_PERFCOUNTER)
  printf("[KAAPI::TERM] end time:%15f, delta: %15f(s)\n", kaapi_get_elapsedtime(), 
        (double)(kaapi_get_elapsedns()-kaapi_default_param.startuptime)*1e-9 );
#endif
  fflush( stdout );

  /* wait end of the initialization */
  kaapi_isterm = 1;
  kaapi_barrier_td_setactive(&kaapi_term_barrier, 0);
  
  while (!kaapi_barrier_td_isterminated( &kaapi_term_barrier ))
  {
    kaapi_sched_advance( kaapi_all_kprocessors[0] );
  }

#if defined(KAAPI_USE_PERFCOUNTER)
  kaapi_perf_thread_fini(kaapi_all_kprocessors[0]);
  kaapi_perf_global_fini();
  
  cnt_tasks       = 0;
  cnt_stealreqok  = 0;
  cnt_stealreq    = 0;
  cnt_stealop     = 0;
  cnt_suspend     = 0;

  t_sched         = 0;
  t_preempt       = 0;
  t_1             = 0;
#endif

  for (i=0; i<kaapi_count_kprocessors; ++i)
  {
#if defined(KAAPI_USE_PERFCOUNTER)

# ifndef PRIu64
#   if (sizeof(unsigned long) == sizeof(uint64_t))
#     define PRIu64 "lu"
#   else
#     define PRIu64 "llu"
#   endif
# endif

# ifndef PRI64
/* #   if (sizeof(unsigned long) == sizeof(uint64_t)) */
/* #     define PRI64 "ld" */
/* #   else */
#     define PRI64 "lld"
/* #   endif */
# endif 

# ifndef PRIu32
#   define PRIu32 "u"
# endif

  cnt_tasks +=      KAAPI_PERF_REG_READALL(kaapi_all_kprocessors[i], KAAPI_PERF_ID_TASKS);
  cnt_stealreqok += KAAPI_PERF_REG_READALL(kaapi_all_kprocessors[i], KAAPI_PERF_ID_STEALREQOK);
  cnt_stealreq +=   KAAPI_PERF_REG_READALL(kaapi_all_kprocessors[i], KAAPI_PERF_ID_STEALREQ);
  cnt_stealop +=    KAAPI_PERF_REG_READALL(kaapi_all_kprocessors[i], KAAPI_PERF_ID_STEALOP);
  cnt_suspend +=    KAAPI_PERF_REG_READALL(kaapi_all_kprocessors[i], KAAPI_PERF_ID_SUSPEND);
  t_sched +=        1e-9*(double)KAAPI_PERF_REG_SYS(kaapi_all_kprocessors[i], KAAPI_PERF_ID_T1);
  t_preempt +=      1e-9*(double)KAAPI_PERF_REG_SYS(kaapi_all_kprocessors[i], KAAPI_PERF_ID_TPREEMPT);
  t_1 +=            1e-9*(double)KAAPI_PERF_REG_USR(kaapi_all_kprocessors[i], KAAPI_PERF_ID_T1); 
    
  /* */
  if (kaapi_default_param.display_perfcounter)
  {

    printf("----- Performance counters, core   : %i\n", i);
    printf("Total number of tasks executed     : %" PRI64", %"PRI64"\n",
	   KAAPI_PERF_REG_USR(kaapi_all_kprocessors[i], KAAPI_PERF_ID_TASKS),
	   KAAPI_PERF_REG_SYS(kaapi_all_kprocessors[i], KAAPI_PERF_ID_TASKS)
    );
    printf("Total number of steal OK requests  : %"PRI64"\n",
	   KAAPI_PERF_REG_SYS(kaapi_all_kprocessors[i], KAAPI_PERF_ID_STEALREQOK)
    );
    printf("Total number of steal BAD requests : %"PRI64"\n",
	   KAAPI_PERF_REG_SYS(kaapi_all_kprocessors[i], KAAPI_PERF_ID_STEALREQ)-
	   KAAPI_PERF_REG_SYS(kaapi_all_kprocessors[i], KAAPI_PERF_ID_STEALREQOK)
    );
    printf("Total number of steal operations   : %"PRI64"\n",
	   KAAPI_PERF_REG_SYS(kaapi_all_kprocessors[i], KAAPI_PERF_ID_STEALOP)
    );
    printf("Total number of suspend operations : %"PRI64"\n",
	   KAAPI_PERF_REG_SYS(kaapi_all_kprocessors[i], KAAPI_PERF_ID_SUSPEND)
    );
    printf("Total compute time                 : %e\n",
       1e-9*(double)KAAPI_PERF_REG_USR(kaapi_all_kprocessors[i], KAAPI_PERF_ID_T1));

    printf("Total idle time                    : %e\n",
       1e-9*(KAAPI_PERF_REG_SYS(kaapi_all_kprocessors[i],KAAPI_PERF_ID_T1)
     + KAAPI_PERF_REG_SYS(kaapi_all_kprocessors[i],KAAPI_PERF_ID_TPREEMPT)) );
  }
#endif

#if defined(KAAPI_USE_CUDA)
    /* initialize cuda processor */
    if (kaapi_all_kprocessors[i]->proc_type == KAAPI_PROC_TYPE_CUDA)
      kaapi_cuda_proc_cleanup(&kaapi_all_kprocessors[i]->cuda_proc);
#endif /* KAAPI_USE_CUDA */
    kaapi_processor_free(kaapi_all_kprocessors[i]);
    kaapi_wsqueuectxt_destroy(&kaapi_all_kprocessors[i]->lsuspend);
    kaapi_all_kprocessors[i]= 0;
  }
  free( kaapi_all_kprocessors );
  kaapi_all_kprocessors =0;

#if defined(KAAPI_USE_PERFCOUNTER)
  /* */
  if (kaapi_default_param.display_perfcounter)
  {
    printf("----- Cumulated Performance counters\n");
    printf("Total number of tasks executed     : %" PRIu64 "\n", cnt_tasks);
    printf("Total number of steal OK requests  : %" PRIu64 "\n", cnt_stealreqok);
    printf("Total number of steal BAD requests : %" PRIu64 "\n", cnt_stealreq-cnt_stealreqok);
    printf("Total number of steal operations   : %" PRIu64 "\n", cnt_stealop);
    printf("Total number of suspend operations : %" PRIu64 "\n", cnt_suspend);
    printf("Total compute time                 : %e\n", t_1*1e-9);
    printf("Total idle time                    : %e\n", t_sched+t_preempt);
    printf("   sched idle time                 : %e\n", t_sched);
    printf("   preemption idle time            : %e\n", t_preempt);
    printf("Average steal requests aggregation : %e\n", ((double)cnt_stealreq)/(double)cnt_stealop);
  }
#endif  
  
  /* TODO: destroy topology data structure */
  return 0;
}
