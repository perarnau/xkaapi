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
#include "kaapi_impl.h"
#include <stdlib.h>
#include <inttypes.h> 

/*
*/
kaapi_uint32_t kaapi_count_kprocessors = 0;

/*
*/
kaapi_processor_t** kaapi_all_kprocessors = 0;


/*
*/
pthread_key_t kaapi_current_processor_key;

/*
*/
pthread_key_t c;

/* 
*/
kaapi_atomic_t kaapi_term_barrier = { 0 };

/* 
*/
volatile int kaapi_isterm = 0;

/** Should be with the same file as kaapi_init
 */
void _kaapi_dummy(void* foo)
{
}

/** Dependencies with kaapi_stack_t* kaapi_self_stack(void)
*/
kaapi_stack_t* kaapi_self_stack(void)
{
  return _kaapi_self_stack();
}

/** Dumy task pushed at startup into the main thread
*/
void kaapi_taskstartup_body( kaapi_task_t* task, kaapi_stack_t* stack)
{
}


/**
*/
void __attribute__ ((constructor)) kaapi_init(void)
{
  kaapi_isterm = 0;
  kaapi_stack_t* stack;
  kaapi_task_t* task;
  kaapi_frame_t frame;
  const char* version __attribute__((unused)) = get_kaapi_version();
  
  /* set up runtime parameters */
  kaapi_assert_m( 0, kaapi_setup_param( 0, 0 ), "kaapi_setup_param" );
  
  /* initialize the kprocessor key */
  kaapi_assert( 0 == pthread_key_create( &kaapi_current_processor_key, 0 ) );
    
  /* setup topology information */
  kaapi_setup_topology();

  /* set the kprocessor AFTER topology !!! */
  kaapi_assert_m( 0, kaapi_setconcurrency( default_param.cpucount ), "kaapi_setconcurrency" );
  
  pthread_setspecific( kaapi_current_processor_key, kaapi_all_kprocessors[0] );

  /* push dummy task in exec mode */
  stack = _kaapi_self_stack();
  kaapi_stack_save_frame(stack, &frame);
  task = kaapi_stack_toptask(stack);
  task->flag  = KAAPI_TASK_STICKY;
  task->body  = &kaapi_taskstartup_body;
  kaapi_task_format_debug( task );
  kaapi_task_setstate( task, KAAPI_TASK_S_EXEC );

  kaapi_stack_pushtask(stack);

  /* push marker of the frame: retn */
  kaapi_stack_pushretn(stack, &frame);
  
  /* dump output information */
#if defined(KAAPI_USE_PERFCOUNTER)
  printf("[KAAPI::INIT] use #physical cpu:%u, start time:%15f\n", default_param.cpucount,kaapi_get_elapsedtime());
#else
  printf("[KAAPI::INIT] use #physical cpu:%u\n", default_param.cpucount);
#endif
  fflush( stdout );
}


/**
*/
void __attribute__ ((destructor)) kaapi_fini(void)
{
  int i;
  kaapi_uint64_t cnt_tasks;
  kaapi_uint64_t cnt_stealreqok;
  kaapi_uint64_t cnt_stealreq;
  kaapi_uint64_t cnt_stealop;
  kaapi_uint64_t cnt_suspend;
  double t_idle;
  
#if defined(KAAPI_USE_PERFCOUNTER)
  printf("[KAAPI::TERM] end time:%15f\n", kaapi_get_elapsedtime());
#else
  printf("[KAAPI::TERM]\n");
#endif
  fflush( stdout );

  /* wait end of the initialization */
  kaapi_isterm = 1;
  kaapi_barrier_td_setactive(&kaapi_term_barrier, 0);
  
  while (!kaapi_barrier_td_isterminated( &kaapi_term_barrier ))
  {
    kaapi_sched_advance( kaapi_all_kprocessors[0] );
  }
  
  cnt_tasks       = 0;
  cnt_stealreqok  = 0;
  cnt_stealreq    = 0;
  cnt_stealop     = 0;
  cnt_suspend     = 0;
  t_idle          = 0;
  for (i=0; i<kaapi_count_kprocessors; ++i)
  {
    cnt_tasks       += kaapi_all_kprocessors[i]->cnt_tasks;
    cnt_stealreqok  += kaapi_all_kprocessors[i]->cnt_stealreqok;
    cnt_stealreq    += kaapi_all_kprocessors[i]->cnt_stealreq;
    cnt_stealop     += kaapi_all_kprocessors[i]->cnt_stealop;
    cnt_suspend     += kaapi_all_kprocessors[i]->cnt_suspend;
    t_idle          += kaapi_all_kprocessors[i]->t_idle;


#if defined(KAAPI_USE_PERFCOUNTER)
  /* */
  if (default_param.display_perfcounter)
  {
    printf("----- Performance counters, core   : %i\n", i);
    printf("%i: Total number of tasks executed     : %u\n", i, kaapi_all_kprocessors[i]->cnt_tasks);
    printf("%i: Total number of steal OK requests  : %u\n", i, kaapi_all_kprocessors[i]->cnt_stealreqok);
    printf("%i: Total number of steal BAD requests : %u\n", i, kaapi_all_kprocessors[i]->cnt_stealreq - kaapi_all_kprocessors[i]->cnt_stealreqok);
    printf("%i: Total number of steal operations   : %u\n", i, kaapi_all_kprocessors[i]->cnt_stealop);
    printf("%i: Total number of suspend operations : %u\n", i, kaapi_all_kprocessors[i]->cnt_suspend);
    printf("%i: Total idle time                    : %e\n", i, kaapi_all_kprocessors[i]->t_idle);
  }
#endif  

    free(kaapi_all_kprocessors[i]);
    kaapi_all_kprocessors[i]= 0;
  }
  free( kaapi_all_kprocessors );
  kaapi_all_kprocessors =0;

#if defined(KAAPI_USE_PERFCOUNTER)
#ifndef PRIu64
# if (sizeof(long) == sizeof(uint64_t))
#  define PRIu64 "lu"
# else
#  define PRIu64 "llu"
# endif
#endif

  /* */
  if (default_param.display_perfcounter)
  {
    printf("----- Cumulated Performance counters\n");
    printf("Total number of tasks executed     : %" PRIu64 "\n", cnt_tasks);
    printf("Total number of steal OK requests  : %" PRIu64 "\n", cnt_stealreqok);
    printf("Total number of steal BAD requests : %" PRIu64 "\n", cnt_stealreq-cnt_stealreqok);
    printf("Total number of steal operations   : %" PRIu64 "\n", cnt_stealop);
    printf("Total number of suspend operations : %u\n", cnt_suspend);
    printf("Total idle time                    : %e\n", t_idle);
    printf("Average steal requests aggregation : %e\n", ((double)cnt_stealreq)/(double)cnt_stealop);
  }
#endif  
  
  /* TODO: destroy topology data structure */
}
