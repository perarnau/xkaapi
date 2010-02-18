#include <stdlib.h>

#if defined(KAAPI_USE_PAPIPERFCOUNTER)
#include <papi.h>
#endif
#include "kaapi_impl.h"

/* internal */
#if defined(KAAPI_USE_PAPIPERFCOUNTER)
static int papi_event_codes[KAAPI_PERF_ID_PAPI_MAX];
static unsigned int papi_event_count = 0;
static int papi_event_set = PAPI_NULL;
#endif

static int get_event_code(char* name, int* code)
{
  char* end;

  /* hexa constant */
  *code = (int)strtoul(name, &end, 16);
  if (end != name)
    return 0;

#if defined(KAAPI_USE_PAPIPERFCOUNTER)
  /* fallback to default case */
  if (PAPI_event_name_to_code(name, code) != PAPI_OK)
    return -1;
#endif

  return 0;
}

#if defined(KAAPI_USE_PAPIPERFCOUNTER)
static int get_papi_events(void)
{
  /* todo: [u|k]:EVENT_NAME */

  unsigned int i = 0;
  unsigned int j;
  const char* p;
  const char* s;
  int err;
  char name[PAPI_MIN_STR_LEN];

  s = getenv("KAAPI_PERF_PAPIES");
  if (s == NULL)
    return 0;

  while (*s)
  {
    if (i >= KAAPI_PERF_ID_PAPI_MAX)
      return -1;

    p = s;

    for (j = 0; j < (sizeof(name) - 1) && *s && (*s != ','); ++s, ++j)
      name[j] = *s;
    name[j] = 0;

    if (get_event_code(name, &papi_event_codes[i]) == -1)
      return -1;

    ++i;

    if (*s == 0)
      break;

    ++s;
  }

  papi_event_count = i;

  /* create event set */
  err = PAPI_create_eventset(&papi_event_set);
  kaapi_assert_m(PAPI_OK, err, "PAPI_create_eventset()\n");

  err = PAPI_add_events
    (papi_event_set, papi_event_codes, papi_event_count);
  kaapi_assert_m(PAPI_OK, err, "PAPI_add_events()\n");

  return 0;
}
#endif


/**
*/
void kaapi_perf_global_init(void)
{
#if defined(KAAPI_USE_PAPIPERFCOUNTER)
  /* must be called once */
  int error;

  error = PAPI_library_init(PAPI_VER_CURRENT);
  kaapi_assert_m(error, PAPI_VER_CURRENT, "PAPI_library_init()");
  
  error = PAPI_thread_init(pthread_self);
  kaapi_assert_m(error, PAPI_OK, "PAPI_thread_init()");

  error = get_papi_events();
  kaapi_assert_m(0, error, "get_papi_events()");

  if (papi_event_count)
  {
    const int err = PAPI_start(papi_event_set);
    kaapi_assert_m(PAPI_OK, err, "PAPI_start()");
  }
#endif
}


/**
*/
void kaapi_perf_global_fini(void)
{
#if defined(KAAPI_USE_PAPIPERFCOUNTER)
  if (papi_event_count)
  {
    PAPI_stop(papi_event_set, NULL);
    PAPI_cleanup_eventset(papi_event_set);
    PAPI_destroy_eventset(&papi_event_set);
  }
#endif
}


/*
*/
void kaapi_perf_thread_init(kaapi_processor_t* kproc, int isuser)
{
  kaapi_assert( (isuser ==0)||(isuser==1) );
  
  memset( kproc->perf_regs, 0, sizeof( kproc->perf_regs) );
  kproc->t_sched        = 0;
  kproc->t_preempt      = 0;

  kproc->curr_perf_regs = kproc->perf_regs[isuser]; 

#if defined(KAAPI_USE_PAPIPERFCOUNTER)
  if (papi_event_count)
  {
    /* recopy global event set even if current hardware can not be used to
       have several different set per CPU: We hope that next hardware can do that per core basis.
    */
    kproc->papi_event_set = papi_event_set;
    kaapi_assert_m( PAPI_OK, PAPI_start(kproc->papi_event_set), "PAPI_start" );
  }
#endif

}



/*
*/
void kaapi_perf_thread_fini(kaapi_processor_t* kproc)
{
#if defined(KAAPI_USE_PAPIPERFCOUNTER)
  if (papi_event_count)
  {
    /* in fact here we stop the counter */
    kaapi_assert_m( PAPI_OK, PAPI_stop(kproc->papi_event_set), "PAPI_stop" );
  }
#endif
}


/*
*/
void kaapi_perf_thread_start(kaapi_processor_t* kproc)
{
#if defined(KAAPI_USE_PAPIPERFCOUNTER)
  if (papi_event_count)
  {
    /* not that event counts between kaapi_perf_thread_init and here represent the
       cost to this thread wait all intialization of other threads, set setconcurrency.
       After this call we assume that we are counting.
    */
    kaapi_assert_m( PAPI_OK, PAPI_reset(kproc->papi_event_set), "PAPI_reset" );
  }
#endif
}


/*
*/
void kaapi_perf_thread_stop(kaapi_processor_t* kproc)
{
#if defined(KAAPI_USE_PAPIPERFCOUNTER)
  if (papi_event_count)
  {
    /* in fact here we accumulate papi counter: the thread do not restart counting */
    kaapi_assert_m( PAPI_OK, PAPI_accum(kproc->papi_event_set, kproc->curr_perf_regs+KAAPI_PERF_ID_PAPI_BASE), "PAPI_accum" );
  }
#endif
}


/*
*/
void kaapi_perf_thread_stopswapstart( kaapi_processor_t* kproc, int isuser )
{
  kaapi_assert( (isuser ==0)||(isuser==1) );
  if (kproc->curr_perf_regs != &kproc->perf_regs[isuser])
  {
#if defined(KAAPI_USE_PAPIPERFCOUNTER)
    if (papi_event_count)
    {
      /* we accumulate papi counter in previous mode; do not reset as in kaapi_perf_thread_start because PAPI_accu do that */
      kaapi_assert_m( PAPI_OK, PAPI_accum(kproc->papi_event_set, kproc->curr_perf_regs+KAAPI_PERF_ID_PAPI_BASE), "PAPI_accum" );
    }
#endif
    kproc->curr_perf_regs = kproc->perf_regs[isuser]; 
  }
}


/*
*/
int kaapi_perf_thread_state(kaapi_processor_t* kproc)
{
  return (kproc->curr_perf_regs == &kproc->perf_regs[KAAPI_PERF_USER_STATE] ? KAAPI_PERF_USER_STATE : KAAPI_PERF_SCHEDULE_STATE);
}



/*
*/
void kaapi_perf_accum_counters(kaapi_perf_id_t id, int isuser, kaapi_perf_counter_t* counter)
{
  kaapi_assert( (isuser ==0)||(isuser==1) );
  unsigned int k;
  
  for (k = 0; k < kaapi_count_kprocessors; ++k)
  {
    const kaapi_processor_t* const kproc = kaapi_all_kprocessors[k];
    unsigned int i;
    
    for (i = 0; i < KAAPI_PERF_ID_MAX; ++i)
      counter[i] += kproc->perf_regs[isuser][i];
  }  
}


/*
*/
void kaapi_perf_read_counters(kaapi_perf_id_t id, int isuser, kaapi_perf_counter_t* counter)
{
  kaapi_assert( (isuser ==0)||(isuser==1) );
  unsigned int k;
  
  memset(counter, 0, sizeof(kaapi_perf_counter_t[KAAPI_PERF_ID_MAX]) );

  for (k = 0; k < kaapi_count_kprocessors; ++k)
  {
    const kaapi_processor_t* const kproc = kaapi_all_kprocessors[k];
    unsigned int i;
    
    for (i = 0; i < KAAPI_PERF_ID_MAX; ++i)
      counter[i] += kproc->perf_regs[isuser][i];
  }  
}
