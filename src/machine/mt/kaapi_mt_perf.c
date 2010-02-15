#include <papi.h>
#include "kaapi_impl.h"



void kaapi_perf_global_init(void)
{
  /* must be called once */

  kaapi_assert(PAPI_library_init(PAPI_VER_CURRENT) == PAPI_VER_CURRENT);
  kaapi_assert(PAPI_thread_init(pthread_self) == PAPI_OK);
}


void kaapi_perf_global_fini(void)
{
}

typedef kaapi_uint64_t kaapi_perf_counter_t;


static void init_event_mapping(void)
{
  /* this is called in global_init
     to build a backend context ie.
     a papi event set for instance.
     the list contains per backend
     symbolics.
   */

  const char* const var = getenv("KAAPI_PERF_NAMES");
  if (var == NULL)
    return ;
}


void kaapi_perf_thread_init(void)
{
  /* retrieve the event mapping
     of the following form
     KAAPI_PERF_EVENTS=insn,l3
   */

  int papi_events[PAPI_NUM_EVENTS] = {PAPI_CODE_EVENTS};
  kaapi_assert(PAPI_start_counters(papi_events, PAPI_NUM_EVENTS) == PAPI_OK);
}


void kaapi_perf_thread_fini(void)
{
  long_long papi_values[PAPI_NUM_EVENTS];
  kaapi_assert(PAPI_stop_counters(papi_values, PAPI_NUM_EVENTS) == PAPI_OK);
}


inline static void read_counters(long_long* values, size_t count)
{
  const int err = PAPI_read_counters(values, count);
  if (err != PAPI_OK)
    printf("papi_error: %s(%s)\n", PAPI_strerror(err), __FUNCTION__);
}


void kaapi_perf_reset_counters(void)
{
  long_long papi_values[PAPI_NUM_EVENTS];
  read_counters(papi_values, PAPI_NUM_EVENTS);
}


void kaapi_perf_read_counters(void)
{
  kaapi_processor_t* const kproc = kaapi_get_current_processor();
  read_counters(kproc->papi_event_value, PAPI_NUM_EVENTS);
}


void kaapi_perf_accum_counters(void)
{
  kaapi_processor_t* const kproc = kaapi_get_current_processor();

  const int err = PAPI_accum_counters(kproc->papi_event_value, PAPI_NUM_EVENTS);
  if (err != PAPI_OK)
    printf("papi_error: %s(%s)\n", PAPI_strerror(err), __FUNCTION__);
}


void kaapi_perf_reduce_all_counters(kaapi_uint64_t* counters)
{
  unsigned int j;
  unsigned int i;

  for (j = 0; j < PAPI_NUM_EVENTS; ++j)
    counters[j] = 0;

  /* snaps */
  counters[j + 0] = 0;
  counters[j + 1] = 0;
  counters[j + 2] = 0;

  for (i = 0; i < (unsigned int)kaapi_count_kprocessors; ++i)
  {
    const kaapi_processor_t* const kproc = kaapi_all_kprocessors[i];

    for (j = 0; j < PAPI_NUM_EVENTS; ++j)
      counters[j] += kproc->papi_event_value[j];

    counters[j + 0] += kproc->cnt_stealreqok;
    counters[j + 1] += kproc->cnt_stealreq;
    counters[j + 2] += kproc->cnt_stealop;
  }
}


void kaapi_perf_reset_all_counters(void)
{
  unsigned int j;
  unsigned int i;

  for (i = 0; i < (unsigned int)kaapi_count_kprocessors; ++i)
  {
    kaapi_processor_t* const kproc = kaapi_all_kprocessors[i];

    for (j = 0; j < PAPI_NUM_EVENTS; ++j)
      kproc->papi_event_value[j] = 0;

    kproc->cnt_stealreqok = 0;
    kproc->cnt_stealreq = 0;
    kproc->cnt_stealop = 0;
  }
}
