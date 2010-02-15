#include <papi.h>
#include "kaapi_impl.h"



/* internal */

static int papi_event_codes[KAAPI_PERF_PAPI_MAX];
static unsigned int papi_event_count = 0;

static int get_papi_events(void)
{
  unsigned int i = 0;
  unsigned int j;
  const char* p;
  const char* s;
  char name[PAPI_MIN_STR_LEN];

  s = getenv("KAAPI_PAPI_EVENTS");
  if (s == NULL)
    return 0;

  while (*s)
  {
    if (i >= KAAPI_PERF_PAPI_MAX)
      return -1;

    p = s;

    for (j = 0; j < (sizeof(name) - 1) && *s && (*s != ','); ++s, ++j)
      name[j] = *s;
    name[j] = 0;

    if (PAPI_event_name_to_code(name, &papi_event_codes[i]) != PAPI_OK)
      return -1;

    ++i;

    if (*s == 0)
      break;

    ++s;
  }

  papi_event_count = i;

  return 0;
}


/* exported */

void kaapi_perf_global_init(void)
{
  /* must be called once */

  int error;

  error = PAPI_library_init(PAPI_VER_CURRENT);
  kaapi_assert_m(error, PAPI_VER_CURRENT, "PAPI_library_init()");
  
  error = PAPI_thread_init(pthread_self);
  kaapi_assert_m(error, PAPI_OK, "PAPI_thread_init()");

  error = get_papi_events();
  kaapi_assert_m(0, error, "get_papi_events()");
}


void kaapi_perf_global_fini(void)
{
}

typedef kaapi_uint64_t kaapi_perf_counter_t;


void kaapi_perf_thread_init(void)
{
  if (papi_event_count)
  {
    const int err = PAPI_start_counters(papi_event_codes, papi_event_count);
    kaapi_assert_m(PAPI_OK, err, "PAPI_start_counters()");
  }
}


void kaapi_perf_thread_fini(void)
{
  if (papi_event_count)
  {
    const int err = PAPI_stop_counters(NULL, papi_event_count);
    kaapi_assert_m(err, PAPI_OK, "PAPI_stop_counters()");
  }
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

#if 0
  if (papi_event_count)
  {
    const int err = PAPI_accum_counters(kproc->papi_event_value, papi_event_count);
#if 0
    if (err != PAPI_OK)
      printf("papi_error: %s(%s)\n", PAPI_strerror(err), __FUNCTION__);
#else
    kaapi_assert_m(PAPI_OK, err, "PAPI_accum_counters");
#endif
  }
#endif
}


void kaapi_perf_reduce_all_counters(kaapi_uint64_t* counters)
{
#if 0
  unsigned int j;
  unsigned int i;

  for (j = 0; j < papi_event_count; ++j)
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
#endif
}


void kaapi_perf_reset_all_counters(void)
{
#if 0
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
#endif
}
