#include <stdlib.h>
#include <papi.h>
#include "kaapi_impl.h"


/* internal */

static int papi_event_codes[KAAPI_PERF_ID_PAPI_MAX];
static unsigned int papi_event_count = 0;
static int papi_event_set = PAPI_NULL;

static int get_event_code(char* name, int* code)
{
  char* end;

  /* hexa constant */
  *code = (int)strtoul(name, &end, 16);
  if (end != name)
    return 0;

  /* fallback to default case */
  if (PAPI_event_name_to_code(name, code) != PAPI_OK)
    return -1;

  return 0;
}

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
  if (papi_event_set != PAPI_NULL)
  {
    PAPI_cleanup_eventset(papi_event_set);
    PAPI_destroy_eventset(&papi_event_set);
  }
}


void kaapi_perf_thread_init(void)
{
  if (papi_event_count)
  {
    const int err = PAPI_start(papi_event_set);
    kaapi_assert_m(PAPI_OK, err, "PAPI_start()");
  }
}


void kaapi_perf_thread_fini(void)
{
  if (papi_event_count)
  {
    const int err = PAPI_stop(papi_event_set, NULL);
    kaapi_assert_m(err, PAPI_OK, "PAPI_stop()");
  }
}


static inline unsigned int get_perf_id(kaapi_perf_id_t* id)
{
  /* must return 0 or 1 since used as an index */

  const unsigned int is_user =
    ((*id) & KAAPI_PERF_ID_USER_MASK) >> KAAPI_PERF_ID_USER_POS;

  (*id) &= ~KAAPI_PERF_ID_USER_MASK;

  kaapi_assert_m(1, (*id < KAAPI_PERF_ID_MAX), "");

  return is_user;
}


void kaapi_perf_zero_counter(kaapi_perf_id_t id)
{
  /* zero the software counter
   */

  kaapi_processor_t* const kproc = kaapi_get_current_processor();
  const unsigned int is_user = get_perf_id(&id);

  if (id == KAAPI_PERF_ID_ALL)
    memset(kproc->counters[is_user], 0, sizeof(kproc->counters[0]));
  else
    kproc->counters[is_user][id] = 0;
}


static inline kaapi_atomic_t* get_internal_register
(
 kaapi_processor_t* kproc,
 kaapi_perf_id_t id
)
{
  /* todo: replace with an array */

  kaapi_atomic_t* p;

  switch (id)
  {
  case KAAPI_PERF_ID_TASKS:
    p = &kproc->cnt_tasks;
    break;

  case KAAPI_PERF_ID_STEALREQOK:
    p = &kproc->cnt_stealreqok;
    break;

  case KAAPI_PERF_ID_STEALREQ:
    p = &kproc->cnt_stealreq;
    break;

  case KAAPI_PERF_ID_STEALOP:
    p = &kproc->cnt_stealop;
    break;

  default:
    /* never reached */
    p = NULL;
    break; 
  }

  return p;
}


static inline kaapi_uint32_t read_internal_register
(
 kaapi_processor_t* kproc,
 kaapi_perf_id_t id
)
{
  /* read and reset semantic */
  kaapi_atomic_t* const r = get_internal_register(kproc, id);
  return KAAPI_ATOMIC_AND(r, 0);
}


static inline void reset_internal_register
(
 kaapi_processor_t* kproc,
 kaapi_perf_id_t id
)
{
  kaapi_atomic_t* const r = get_internal_register(kproc, id);
  KAAPI_ATOMIC_AND(r, 0);
}


void kaapi_perf_reset_counter(kaapi_perf_id_t id)
{
  /* reset the hardware counters
   */

  kaapi_processor_t* const kproc = kaapi_get_current_processor();

  get_perf_id(&id);

  if (id == KAAPI_PERF_ID_ALL)
  {
    kaapi_perf_id_t i;
    for (i = KAAPI_PERF_ID_TASKS; i < KAAPI_PERF_ID_PAPI_BASE; ++i)
      reset_internal_register(kproc, i);

    if (papi_event_count)
    {
      const int err = PAPI_reset(papi_event_set);
      kaapi_assert_m(PAPI_OK, err, "PAPI_read_0");
    }

    return ;
  }

  /* single counter case */

  if (id < KAAPI_PERF_ID_PAPI_BASE)
  {
    reset_internal_register(kproc, id);
  }
  else if (papi_event_count)
  {
    /* todo: should read the single counter */
    const int err = PAPI_reset(papi_event_set);
    kaapi_assert_m(PAPI_OK, err, "PAPI_read_1");
  }
}


void kaapi_perf_snap_counter(kaapi_perf_id_t id)
{
  /* store the hardware counter
   */

  kaapi_processor_t* const kproc = kaapi_get_current_processor();
  const unsigned int is_user = get_perf_id(&id);

  if (id == KAAPI_PERF_ID_ALL)
  {
    kaapi_perf_id_t i;
    for (i = KAAPI_PERF_ID_TASKS; i < KAAPI_PERF_ID_PAPI_BASE; ++i)
      kproc->counters[is_user][i] = read_internal_register(kproc, i);

    if (papi_event_count)
    {
      /* todo: read only the given register */
      const int err = PAPI_read
	(
	 papi_event_set,
	 &kproc->counters[is_user][KAAPI_PERF_ID_PAPI_BASE]
	);
      kaapi_assert_m(PAPI_OK, err, "PAPI_read_2");
    }

    return ;
  }

  /* single counter case */

  if (id < KAAPI_PERF_ID_PAPI_BASE)
  {
    /* todo: atomic */
    kproc->counters[is_user][id] = read_internal_register(kproc, id);
  }
  else if (papi_event_count)
  {
    /* todo: read only the given register */
    long_long papi_values[KAAPI_PERF_ID_PAPI_MAX];

    const int err = PAPI_read(papi_event_set, papi_values);
    kaapi_assert_m(PAPI_OK, err, "PAPI_read_3");

    kproc->counters[is_user][id] = papi_values[id - KAAPI_PERF_ID_PAPI_BASE];
  }
}


void kaapi_perf_accum_counter(kaapi_perf_id_t id)
{
  /* accumulate the hardware counter
   */

  kaapi_processor_t* const kproc = kaapi_get_current_processor();
  const unsigned int is_user = get_perf_id(&id);

  if (id == KAAPI_PERF_ID_ALL)
  {
    kaapi_perf_id_t i;
    for (i = KAAPI_PERF_ID_TASKS; i < KAAPI_PERF_ID_PAPI_BASE; ++i)
      kproc->counters[is_user][i] += read_internal_register(kproc, i);

    if (papi_event_count)
    {
      const int err = PAPI_accum
	(
	 papi_event_set,
	 &kproc->counters[is_user][KAAPI_PERF_ID_PAPI_BASE]
	);

      kaapi_assert_m(PAPI_OK, err, "PAPI_accum_0");
    }

    return ;
  }

  /* single counter case */

  if (id < KAAPI_PERF_ID_PAPI_BASE)
  {
    kproc->counters[is_user][id] += read_internal_register(kproc, id);
  }
  else if (papi_event_count)
  {
    /* todo: should be accum */
    long_long papi_values[KAAPI_PERF_ID_PAPI_MAX];
    const int err = PAPI_read(papi_event_set, papi_values);
    kaapi_assert_m(PAPI_OK, err, "PAPI_accum_1");
    kproc->counters[is_user][id] += papi_values[id - KAAPI_PERF_ID_PAPI_BASE];
  }
}


void kaapi_perf_reduce_counter(kaapi_perf_id_t id, kaapi_perf_counter_t* counter)
{
  /* reduce the software counters
   */

  const unsigned int is_user = get_perf_id(&id);
  unsigned int k;

  if (id == KAAPI_PERF_ID_ALL)
  {
    kaapi_perf_id_t i;
    unsigned int j;

    memset(counter, 0, sizeof(kaapi_perf_counter_t) * KAAPI_PERF_ID_MAX);

    for (k = 0; k < kaapi_count_kprocessors; ++k)
    {
      const kaapi_processor_t* const kproc = kaapi_all_kprocessors[k];

      for (i = KAAPI_PERF_ID_TASKS; i < KAAPI_PERF_ID_PAPI_BASE; ++i)
	counter[i] += kproc->counters[is_user][i];

      for (j = 0; j < papi_event_count; ++j, ++i)
	counter[i] += kproc->counters[is_user][i];
    }

    return ;
  }

  /* single counter case */

  *counter = 0;

  for (k = 0; k < kaapi_count_kprocessors; ++k)
  {
    const kaapi_processor_t* const kproc = kaapi_all_kprocessors[k];

    *counter += kproc->counters[is_user][id];
  }
}
