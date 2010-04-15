#include <stdlib.h>

#if defined(KAAPI_USE_PAPIPERFCOUNTER)
#include <papi.h>
#endif
#include "kaapi_impl.h"

/* internal */
#if defined(KAAPI_USE_PAPIPERFCOUNTER)
static int papi_event_codes[KAAPI_PERF_ID_PAPI_MAX];
static unsigned int papi_event_count = 0;
#endif

/**/
#define KAAPI_GET_THREAD_STATE(kproc)\
  ((kproc)->curr_perf_regs == (kproc)->perf_regs[KAAPI_PERF_USER_STATE] ? KAAPI_PERF_USER_STATE : KAAPI_PERF_SCHEDULE_STATE)


#if 0 // unused
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
#endif

#if defined(KAAPI_USE_PAPIPERFCOUNTER)
static int get_papi_events(void)
{
  /* todo: [u|k]:EVENT_NAME */

  unsigned int i = 0;
  unsigned int j;
  const char* p;
  const char* s;
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
#endif
}


/**
*/
void kaapi_perf_global_fini(void)
{
}


/*
*/
void kaapi_perf_thread_init(kaapi_processor_t* kproc, int isuser)
{
  kaapi_assert( (isuser ==0)||(isuser==1) );
  
  memset( kproc->perf_regs, 0, sizeof( kproc->perf_regs) );
  kproc->start_t[0] = kproc->start_t[1] = 0;

  kproc->curr_perf_regs = kproc->perf_regs[isuser]; 

#if defined(KAAPI_USE_PAPIPERFCOUNTER)
  kproc->papi_event_count = 0;

  if (papi_event_count)
  {
    int err;
    PAPI_option_t opt;

    /* create event set */
    kproc->papi_event_set = PAPI_NULL;
    err = PAPI_create_eventset(&kproc->papi_event_set);
    kaapi_assert_m(PAPI_OK, err, "PAPI_create_eventset()\n");

    /* thread granularity */
    memset(&opt, 0, sizeof(opt));
    opt.granularity.eventset = kproc->papi_event_set;
    opt.granularity.granularity = PAPI_GRN_THR;
    err = PAPI_set_opt(PAPI_GRANUL, &opt);
    kaapi_assert_m(PAPI_OK, err, "PAPI_set_opt_grn()");

    /* user domain */
    memset(&opt, 0, sizeof(opt));
    opt.domain.eventset = kproc->papi_event_set;
    opt.domain.domain = PAPI_DOM_USER;
    err = PAPI_set_opt(PAPI_DOMAIN, &opt);
    kaapi_assert_m(PAPI_OK, err, "PAPI_set_opt_dom()");

    err = PAPI_add_events
      (kproc->papi_event_set, papi_event_codes, papi_event_count);
    kaapi_assert_m(PAPI_OK, err, "PAPI_add_events()\n");

    kproc->papi_event_count = papi_event_count;
  }
#endif
}


/*
*/
void kaapi_perf_thread_fini(kaapi_processor_t* kproc)
{
#if defined(KAAPI_USE_PAPIPERFCOUNTER)
  if (kproc->papi_event_count)
  {
    PAPI_stop(kproc->papi_event_set, NULL);
    PAPI_cleanup_eventset(kproc->papi_event_set);
    PAPI_destroy_eventset(&kproc->papi_event_set);
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
  kproc->start_t[KAAPI_GET_THREAD_STATE(kproc)] = kaapi_get_elapsedns();
printf("%lu:: %p start in mode '%s'\n", kaapi_get_elapsedns(),
  (void*)kproc, (KAAPI_GET_THREAD_STATE(kproc) ==0 ? "USER" : "SYS" ) );
}


/*
*/
void kaapi_perf_thread_stop(kaapi_processor_t* kproc)
{
  int mode;
  kaapi_perf_counter_t delay;
#if defined(KAAPI_USE_PAPIPERFCOUNTER)
  if (papi_event_count)
  {
    /* in fact here we accumulate papi counter: the thread do not restart counting */
    kaapi_assert_m( PAPI_OK, PAPI_accum(kproc->papi_event_set, kproc->curr_perf_regs+KAAPI_PERF_ID_PAPI_BASE), "PAPI_accum" );
  }
#endif
printf("%lu:: %p stop in mode '%s'\n", kaapi_get_elapsedns(),
  (void*)kproc, (KAAPI_GET_THREAD_STATE(kproc) ==0 ? "USER" : "SYS" ) );
  mode = KAAPI_GET_THREAD_STATE(kproc);
  kaapi_assert_debug( kproc->start_t[mode] != 0 ); /* else none started */
  delay = kaapi_get_elapsedns() - kproc->start_t[mode];
  KAAPI_PERF_REG(kproc, KAAPI_PERF_ID_T1) += delay;
  kproc->start_t[mode] =0;
}


/*
*/
kaapi_uint64_t kaapi_perf_thread_delayinstate(kaapi_processor_t* kproc)
{
  kaapi_perf_counter_t delay = kaapi_get_elapsedns() - kproc->start_t[KAAPI_GET_THREAD_STATE(kproc)];
  return delay;
}

/*
*/
void kaapi_perf_thread_stopswapstart( kaapi_processor_t* kproc, int isuser )
{
  kaapi_assert( (isuser ==KAAPI_PERF_SCHEDULE_STATE)||(isuser==KAAPI_PERF_USER_STATE) );
  if (kproc->curr_perf_regs != kproc->perf_regs[isuser])
  { /* doit only iff real swap */
    kaapi_assert_debug( kproc->start_t[1-isuser] != 0 ); /* else none started */
    kaapi_assert_debug( kproc->start_t[isuser] == 0 );   /* else already started */
    kaapi_perf_counter_t date = kaapi_get_elapsedns();
    KAAPI_PERF_REG(kproc, KAAPI_PERF_ID_T1) += date - kproc->start_t[1-isuser];
    kproc->start_t[isuser]   = date;
    kproc->start_t[1-isuser] = 0;
printf("%lu:: %p swap to mode '%s'\n", kaapi_get_elapsedns(),
  (void*)kproc, (isuser ==0 ? "USER" : "SYS" ) );
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
  return KAAPI_GET_THREAD_STATE(kproc);
}



/* convert the idset passed as an argument
*/
static const kaapi_perf_idset_t* get_perf_idset
(
 const kaapi_perf_idset_t* param,
 kaapi_perf_idset_t* storage
)
{
  const kaapi_perf_id_t casted_param =
    (kaapi_perf_id_t)(uintptr_t)param;

  switch (casted_param)
  {
  case KAAPI_PERF_ID_TASKS:
  case KAAPI_PERF_ID_STEALREQOK:
  case KAAPI_PERF_ID_STEALREQ:
  case KAAPI_PERF_ID_STEALOP:
  case KAAPI_PERF_ID_SUSPEND:
  case KAAPI_PERF_ID_T1:
/*  case KAAPI_PERF_ID_TIDLE: */
  case KAAPI_PERF_ID_TPREEMPT:
  case KAAPI_PERF_ID_PAPI_0:
  case KAAPI_PERF_ID_PAPI_1:
  case KAAPI_PERF_ID_PAPI_2:
    {
      storage->count = 1;
      storage->idmap[0] = casted_param;
      param = storage;
      break;
    }

  default:
    break;
  }

  return param;
}


/*
*/
void _kaapi_perf_accum_counters(const kaapi_perf_idset_t* idset, int isuser, kaapi_perf_counter_t* counter)
{
  kaapi_assert( (isuser ==KAAPI_PERF_USR_COUNTER)||
                (isuser==KAAPI_PERF_SYS_COUNTER)||
                (isuser== (KAAPI_PERF_SYS_COUNTER|KAAPI_PERF_USR_COUNTER))
  );
  unsigned int k;
  unsigned int i;
  unsigned int j;
  kaapi_perf_idset_t local_idset;
  kaapi_perf_counter_t accum[KAAPI_PERF_ID_MAX];

  idset = get_perf_idset(idset, &local_idset);

  memset(accum, 0, sizeof(accum));
  
  for (k = 0; k < kaapi_count_kprocessors; ++k)
  {
    const kaapi_processor_t* const kproc = kaapi_all_kprocessors[k];
    
    if (isuser ==KAAPI_PERF_USR_COUNTER)
    {
      for (i = 0; i < KAAPI_PERF_ID_MAX; ++i)
        accum[i] += kproc->perf_regs[KAAPI_PERF_USER_STATE][i];
    }
    else if (isuser ==KAAPI_PERF_SYS_COUNTER)
    {
      for (i = 0; i < KAAPI_PERF_ID_MAX; ++i)
        accum[i] += kproc->perf_regs[KAAPI_PERF_SCHEDULE_STATE][i];
    }
    else
      for (i = 0; i < KAAPI_PERF_ID_MAX; ++i)
        accum[i] += kproc->perf_regs[KAAPI_PERF_USER_STATE][i] + kproc->perf_regs[KAAPI_PERF_SCHEDULE_STATE][i];
  }
  
  /* filter */
  for (j = 0, i = 0; j < idset->count; ++i)
    if (idset->idmap[i])
      counter[j++] += accum[i];
}


/*
*/
void _kaapi_perf_read_counters(const kaapi_perf_idset_t* idset, int isuser, kaapi_perf_counter_t* counter)
{
  kaapi_assert( (isuser ==0)||(isuser==1) );

  memset(counter, 0, idset->count * sizeof(kaapi_perf_counter_t) );
  _kaapi_perf_accum_counters( idset, isuser, counter );
}


/*
 */
void _kaapi_perf_read_register(const kaapi_perf_idset_t* idset, int isuser, kaapi_perf_counter_t* counter)
{
  kaapi_assert( (isuser ==0)||(isuser==1) );
  kaapi_processor_t* const kproc = kaapi_get_current_processor();
  unsigned int j;
  unsigned int i;
  kaapi_perf_idset_t local_idset;

  idset = get_perf_idset(idset, &local_idset);

  for (j = 0, i = 0; j < idset->count; ++i)
    if (idset->idmap[i])
      counter[j++] = kproc->perf_regs[isuser][i];
}


/*
 */
void _kaapi_perf_accum_register(const kaapi_perf_idset_t* idset, int isuser, kaapi_perf_counter_t* accum)
{
  kaapi_processor_t* const kproc = kaapi_get_current_processor();
  unsigned int j;
  unsigned int i;
  kaapi_perf_idset_t local_idset;

  idset = get_perf_idset(idset, &local_idset);

  for (j = 0, i = 0; j < idset->count; ++i)
    if (idset->idmap[i])
      accum[j++] += kproc->perf_regs[isuser][i];
}


const char* kaapi_perf_id_to_name(kaapi_perf_id_t id)
{
  static const char* names[] =
  {
    "TASKS",
    "STEALREQOK",
    "STEALREQ",
    "STEALOP",
    "SUSPEND",
    "PAPI_0",
    "PAPI_1",
    "PAPI_2"
  };

  return names[(size_t)id];
}


size_t kaapi_perf_counter_num(void)
{
  return KAAPI_PERF_ID_PAPI_BASE
#if defined(KAAPI_USE_PAPIPERFCOUNTER)
    + papi_event_count
#endif
    ;
}


void kaapi_perf_idset_zero(kaapi_perf_idset_t* set)
{
  set->count = 0;
  memset(set->idmap, 0, sizeof(set->idmap));
}


void kaapi_perf_idset_add(kaapi_perf_idset_t* set, kaapi_perf_id_t id)
{
  if (set->idmap[(size_t)id])
    return ;

  set->idmap[(size_t)id] = 1;
  ++set->count;
}
