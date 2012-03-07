/*
** xkaapi
** 
**
** Copyright 2009 INRIA.
**
** Contributors :
**
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
#include <stdlib.h>

#if defined(KAAPI_USE_PAPIPERFCOUNTER)
#include <papi.h>
#endif
#include "kaapi_impl.h"
#include "kaapi_event_recorder.h"

/* internal */
#if defined(KAAPI_USE_PAPIPERFCOUNTER)
static int papi_event_codes[KAAPI_PERF_ID_PAPI_MAX];
static unsigned int papi_event_count = 0;
#endif

/**/
#define KAAPI_GET_THREAD_STATE(kproc)\
  ((kproc)->curr_perf_regs == (kproc)->perf_regs[KAAPI_PERF_USER_STATE] ? KAAPI_PERF_USER_STATE : KAAPI_PERF_SCHEDULE_STATE)


/*
*/
int kaapi_mt_perf_thread_state(kaapi_processor_t* kproc)
{
  return KAAPI_GET_THREAD_STATE(kproc);
}


#if defined(KAAPI_USE_PAPIPERFCOUNTER)

static inline int get_event_code(char* name, int* code)
{
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
void kaapi_mt_perf_init(void)
{
#if defined(KAAPI_USE_PAPIPERFCOUNTER)
  /* must be called once */
  int error;

  error = PAPI_library_init(PAPI_VER_CURRENT);
  kaapi_assert_m(error == PAPI_VER_CURRENT, "PAPI_library_init()");
  
  error = PAPI_thread_init(pthread_self);
  kaapi_assert_m(error == PAPI_OK, "PAPI_thread_init()");

  error = get_papi_events();
  kaapi_assert_m(0 == error, "get_papi_events()");
#endif
}


/**
*/
void kaapi_mt_perf_fini(void)
{
#if defined(KAAPI_USE_PAPIPERFCOUNTER)
  PAPI_shutdown();
#endif
}


/*
*/
void kaapi_mt_perf_thread_init( kaapi_processor_t* kproc, int isuser )
{
  kaapi_assert( (isuser ==0)||(isuser==1) );
  
#if defined(KAAPI_USE_PAPIPERFCOUNTER)
  kproc->papi_event_count = 0;

  if (papi_event_count)
  {
    int err;
    PAPI_option_t opt;

    /* register the thread */
    err = PAPI_register_thread();
    kaapi_assert_m(PAPI_OK == err, "PAPI_register_thread()\n");

    /* create event set */
    kproc->papi_event_set = PAPI_NULL;
    err = PAPI_create_eventset(&kproc->papi_event_set);
    kaapi_assert_m(PAPI_OK == err, "PAPI_create_eventset()\n");

    /* set cpu as the default component. mandatory in newer interfaces. */
    err = PAPI_assign_eventset_component(kproc->papi_event_set, 0);
    kaapi_assert_m(PAPI_OK == err, "PAPI_assign_eventset_component()\n");

    /* thread granularity */
    memset(&opt, 0, sizeof(opt));
    opt.granularity.def_cidx = kproc->papi_event_set;
    opt.granularity.eventset = kproc->papi_event_set;
    opt.granularity.granularity = PAPI_GRN_THR;
    err = PAPI_set_opt(PAPI_GRANUL, &opt);
    kaapi_assert_m(PAPI_OK == err, "PAPI_set_opt_grn()");

    /* user domain */
    memset(&opt, 0, sizeof(opt));
    opt.domain.eventset = kproc->papi_event_set;
    opt.domain.domain = PAPI_DOM_USER;
    err = PAPI_set_opt(PAPI_DOMAIN, &opt);
    kaapi_assert_m(PAPI_OK == err, "PAPI_set_opt_dom()");

    err = PAPI_add_events
      (kproc->papi_event_set, papi_event_codes, papi_event_count);
    kaapi_assert_m(PAPI_OK == err, "PAPI_add_events()\n");

    kproc->papi_event_count = papi_event_count;

    err = PAPI_start(kproc->papi_event_set);
    kaapi_assert_m(PAPI_OK == err, "PAPI_start()\n");
  }
#endif
}


/*
*/
#if defined(KAAPI_USE_PAPIPERFCOUNTER)
void kaapi_mt_perf_thread_fini(kaapi_processor_t* kproc)
{
  if (kproc->papi_event_count)
  {
    PAPI_stop(kproc->papi_event_set, NULL);
    PAPI_cleanup_eventset(kproc->papi_event_set);
    PAPI_destroy_eventset(&kproc->papi_event_set);
    kproc->papi_event_set = PAPI_NULL;
    kproc->papi_event_count = 0;
    PAPI_unregister_thread();
  }
}
#else
void kaapi_mt_perf_thread_fini(kaapi_processor_t* kproc __attribute__((unused)) )
{
}
#endif


/*
*/
#if defined(KAAPI_USE_PAPIPERFCOUNTER)
void kaapi_mt_perf_thread_start(kaapi_processor_t* kproc)
{
  if (kproc->papi_event_count)
  {
    /* not that event counts between kaapi_perf_thread_init and here represent the
       cost to this thread wait all intialization of other threads, set setconcurrency.
       After this call we assume that we are counting.
    */
    kaapi_assert_m(PAPI_OK == PAPI_reset(kproc->papi_event_set), "PAPI_reset" );
  }
}
#else
void kaapi_mt_perf_thread_start(kaapi_processor_t* kproc __attribute__((unused)) )
{
}
#endif


/*
*/
#if defined(KAAPI_USE_PAPIPERFCOUNTER)
void kaapi_mt_perf_thread_stop(kaapi_processor_t* kproc)
{
  if (kproc->papi_event_count)
  {
    /* in fact here we accumulate papi counter:
       the thread do not restart counting
    */
    const int err = PAPI_accum
    (
     kproc->papi_event_set,
     (long_long*)(kproc->curr_perf_regs + KAAPI_PERF_ID_PAPI_BASE)
    );
    kaapi_assert_m(PAPI_OK == err, "PAPI_accum");
  }
}
#else
void kaapi_mt_perf_thread_stop(kaapi_processor_t* kproc __attribute__((unused)) )
{
}
#endif


/*
*/
#if defined(KAAPI_USE_PAPIPERFCOUNTER)
void kaapi_mt_perf_thread_stopswapstart( kaapi_processor_t* kproc, int isuser )
{
  if (papi_event_count)
  {
    /* we accumulate papi counter in previous mode
       do not reset as in kaapi_perf_thread_start
       because PAPI_accum does that
    */
    const int err = PAPI_accum
    (
     kproc->papi_event_set,
     (long_long*)(kproc->curr_perf_regs + KAAPI_PERF_ID_PAPI_BASE)
    );
    kaapi_assert_m(PAPI_OK == err, "PAPI_accum");
  }
}
#else
void kaapi_mt_perf_thread_stopswapstart( 
  kaapi_processor_t* kproc __attribute__((unused)), 
  int isuser __attribute__((unused))
)
{
}
#endif

/*
*/
uint64_t kaapi_mt_perf_thread_delayinstate(kaapi_processor_t* kproc)
{
  kaapi_perf_counter_t delay = kaapi_get_elapsedns() - kproc->start_t[KAAPI_GET_THREAD_STATE(kproc)];
  return delay;
}


const char* kaapi_mt_perf_id_to_name(kaapi_perf_id_t id)
{
  static const char* names[] =
  {
    "PAPI_0",
    "PAPI_1",
    "PAPI_2"
  };
  kaapi_assert_debug( (0 <= id) && (id <3) );
  return names[(size_t)id];
}


size_t kaapi_mt_perf_counter_num(void)
{
  return KAAPI_PERF_ID_PAPI_BASE
#if defined(KAAPI_USE_PAPIPERFCOUNTER)
    + papi_event_count
#endif
    ;
}

