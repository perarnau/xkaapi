#include <string.h>
#include "kaapi_impl.h"


static kaapi_atomic_t hws_steal_counters[KAAPI_MAX_PROCESSOR][KAAPI_MAX_PROCESSOR];
static kaapi_atomic_t hws_hit_counters[KAAPI_MAX_PROCESSOR][KAAPI_MAX_PROCESSOR];
static kaapi_atomic_t hws_pop_counters[KAAPI_MAX_PROCESSOR];


void kaapi_hws_init_counters(void)
{
  memset(hws_steal_counters, 0, sizeof(hws_steal_counters));
  memset(hws_hit_counters, 0, sizeof(hws_hit_counters));
  memset(hws_pop_counters, 0, sizeof(hws_pop_counters));
}


void kaapi_hws_inc_steal_counter
(kaapi_processor_id_t fu, kaapi_processor_id_t bar)
{
  KAAPI_ATOMIC_INCR(&hws_steal_counters[fu][bar]);
}


void kaapi_hws_inc_hit_counter
(kaapi_processor_id_t fu, kaapi_processor_id_t bar)
{
  KAAPI_ATOMIC_INCR(&hws_hit_counters[fu][bar]);
}


void kaapi_hws_inc_pop_counter(kaapi_processor_id_t fu)
{
  KAAPI_ATOMIC_INCR(&hws_pop_counters[fu]);
}


static inline unsigned int is_kid_used(kaapi_processor_id_t kid)
{
  return (kaapi_default_param.kid2cpu[kid] != -1);
}

static inline unsigned int kid2cpu(kaapi_processor_id_t kid)
{
  return kaapi_default_param.kid2cpu[kid];
}

void kaapi_hws_print_counters(void)
{
  kaapi_processor_id_t fu;
  kaapi_processor_id_t bar;

  for (fu = 0; fu < KAAPI_MAX_PROCESSOR; ++fu)
  {
    if (is_kid_used(fu) == 0) continue ;

    printf("%02u:", kid2cpu(fu));

    printf(" %08u", KAAPI_ATOMIC_READ(&hws_pop_counters[fu]));

    for (bar = 0; bar < KAAPI_MAX_PROCESSOR; ++bar)
    {
      if (is_kid_used(bar) == 0) continue ;
      printf(" %08u,%08u",
	     KAAPI_ATOMIC_READ(&hws_steal_counters[fu][bar]),
	     KAAPI_ATOMIC_READ(&hws_hit_counters[fu][bar]));
    }

    printf("\n");
  }
}
