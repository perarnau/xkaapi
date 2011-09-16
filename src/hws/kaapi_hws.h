#ifndef KAAPI_HWS_H_INCLUDED
# define KAAPI_HWS_H_INCLUDED


#define CONFIG_HWS_COUNTERS 1


#include "kaapi_impl.h"


/* internal to hws */


struct kaapi_ws_queue;


typedef struct kaapi_ws_block
{
  /* concurrent workstealing sync */
  /* todo: cache aligned, alone in the line */
  kaapi_atomic_t lock;

  /* workstealing queue */
  struct kaapi_ws_queue* queue;

  /* kid map of all the participants */
  kaapi_processor_id_t* kids;
  unsigned int kid_count;

#if CONFIG_HWS_COUNTERS
  /* counters, one per remote kid */
  kaapi_atomic_t steal_counters[KAAPI_MAX_PROCESSOR];
  kaapi_atomic_t pop_counters[KAAPI_MAX_PROCESSOR];
#endif /* CONFIG_HWS_COUNTERS */

} kaapi_ws_block_t;


typedef struct kaapi_hws_level
{
  kaapi_ws_block_t** kid_to_block;

  kaapi_ws_block_t* blocks;
  unsigned int block_count;

} kaapi_hws_level_t;


/* globals */

static const unsigned int hws_level_count = KAAPI_HWS_LEVELID_MAX;
extern kaapi_hws_level_t* hws_levels;
extern kaapi_hws_levelmask_t hws_levelmask;

/* internal exported functions */
extern const char* kaapi_hws_levelid_to_str(kaapi_hws_levelid_t);

static inline const unsigned int
kaapi_hws_is_levelid_set(kaapi_hws_levelid_t levelid)
{
  return hws_levelmask & (1 << levelid);
}

/* hws counters */

static inline void kaapi_hws_inc_pop_counter(void* q)
{
  /* increment the counter associated to the block containing a queue */
  KAAPI_ATOMIC_INCR(&hws_pop_counters[fu]);
}

static inline void kaapi_hws_inc_pop_counter(void* q)
{
  /* increment the counter associated to the block containing a queue */
  KAAPI_ATOMIC_INCR(&hws_pop_counters[fu]);
}


#endif /* ! KAAPI_HWS_H_INCLUDED */
