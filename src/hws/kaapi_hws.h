#ifndef KAAPI_HWS_H_INCLUDED
# define KAAPI_HWS_H_INCLUDED


#define CONFIG_HWS_COUNTERS 1


#include "kaapi_impl.h"
#include "kaapi_ws_queue.h"


/* internal to hws */


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
  kaapi_atomic_t pop_counter;
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


#if CONFIG_HWS_COUNTERS

#define container_of(__ptr, __type, __member) \
  (__type*)((char*)__ptr - offsetof(__type, __member));

static inline kaapi_ws_block_t* __p_to_block(void* p)
{
  kaapi_ws_queue_t* const q = container_of(p, kaapi_ws_queue_t, data);
  return container_of(q, kaapi_ws_block_t, queue);
}

static inline void kaapi_hws_inc_pop_counter(void* p)
{
  /* increment the counter associated to the block containing a queue */
  kaapi_ws_block_t* const block = __p_to_block(p);
  printf("INCR_POP(%lx)\n", (unsigned long)block);
  KAAPI_ATOMIC_INCR(&block->pop_counter);
}

static inline void kaapi_hws_inc_steal_counter
(void* p, kaapi_processor_id_t kid)
{
  /* increment the counter associated to the block containing a queue */
  kaapi_ws_block_t* const block = __p_to_block(p);
  printf("INCR_STEAL(%lx)\n", (unsigned long)block);
  KAAPI_ATOMIC_INCR(&block->steal_counters[kid]);
}

#endif /* CONFIG_HWS_COUNTERS */


#endif /* ! KAAPI_HWS_H_INCLUDED */
