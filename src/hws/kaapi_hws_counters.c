#include <string.h>
#include "kaapi_impl.h"
#include "kaapi_hws.h"
#include "kaapi_ws_queue.h"


#if CONFIG_HWS_COUNTERS


static inline unsigned int is_kid_used(kaapi_processor_id_t kid)
{
  return (kaapi_default_param.kid2cpu[kid] != -1);
}

static inline unsigned int kid_to_cpu(kaapi_processor_id_t kid)
{
  return kaapi_default_param.kid2cpu[kid];
}

void kaapi_hws_print_counters(void)
{
  /* walk the tree and print for all blocks */
  kaapi_hws_level_t* level;
  kaapi_ws_block_t* block;
  kaapi_processor_id_t kid;
  kaapi_hws_levelid_t levelid;
  unsigned int i;
  unsigned long val;

  printf("= HWS_STEAL_COUNTERS\n");

  for (levelid = 0; levelid < hws_level_count; ++levelid)
  {
    level = &hws_levels[levelid];
    block = level->blocks;

    /* have to prinr the local pop counters even if flat not set */
    if (!kaapi_hws_is_levelid_set(levelid)) continue ;

    printf("\n- %s\n", kaapi_hws_levelid_to_str(levelid));

    /* print used kids */
    printf("CPU ");
    i = 3;
    for (kid = 0; kid < KAAPI_MAX_PROCESSOR; ++kid)
      if (is_kid_used(kid))
      {
	printf("%08u ", kid_to_cpu(kid));
	i += 9;
      }
    printf("\n");
    for (; i; --i) printf("-");
    printf("\n");

    /* print steal counters */
    for (i = 0; i < level->block_count; ++i, ++block)
    {
      /* block count */
      printf("#%02u", i);

      /* dont print steal counters if FLAT not set */
      if (levelid == KAAPI_HWS_LEVELID_FLAT)
	if (!kaapi_hws_is_levelid_set(KAAPI_HWS_LEVELID_FLAT))
	  continue ;

      for (kid = 0; kid < KAAPI_MAX_PROCESSOR; ++kid)
      {
	if (is_kid_used(kid) == 0) continue ;

	val = KAAPI_ATOMIC_READ(&block->queue->steal_counters[kid]);
	if (val) printf(" %08lu", val);
	else printf("         ");
      }

      printf("\n");
    }
    printf("\n");
  }

  /* flat level */
  printf("= HWS_POP_COUNTERS\n");
  printf("- KAAPI_HWS_LEVELID_FLAT\n");

  level = &hws_levels[KAAPI_HWS_LEVELID_FLAT];
  block = level->blocks;

  /* print used kids */
  printf("CPU ");
  i = 3;
  for (kid = 0; kid < KAAPI_MAX_PROCESSOR; ++kid)
    if (is_kid_used(kid))
    {
      printf("%08u ", kid_to_cpu(kid));
      i += 9;
    }
  printf("\n");
  for (; i; --i) printf("-");
  printf("\n");

  for (i = 0; i < level->block_count; ++i, ++block)
  {
    /* print pop counters */
    val = KAAPI_ATOMIC_READ(&block->queue->pop_counter);
    if (val) printf("%08lu", val);
    printf("\n");
  }
}


#endif /* CONFIG_HWS_COUNTERS */
