#include <string.h>
#include "kaapi_impl.h"
#include "kaapi_hws.h"


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

  kaapi_hws_levelid_t levelid;

  printf("== HWS_COUNTERS\n");

  for (levelid = 0; levelid < hws_level_count; ++levelid)
  {
    kaapi_hws_level_t* const level = &hws_levels[levelid];
    kaapi_ws_block_t* block = level->blocks;
    unsigned int i;

    /* have to prinr the local pop counters even if flat not set */
    if (levelid != KAAPI_HWS_LEVELID_FLAT)
      if (!kaapi_hws_is_levelid_set(levelid)) continue ;

    printf("-- level %s\n", kaapi_hws_levelid_to_str(levelid));

    for (i = 0; i < level->block_count; ++i, ++block)
    {
      kaapi_processor_id_t kid;

      printf("BLOCK: %lx\n", (unsigned long)block);

      printf("--- #pop  : %08u\n", KAAPI_ATOMIC_READ(&block->pop_counter));

      /* dont print steal counters if FLAT not set */
      if (levelid == KAAPI_HWS_LEVELID_FLAT)
	if (!kaapi_hws_is_levelid_set(KAAPI_HWS_LEVELID_FLAT))
	  continue ;

      printf("--- #steal:");

      for (kid = 0; kid < KAAPI_MAX_PROCESSOR; ++kid)
      {
	if (is_kid_used(kid) == 0) continue ;
	printf(" (%02u)%08u", kid_to_cpu(kid),
	       KAAPI_ATOMIC_READ(&block->steal_counters[kid]));
      }

      printf("\n");
    }

    printf("\n");
  }
}


#endif /* CONFIG_HWS_COUNTERS */
