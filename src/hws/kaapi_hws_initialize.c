/* todo
   . initial splitting: local request list + splitter + push in right queues
 */


#include <stdio.h>
#include <string.h>
#include <sys/types.h>

#include "kaapi_impl.h"
#include "kaapi_procinfo.h"
#include "kaapi_hws.h"
#include "kaapi_ws_queue.h"


static const kaapi_hws_levelmask_t hws_default_levelmask =
  KAAPI_HWS_LEVELMASK_NUMA |
  KAAPI_HWS_LEVELMASK_SOCKET |
  KAAPI_HWS_LEVELMASK_MACHINE |
  KAAPI_HWS_LEVELMASK_FLAT;


const char* kaapi_hws_levelid_to_str(kaapi_hws_levelid_t levelid)
{
  static const char* const strs[] =
  {
    "KAAPI_HWS_LEVELID_L3",
    "KAAPI_HWS_LEVELID_NUMA",
    "KAAPI_HWS_LEVELID_SOCKET",
    "KAAPI_HWS_LEVELID_MACHINE",
    "KAAPI_HWS_LEVELID_FLAT"
  };

  return strs[(unsigned int)levelid];
}


static kaapi_hws_levelid_t str_to_levelid(const char* str, unsigned int len)
{
#define STATIC_STREQ(__fu, __bar, __baz) \
  ((__bar == (sizeof(__baz) - 1)) && (memcmp(__fu, __baz, __bar) == 0))

  if (STATIC_STREQ(str, len, "L3"))
    return KAAPI_HWS_LEVELID_L3;
  else if (STATIC_STREQ(str, len, "NUMA"))
    return KAAPI_HWS_LEVELID_NUMA;
  else if (STATIC_STREQ(str, len, "SOCKET"))
    return KAAPI_HWS_LEVELID_SOCKET;
  else if (STATIC_STREQ(str, len, "MACHINE"))
    return KAAPI_HWS_LEVELID_MACHINE;
  else if (STATIC_STREQ(str, len, "FLAT"))
    return KAAPI_HWS_LEVELID_FLAT;

  return KAAPI_HWS_LEVELID_MAX;
}


static kaapi_hws_levelmask_t levelmask_from_env(void)
{
  const char* s = getenv("KAAPI_HWS_LEVELS");
  const char* p = s;
  kaapi_hws_levelmask_t levelmask = 0;
  kaapi_hws_levelid_t levelid;

  if (s == NULL) return hws_default_levelmask;

  while (1)
  {
    if ((*s == ',') || (*s == 0))
    {
      const size_t len = s - p;

      if (STATIC_STREQ(p, len, "ALL"))
	return KAAPI_HWS_LEVELMASK_ALL;
      else if (STATIC_STREQ(p, len, "NONE"))
	return 0;

      levelid = str_to_levelid(p, len);
      levelmask |= 1 << levelid;
      p = s + 1;
    }

    if (*s == 0) break ;

    ++s;
  }

  return levelmask;
}


__attribute__((unused))
static void print_selftopo_levels(kaapi_processor_t* kproc)
{
  int depth;

  for (depth = 0; depth < kproc->hlevel.depth; ++depth)
  {
    const unsigned int nkids = kproc->hlevel.levels[depth].nkids;
    const kaapi_processor_id_t* kids = kproc->hlevel.levels[depth].kids;
    unsigned int i;

    printf("%u level[%u]: ", kproc->kid, depth);
    for (i = 0; i < nkids; ++i)
    {
      const unsigned int cpu = kaapi_default_param.kid2cpu[kids[i]]; 
      printf(" %u(%u)", kids[i], cpu);
    }
    printf("\n");
  }

  printf("\n");
}

__attribute__((unused))
static void print_hws_levels(void)
{
  kaapi_hws_levelid_t levelid;

  for (levelid = 0; levelid < hws_level_count; ++levelid)
  {
    kaapi_hws_level_t* const level = &hws_levels[levelid];
    kaapi_ws_block_t* block = level->blocks;
    unsigned int i;

    if (!kaapi_hws_is_levelid_set(levelid)) continue ;

    printf("-- level: %s, #%u\n",
	   kaapi_hws_levelid_to_str(levelid),
	   level->block_count);

    for (i = 0; i < level->block_count; ++i, ++block)
    {
      unsigned int j;

      /* is there actually a kid in this block */
      if (block->kid_count == 0) continue ;

      printf("  -- block[%u] #%u: ", i, block->kid_count);
      for (j = 0; j < block->kid_count; ++j)
	printf(" %u", kaapi_default_param.kid2cpu[block->kids[j]]);
      printf("\n");
    }
  }
}


int kaapi_hws_init_perproc(kaapi_processor_t* kproc)
{
  /* assume kaapi_processor_computetopo called */

  /* print_selftopo_levels(kproc); */

  return 0;
}


int kaapi_hws_fini_perproc(kaapi_processor_t* kproc)
{
  /* assume kaapi_hws_initialize called */
  return 0;
}


int kaapi_hws_init_global(void)
{
  /* assume kaapi_hw_init() called */

  /* build stealing blocks. redundant with kaapi_processor_computetopo. */

  const unsigned int kid_count = kaapi_default_param.kproc_list->count;

  kaapi_hws_level_t* hws_level;

  int depth;
  kaapi_hierarchy_one_level_t flat_level;
  kaapi_hierarchy_one_level_t* one_level;
  kaapi_affinityset_t flat_affin_set[KAAPI_MAX_PROCESSOR];

  hws_levelmask = levelmask_from_env();

  hws_levels = malloc(hws_level_count * sizeof(kaapi_hws_level_t));
  kaapi_assert(hws_levels);

  /* create the flat level even if not masked, since a local
     queue is always required an no longer stored in the kproc.
   */
  /* if (hws_levelmask & KAAPI_HWS_LEVELMASK_FLAT) */
  {
    /* build a flat level containing all the kids */

    unsigned int i;

    for (i = 0; i < kid_count; ++i)
    {
      kaapi_affinityset_t* const affin_set = &flat_affin_set[i];
      kaapi_procinfo_t* pos = kaapi_default_param.kproc_list->head;
      kaapi_cpuset_clear(&affin_set->who);
      for (; pos != NULL; pos = pos->next)
	kaapi_cpuset_set(&affin_set->who, pos->bound_cpu);
      affin_set->ncpu = kid_count;
    }

    flat_level.count = kid_count;
    flat_level.affinity = flat_affin_set;
    flat_level.levelid = KAAPI_HWS_LEVELID_FLAT;
    one_level = &flat_level;

    /* this level is not part of the hwloc topo */
    depth = -1;

    goto add_hws_level;
  }

  /* foreach non filtered discovered level, create a hws_level */
  for (depth = 0; depth < kaapi_default_param.memory.depth; ++depth)
  {
    unsigned int node_count;
    unsigned int node;

    one_level = &kaapi_default_param.memory.levels[depth];

    if (!kaapi_hws_is_levelid_set(one_level->levelid)) continue ;

  add_hws_level:
    hws_level = &hws_levels[one_level->levelid];

    node_count = one_level->count;

    /* allocate steal blocks for this level */
    hws_level->block_count = node_count;
    hws_level->blocks = malloc(node_count * sizeof(kaapi_ws_block_t));
    kaapi_assert(hws_level->blocks);
    hws_level->kid_to_block = malloc(kid_count * sizeof(kaapi_ws_block_t*));
    kaapi_assert(hws_level->kid_to_block);

    /* debug only */
    memset(hws_level->kid_to_block, 0, kid_count * sizeof(kaapi_ws_block_t*));

    /* foreach node at level */
    for (node = 0; node < node_count; ++node)
    {
      kaapi_affinityset_t* const affin_set = &one_level->affinity[node];
      kaapi_ws_block_t* const block = &hws_level->blocks[node];
      kaapi_procinfo_t* pos = kaapi_default_param.kproc_list->head;
      unsigned int i = 0;

      /* initialize the block */
      /* todo: allocate on a page boundary pinned on the node */
      kaapi_sched_initlock(&block->lock);

      block->kids = malloc(affin_set->ncpu * sizeof(kaapi_processor_id_t));
      kaapi_assert(block->kids);

      block->queue = kaapi_ws_queue_create_lifo();
      kaapi_assert(block->queue);

#if CONFIG_HWS_COUNTERS
      memset(block->steal_counters, 0, sizeof(block->steal_counters));
      memset(&block->pop_counter, 0, sizeof(block->pop_counter));
#endif /* CONFIG_HWS_COUNTERS */

      /* for each cpu in this node */
      for (; pos != NULL; pos = pos->next)
      {
	if (!kaapi_cpuset_has(&affin_set->who, pos->bound_cpu))
	  continue ;

	hws_level->kid_to_block[pos->kid] = block;
	block->kids[i] = pos->kid;

	++i;

      } /* foreach cpu in node */

      /* set to the actual kid count */
      block->kid_count = i;

    } /* foreach node in level */

  } /* foreach level in topo */

#if 1 /* debug */
  print_hws_levels();
#endif /* debug */

  return 0;
}


int kaapi_hws_fini_global(void)
{
  /* todo: release maps, blocks, levels */
  return 0;
}
