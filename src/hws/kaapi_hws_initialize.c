/*
** kaapi_hws_initialize.c
** xkaapi
** 
** Created on Tue Mar 31 15:19:14 2009
** Copyright 2009 INRIA.
**
** Contributors :
**
** thierry.gautier@inrialpes.fr
** fabien.lementec@gmail.com / fabien.lementec@imag.fr
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


#include <stdio.h>
#include <string.h>
#include <sys/types.h>

#include "kaapi_impl.h"
#include "kaapi_procinfo.h"

/* include before ws_queue, for CONFIG_HWS_COUNTERS */
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

      printf("  -- block[%u, phys=%u] #%u: ", i, i, block->kid_count);
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

  /* assume kid_count is (kid_max - 1) */
  const unsigned int kid_count = kaapi_default_param.kproc_list->count;

  kaapi_procinfo_t* pos;
  kaapi_hws_level_t* hws_level;
  kaapi_processor_id_t kid;
  int depth;
  unsigned int i;
  kaapi_hierarchy_one_level_t flat_level;
  kaapi_hierarchy_one_level_t* one_level;
  kaapi_affinityset_t flat_affin_set[KAAPI_MAX_PROCESSOR];

  /* toremove */
  kaapi_hws_sched_init_sync();
  /* toremove */

  /* initialize the request list */
  kaapi_bitmap_clear(&hws_requests.bitmap);
  for (kid = 0; kid < KAAPI_MAX_PROCESSOR; ++kid)
    hws_requests.requests[kid].kid = kid;

  hws_levelmask = levelmask_from_env();

  hws_levels = malloc(hws_level_count * sizeof(kaapi_hws_level_t));
  kaapi_assert(hws_levels);

  /* create the flat level even if not masked, since a local queue
     is always required and stack no longer stored in the kproc.
     the flat level contains one block per kid. each kid belongs to
     the block. special care is taken later to initialize kid_to_block 
   */

  for (i = 0; i < kid_count; ++i)
  {
    kaapi_affinityset_t* const affin_set = &flat_affin_set[i];
    kaapi_cpuset_clear(&affin_set->who);
    pos = kaapi_default_param.kproc_list->head;
    for (; pos != NULL; pos = pos->next)
      kaapi_cpuset_set(&affin_set->who, pos->bound_cpu);
    affin_set->ncpu = kid_count;
    affin_set->os_index = (unsigned int)i;
  }

  flat_level.count = kid_count;
  flat_level.affinity = flat_affin_set;
  flat_level.levelid = KAAPI_HWS_LEVELID_FLAT;
  one_level = &flat_level;

  /* this level is not part of the hwloc topo */
  depth = -1;

  /* bypass first iteration to add the flat level */
  goto add_hws_level;

  /* foreach non filtered discovered level, create a hws_level */
  for (depth = 0; depth < kaapi_default_param.memory.depth; ++depth)
  {
    unsigned int node_count;
    unsigned int lnode; /* the hwloc::LOGICAL node index */

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
    for (lnode = 0; lnode < node_count; ++lnode)
    {
      kaapi_affinityset_t* const affin_set = &one_level->affinity[lnode];

      /* use the hwloc::PHYSICAL node as the hws index */
      const unsigned int pnode = affin_set->os_index;

      kaapi_ws_block_t* const block = &hws_level->blocks[pnode];
      kaapi_procinfo_t* pos;
      unsigned int i = 0;

      kaapi_assert(pnode < node_count);

      /* flat level special handling */
      if (one_level->levelid == KAAPI_HWS_LEVELID_FLAT)
	hws_level->kid_to_block[pnode] = block;

      /* initialize the block */
      /* todo: allocate on a page boundary pinned on the node */
      kaapi_ws_lock_init(&block->lock);

      block->kids = malloc(affin_set->ncpu * sizeof(kaapi_processor_id_t));
      kaapi_assert(block->kids);

      kaapi_bitmap_value_clear(&block->kid_mask);

      if (one_level->levelid == KAAPI_HWS_LEVELID_FLAT)
      {
	/* lnode corresponds to the kid */
	kaapi_assert(kaapi_all_kprocessors[lnode]);
	block->queue = kaapi_ws_queue_create_kproc(kaapi_all_kprocessors[lnode]);
      }
      else
      {
	block->queue = kaapi_ws_queue_create_lifo();
      }
      kaapi_assert(block->queue);

      /* for each cpu in this node */
      for (pos = kaapi_default_param.kproc_list->head; pos; pos = pos->next)
      {
	if (!kaapi_cpuset_has(&affin_set->who, pos->bound_cpu))
	  continue ;

	if (one_level->levelid != KAAPI_HWS_LEVELID_FLAT)
	{
	  /* assume kid_to_block[pos->kid] == 0 */
	  hws_level->kid_to_block[pos->kid] = block;
	}

	block->kids[i] = pos->kid;

	kaapi_bitmap_value_set(&block->kid_mask, pos->kid);

	++i;

      } /* foreach cpu in node */

      /* set to the actual kid count */
      block->kid_count = i;

    } /* foreach node in level */

  } /* foreach level in topo */

#if 0 /* debug */
  print_hws_levels();
#endif /* debug */

  return 0;
}


int kaapi_hws_fini_global(void)
{
  /* todo: release maps, blocks, levels */
  return 0;
}
