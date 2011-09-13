#include <stdio.h>
#include <string.h>

#include "kaapi_impl.h"
#include "kaapi_procinfo.h"


typedef struct kaapi_ws_block
{
  /* kid map of all the participants */
  kaapi_atomic_t lock;
  kaapi_processor_id_t* kids;
  unsigned int kid_count;
} kaapi_ws_block_t;


typedef struct kaapi_hws_level
{
  /* describes the kids involved a given memory level */
  kaapi_ws_block_t** kid_to_block;
  kaapi_ws_block_t* blocks;
  unsigned int block_count;
} kaapi_hws_level_t;


/* globals
 */

static unsigned int hws_level_count;
static kaapi_hws_level_t* hws_levels;


/* lookup the ws block given
 */

static inline kaapi_ws_block_t* get_self_ws_block
(kaapi_processor_id_t kid, unsigned int level)
{
  /* assume self,level dont overflow */
  return hws_levels[level].kid_to_block[kid];
}


#if 0

/* what is it used for:
 */

static void get_self_siblings
(kaapi_processor_t* self, kaapi_bitmap_value_t* siblings)
{
  /* get all the sibling nodes of a given */
}

#endif

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
  unsigned int depth;

  for (depth = 0; depth < hws_level_count; ++depth)
  {
    kaapi_hws_level_t* const level = &hws_levels[depth];
    kaapi_ws_block_t* block = level->blocks;
    unsigned int i;

    printf("-- depth[%u], #%u\n", depth, level->block_count);

    for (i = 0; i < level->block_count; ++i, ++block)
    {
      unsigned int j;

      /* is there actually a kid in this block */
      if (block->kid_count == 0) continue ;

      printf("  -- block[%u] #%u: ", i, block->kid_count);
      for (j = 0; j < block->kid_count; ++j)
	printf(" %u", block->kids[j]);
      printf("\n");
    }
  }
}


int kaapi_hws_init_perproc(kaapi_processor_t* kproc)
{
  /* assume kaapi_processor_computetopo called */

#if 0
  print_selftopo_levels(kproc);
#endif

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

  kaapi_assert(kaapi_default_param.kproc_list);

  const unsigned int kid_count = kaapi_default_param.kproc_list->count;

  unsigned int depth;

  hws_level_count = kaapi_default_param.memory.depth;
  hws_levels = malloc(hws_level_count * sizeof(kaapi_hws_level_t));
  kaapi_assert(hws_levels);

  /* foreach level */
  for (depth = 0; depth < hws_level_count; ++depth)
  {
    kaapi_hws_level_t* const hws_level = &hws_levels[depth];

    kaapi_hierarchy_one_level_t* const one_level =
      &kaapi_default_param.memory.levels[depth];

    const unsigned int node_count = one_level->count;

    /* allocate steal blocks for this level */
    hws_level->block_count = node_count;
    hws_level->blocks = malloc(node_count * sizeof(kaapi_ws_block_t));
    kaapi_assert(hws_level->blocks);
    hws_level->kid_to_block = malloc(kid_count * sizeof(kaapi_ws_block_t*));
    kaapi_assert(hws_level->kid_to_block);

    /* debug only */
    memset(hws_level->kid_to_block, 0, kid_count * sizeof(kaapi_ws_block_t*));

    /* foreach node at level */
    unsigned int node;
    for (node = 0; node < node_count; ++node)
    {
      kaapi_affinityset_t* const affin_set = &one_level->affinity[node];
      kaapi_ws_block_t* const block = &hws_level->blocks[node];
      unsigned int i = 0;

      /* initialize the block */
      /* todo: allocate on a page boundary pinned on the node */
      KAAPI_ATOMIC_WRITE(&block->lock, 0);

      block->kids = malloc(affin_set->ncpu * sizeof(kaapi_processor_id_t));
      kaapi_assert(block->kids);

      /* for each cpu in this node */
      kaapi_procinfo_t* pos = kaapi_default_param.kproc_list->head;
      for (; pos != NULL; pos = pos->next)
      {
	if (!kaapi_cpuset_has(&affin_set->who, pos->bound_cpu))
	  continue ;

	hws_level->kid_to_block[pos->kid] = block;
	block->kids[i++] = pos->kid;

      } /* foreach cpu in node */

      /* set to the actual kid count */
      block->kid_count = i;

    } /* foreach node in level */
  } /* foreach level in topo */

  /* print_hws_levels(); */

  return 0;
}


int kaapi_hws_fini_global(void)
{
  return 0;
}


/* steal request emission
 */
