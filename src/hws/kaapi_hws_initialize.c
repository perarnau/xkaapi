#include <stdio.h>
#include <string.h>

#include "kaapi_impl.h"
#include "kaapi_procinfo.h"
#include "kaapi_ws_queue.h"


#if 0 /* unused */
typedef struct kaapi_hws_request
{
  void* reply_area;

} kaapi_hws_request_t;
#endif /* unused */


typedef struct kaapi_ws_block
{
  /* kid map of all the participants */
  kaapi_processor_id_t* kids;
  unsigned int kid_count;

  /* concurrent workstealing sync */
  kaapi_atomic_t lock;

  /* workstealing queue */
  kaapi_ws_queue_t* queue;
  
#if 0
  kaapi_ws_request_t* requests;
#endif

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


static inline kaapi_ws_block_t* get_self_ws_block
(kaapi_processor_t* self, unsigned int level)
{
  /* return the ws block for the kproc at given level */
  /* assume self, level dont overflow */

  return hws_levels[level].kid_to_block[self->kid];
}

static kaapi_processor_id_t select_victim
(kaapi_ws_block_t* block, kaapi_processor_id_t self_kid)
{
  /* select a victim in the block */
  /* assume block->kid_count > 1 */

  unsigned int kid;

 redo_rand:
  kid = block->kids[rand() % block->kid_count];
  if (kid == self_kid) goto redo_rand;
  return (kaapi_processor_id_t)kid;
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
      kaapi_sched_initlock(&block->lock);

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
  /* todo: release maps, blocks, levels */
  return 0;
}


/* hierarchical workstealing request emission
 */

kaapi_thread_context_t* kaapi_hws_emitsteal(kaapi_processor_t* kproc)
{
  kaapi_ws_block_t* block;
  kaapi_processor_id_t victim_kid = (kaapi_processor_id_t)-1;
  unsigned int level = 0;

  for (; level < hws_level_count; ++level)
  {
    block = get_self_ws_block(kproc, level);

    /* next_level if alone in the block */
    if (block->kid_count <= 1) continue ;

    victim_kid = select_victim(block, kproc->kid);

#if 0 /* todo */
    /* emit the request */
    post_hws_request();
#endif /* todo */

    /* wait for lock or reply */
    while (1)
    {
      if (kaapi_sched_trylock(&block->lock))
      {
	/* got the lock, reply all. if there is no
	   task to extract for this level, go next.
	 */

#if 0 /* todo */
	kaapi_ws_queue_t* const q = block->queue;
	if (q->stealn(block, reqs) == failed)
	{
	  goto next_level;
	}
#endif /* todo */
      }

#if 0 /* todo */
      if (kaapi_hws_reply_test(reply))
      {
	/* request got replied */
      }
#endif /* todo */

    }
  }

  /* unlock all levels < level */
  if (level > 0)
  {
    printf("[%u] unlocking from %u\n", kproc->kid, level);
    for (; level; --level)
    {
      block = get_self_ws_block(kproc, level - 1);
      if (block->kid_count <= 1) continue ;
      kaapi_sched_unlock(&block->lock);
    }
  }

  printf("[%u] extract replied task\n", kproc->kid);

#if 0 /* TODO */

  /* extract the replied task */
  switch (kaapi_hws_reply_status(reply))
  {
  case KAAPI_REPLY_S_TASK_FMT:
    /* convert fmtid to a task body */
    reply->u.s_task.body = kaapi_format_resolvebyfmit
      (reply->u.s_taskfmt.fmt)->entrypoint[kproc->proc_type];
    kaapi_assert_debug(reply->u.s_task.body);

  case KAAPI_REPLY_S_TASK:
    /* initialize and push the task */
    self_thread = kaapi_threadcontext2thread(kproc->thread);
    kaapi_task_init
    (
     kaapi_thread_toptask(self_thread),
     reply->u.s_task.body,
     (void*)(reply->udata + reply->offset)
    );

    kaapi_thread_pushtask(self_thread);

#if defined(KAAPI_USE_PERFCOUNTER)
    ++KAAPI_PERF_REG(kproc, KAAPI_PERF_ID_STEALREQOK);
#endif
    return kproc->thread;
    break ; /* KAAPI_REPLY_S_TASK */

  case KAAPI_REPLY_S_THREAD:
#if defined(KAAPI_USE_PERFCOUNTER)
    ++KAAPI_PERF_REG(kproc, KAAPI_PERF_ID_STEALREQOK);
#endif
    (*kproc->fnc_select)(kproc, &victim, KAAPI_STEAL_SUCCESS);
    return reply->u.s_thread;
    break ; /* KAAPI_REPLY_S_THREAD */

  case KAAPI_REPLY_S_NOK:
    return 0;
    break ;

  case KAAPI_REPLY_S_ERROR:
    kaapi_assert_debug_m(0, "Error code in request status");
    break ;

  default:
    kaapi_assert_debug_m(0, "Bad request status");
    break ;
  }

#endif /* TODO */

  return 0;
}
