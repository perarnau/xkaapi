/* desgin
   . hlrequest contains on request per participant and a map
   -> create_hl_requests(kids);

   . one queue per kproc
   . one queue per level
   algorithm
   . steal in level queues first
   -> aggregation + lock
   . then steal in all the kproc belonging to a given level
   -> kaapi_sched_emitsteal_with_kids(kids)
 */

/* TODO
   . one request block per memory level,block having more than one kid
   . build a kaapi_listrequest_iterator as the levels are walked
   . allocate data in local pages
   . ws_queue interface error codes
 */

/* todo

   queue container
   . there must be one queue per kproc per memory level
   push(task, numa) is pushing the task in queue(self->kid, level)
   . queue_map[level, kid]
   . {get,set}_self_queue

   stealing protocol
   . select a random queue, q = queue_map(level, node, kid)
   . post a request at to q
   . synchronize concurrent stealing on q->lock
   . locking is having access to all the queues

   task push at level
   . refer to above comment
 */

/* notes
   2 possibilities
   . either a per kproc queue, per level
   -> currently the case for flat stealing
   -> a lock is put to synchronize
   . or a per level queue
   -> 
 */

#include <stdio.h>
#include <string.h>

#include "kaapi_impl.h"
#include "kaapi_procinfo.h"
#include "kaapi_ws_queue.h"


#if 1 /* todo: replace by kaapi_request_t */

typedef struct kaapi_ws_request
{
  void* reply_area;
} kaapi_ws_request_t;

static void init_request(kaapi_ws_request_t* r)
{
}

static void post_request(kaapi_ws_request_t* r)
{
}

static int test_reply(kaapi_ws_request_t* r)
{
  return 0;
}

#endif /* todo */


typedef struct kaapi_ws_block
{
  /* concurrent workstealing sync */
  /* todo: cache aligned, alone in the line */
  kaapi_atomic_t lock;

  /* workstealing queue */
  kaapi_ws_queue_t* queue;

  /* kid map of all the participants */
  kaapi_processor_id_t* kids;
  unsigned int kid_count;

  /* one request per kid */
  kaapi_ws_request_t* requests;

  /* req = kid_to_req[kid] map */
  /* todo: this is equivalent hlrequest in kproc */
  kaapi_ws_request_t** kid_to_req;

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

  unsigned int depth;
  kaapi_hierarchy_one_level_t flat_level;
  kaapi_affinityset_t flat_affin_set[KAAPI_MAX_PROCESSOR];

  /* add 1 for the flat level */
  hws_level_count = kaapi_default_param.memory.depth + 1;
  hws_levels = malloc(hws_level_count * sizeof(kaapi_hws_level_t));
  kaapi_assert(hws_levels);

  /* foreach level */
  for (depth = 0; depth < hws_level_count; ++depth)
  {
    kaapi_hws_level_t* const hws_level = &hws_levels[depth];
    kaapi_hierarchy_one_level_t* one_level;
    unsigned int node_count;

    if (depth == (hws_level_count - 1))
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
      one_level = &flat_level;
    }
    else /* != flat level */
    {
      one_level = &kaapi_default_param.memory.levels[depth];
    }

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
    unsigned int node;
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

      block->requests = malloc(affin_set->ncpu * sizeof(kaapi_ws_request_t));
      kaapi_assert(block->requests);

      block->kid_to_req = malloc(kid_count * sizeof(kaapi_ws_request_t*));
      kaapi_assert(block->kid_to_req);

      block->queue = kaapi_ws_queue_create_lifo();
      kaapi_assert(block->queue);

      /* for each cpu in this node */
      for (; pos != NULL; pos = pos->next)
      {
	if (!kaapi_cpuset_has(&affin_set->who, pos->bound_cpu))
	  continue ;

	hws_level->kid_to_block[pos->kid] = block;
	block->kids[i] = pos->kid;

	/* todo: initialize the request */
	init_request(&block->requests[i]);

	/* update the request mapping table */
	block->kid_to_req[pos->kid] = &block->requests[i];

	++i;

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

static kaapi_thread_context_t* steal_level_queue
(kaapi_processor_t* kproc, kaapi_ws_block_t* block)
{
  return NULL;
}

static kaapi_thread_context_t* steal_kproc_queues
(kaapi_processor_t* kproc, kaapi_ws_block_t* block)
{
#if 0
  /* todo */
  return kaapi_sched_emitsteal_with_kids(block->kids, block->kid_count);
#else
  return NULL;
#endif
}

kaapi_thread_context_t* kaapi_hws_emitsteal(kaapi_processor_t* kproc)
{
  kaapi_ws_block_t* block;
  kaapi_processor_id_t victim_kid = (kaapi_processor_id_t)-1;
  unsigned int level = 0;
  kaapi_ws_request_t* req;

  for (; level < hws_level_count; ++level)
  {
    block = get_self_ws_block(kproc, level);

    /* steal in the level queue first */
    
    /* next_level if alone in the block */
    if (block->kid_count <= 1) continue ;

    victim_kid = select_victim(block, kproc->kid);

    /* emit the request */
    req = block->kid_to_req[kproc->kid];
    post_request(req);

    /* wait for lock or reply */
    while (1)
    {
      if (kaapi_sched_trylock(&block->lock))
      {
	/* got the lock, reply all. if there is no
	   task to extract for this level, go next.
	 */

	kaapi_ws_queue_t* const queue = block->queue;
	const kaapi_ws_error_t error = kaapi_ws_queue_stealn(queue, NULL);
	if (error == KAAPI_WS_ERROR_EMPTY)
	{
	  goto next_level;
	}
      }

      if (test_reply(req))
      {
	/* request got replied */
	goto on_replied;
      }

    }

  next_level: ;
  }

 on_replied:
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
