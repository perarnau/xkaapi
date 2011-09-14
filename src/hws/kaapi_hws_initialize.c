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
  kaapi_hws_levelid_t levelid;

  kaapi_ws_block_t** kid_to_block;

  kaapi_ws_block_t* blocks;
  unsigned int block_count;

} kaapi_hws_level_t;


/* globals
 */

static const unsigned int hws_level_count = KAAPI_HWS_LEVELID_MAX;
static kaapi_hws_level_t* hws_levels;

/* levelid filtering. todo: parametric */
static const kaapi_hws_levelmask_t hws_levelmask =
  KAAPI_HWS_LEVELMASK_NUMA |
  KAAPI_HWS_LEVELMASK_SOCKET |
  KAAPI_HWS_LEVELMASK_MACHINE |
  KAAPI_HWS_LEVELMASK_FLAT;


static const char* levelid_to_str(kaapi_hws_levelid_t levelid)
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


static inline kaapi_ws_block_t* get_self_ws_block
(kaapi_processor_t* self, unsigned int level)
{
  /* return the ws block for the kproc at given level */
  /* assume self, level dont overflow */

  return hws_levels[level].kid_to_block[self->kid];
}


static int select_block_victim
(
 kaapi_processor_t* kproc,
 kaapi_victim_t* victim,
 kaapi_selecvictim_flag_t flag
)
{
  /* select a victim in the block */
  /* assume block->kid_count > 1 */

  kaapi_ws_block_t* block;
  unsigned int kid;

  if (flag != KAAPI_SELECT_VICTIM) return 0;

  block = kproc->fnc_selecarg[0];

 redo_rand:
  kid = block->kids[rand() % block->kid_count];
  if (kid == kproc->kid) goto redo_rand;

  victim->kproc = kaapi_all_kprocessors[kid];
  victim->level = 0;

  return 0;
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

    if (!(hws_levelmask & (1 << levelid))) continue ;

    printf("-- level: %s, #%u\n", levelid_to_str(levelid), level->block_count);

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

  /* add 1 for the flat level */
  hws_levels = malloc(hws_level_count * sizeof(kaapi_hws_level_t));
  kaapi_assert(hws_levels);

  /* create the flat level if needed */
  if (hws_levelmask & KAAPI_HWS_LEVELMASK_FLAT)
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

    if (!(hws_levelmask & (1 << one_level->levelid))) continue ;

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

  print_hws_levels();
  while (1) ;

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
  kaapi_thread_context_t* thread = NULL;
  kaapi_ws_block_t* block;
  unsigned int level = 0;
  kaapi_ws_request_t* req;

  for (; level < hws_level_count; ++level)
  {
    block = get_self_ws_block(kproc, level);

    /* steal in the level queue first */
    {
    }
    /* steal in the level queue first */

    /* randomly steal in a random kproc local queue */
    if (block->kid_count >= 2)
    {
      kaapi_selectvictim_fnc_t saved_fn = kproc->fnc_select;
      void* saved_arg = kproc->fnc_selecarg[0];

      kproc->fnc_select = select_block_victim;
      kproc->fnc_selecarg[0] = block;
      thread = kaapi_sched_emitsteal(kproc);
      kproc->fnc_select = saved_fn;
      kproc->fnc_selecarg[0] = saved_arg;

      if (thread != NULL)
      {
	goto on_replied;
      }
    }
    /* randomly steal in a random kproc local queue */

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

  return thread;
}


/* push a task at a given hierarchy level
 */

int kaapi_hws_pushtask
(void* task, void* data, kaapi_hws_levelid_t levelid)
{
  kaapi_assert(hws_levelmask & (1 << levelid));

#if 0 /* TODO */
  kaapi_processor_t* const kproc = kaapi_get_current_processor();
  const kaapi_processor_id_t kid = self_proc->kid;
  kaapi_hws_level_t* const level = levelid_to_level[levelid];
  kaapi_ws_block_t* const block = level->blocks[level->kid_to_block[kid]];
  kaapi_ws_queue_t* const queue = block->queue;
  kaapi_ws_queue_push(queue, task, data);
#endif
  return 0;
}
