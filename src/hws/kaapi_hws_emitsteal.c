/*
 ** kaapi_hws_emitsteal.c
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

/* todo: kaapi_hws_level_iterator_t
 */
#include "kaapi_impl.h"
#include "kaapi_hws.h"
#include "kaapi_ws_queue.h"


static inline kaapi_ws_block_t* get_self_ws_block(
  kaapi_processor_t* self, 
  kaapi_hws_levelid_t levelid
)
{
  /* return the ws block for the kproc at given level */
  /* assume self, level dont overflow */
  
  return hws_levels[levelid].kid_to_block[self->kid];
}

/*
*/
static void fail_requests
(
  kaapi_listrequest_t* lr,
  kaapi_listrequest_iterator_t* lri
)
{
  kaapi_request_t* req;
  
  req = kaapi_listrequest_iterator_get(lr, lri);
  while (req != NULL)
  {
    kaapi_request_replytask( req, KAAPI_REQUEST_S_NOK);
    req = kaapi_listrequest_iterator_next(lr, lri);
  }
}


/* Steal a bloc of the hierarchy using the aggregation protocol.
   (1) request was already posted
   (2) acquire the lock and return if some body has replied
   (3) steal work inside the internal queue
   (3) release the lock
*/
static kaapi_request_status_t steal_block
(
  kaapi_ws_block_t* block,
  kaapi_processor_t* kproc,
  kaapi_request_t* request,
  kaapi_listrequest_t* lr,
  kaapi_listrequest_iterator_t* lri
)
{
  /* try lock and return if reply */
  while (!kaapi_ws_lock_trylock(&block->lock))
  {
    if (kaapi_request_test(request))
    {
      /* need to do memory barrier here before reading the data */
      goto on_request_replied;
    }
  }
  
  /* got the lock: reply and unlock */
  kaapi_listrequest_iterator_update(lr, lri, &block->kid_mask);
  
  /* check bits consistency */
#if 0
  {
    unsigned int i = 0;
    for (i =0 ; i < 64; ++i)
    {
      if (!(lri->bitmap.proc32 & (1 << i))) continue ;
      if (!(isin & (1 << i)))
      {
	printf("INVALID\n");
	fflush(stdout);
	while (1);
      }
    }
  }
#endif
  
  if (!kaapi_listrequest_iterator_empty(lri))
    kaapi_ws_queue_steal(block, block->queue, lr, lri);
  
  /* do not need: kaapi_request_syncdata(request); it is myself that replied
  */
//TGTG  kaapi_ws_lock_unlock(&block->lock);
  
on_request_replied:
  return kaapi_request_status(request);
}



/* This is the flat random steal:
   - select a random WS block into the leave and steal it.
*/
static kaapi_request_status_t steal_block_leaves
(
  kaapi_ws_block_t* block,
  kaapi_processor_t* kproc,
  kaapi_request_t* request,
  kaapi_listrequest_t* lr,
  kaapi_listrequest_iterator_t* lri
)
{
  /* steal randomly amongst the block leaves */
  kaapi_processor_id_t kid;
  kaapi_ws_block_t* leaf_block;
  
  /* actually block->kid_count == 1 */
//  if (block->kid_count <= 1) 
//    return KAAPI_REQUEST_S_NOK;
  
redo_rand:
#warning "USE rand_r"
  kid = block->kids[rand() % block->kid_count];
//  if (kid == kproc->kid) goto redo_rand;
  
  /* get the leaf block (ie. block at flat level) */
  leaf_block = hws_levels[KAAPI_HWS_LEVELID_FLAT].kid_to_block[kid];
  
  return steal_block(leaf_block, kproc, request, lr, lri);
}


/* Try to steal on each WS block at a given level.
   Stop until somebody (or may self) has replied successfully.
*/
static kaapi_request_status_t steal_level
(
  kaapi_hws_level_t*            level,
  kaapi_processor_t*            kproc,
  kaapi_request_t*              request,
  kaapi_listrequest_t*          lr,
  kaapi_listrequest_iterator_t* lri
)
{
  unsigned int i;
  
  for (i = 0; i < level->block_count; ++i)
  {
    if (KAAPI_REQUEST_S_OK == steal_block(&level->blocks[i], kproc, request, lr, lri))
      return KAAPI_REQUEST_S_OK;
  }
  
  return KAAPI_REQUEST_S_NOK;
}


__attribute__((unused))
static kaapi_thread_context_t* pop_block
(
  kaapi_ws_block_t* block,
  kaapi_processor_t* kproc
)
{

//DEBUG:
kaapi_assert(0);

  /* not a real steal operation, dont actually post */
  kaapi_request_t* const req = &hws_requests.requests[kproc->kid];
  kaapi_task_t*          thief_task;
  kaapi_tasksteal_arg_t* thief_sp;
  kaapi_ws_error_t err;
 
  kaapi_thread_t* const self_thread =
      kaapi_threadcontext2thread(kproc->thread);
  thief_task = kaapi_thread_toptask( self_thread );
  thief_sp = kaapi_thread_pushdata(self_thread, sizeof(kaapi_tasksteal_arg_t));
  kaapi_task_init(  thief_task, 
                    kaapi_tasksteal_body, 
                    thief_sp
  );
  req->status       = KAAPI_REQUEST_S_POSTED;
  req->ident        = kproc->kid;
  req->thief_task   = thief_task;
  thief_task->state = KAAPI_TASK_STATE_ALLOCATED;
  req->thief_sp     = thief_sp;
  
  err = kaapi_ws_queue_pop(block, block->queue, req);
  if (err != KAAPI_WS_ERROR_SUCCESS) return NULL;
  
  switch (kaapi_request_status(req))
  {      
    case KAAPI_REQUEST_S_OK:
    {
      kaapi_thread_pushtask(self_thread);
      
#if defined(KAAPI_USE_PERFCOUNTER)
      ++KAAPI_PERF_REG(kproc, KAAPI_PERF_ID_STEALREQOK);
#endif
      return kproc->thread;
      
    } /* KAAPI_REPLY_S_TASK */

    default: break ;
  }
  
  return NULL;
}


static kaapi_request_t* post_request
(kaapi_processor_t* kproc, kaapi_atomic_t* status)
{
  kaapi_request_t* const req = &hws_requests.requests[kproc->kid];

#if defined(KAAPI_DEBUG)
  req->version = ++kproc->req_version;
#endif
  
  /* from kaapi_mt_machine.h/kaapi_request_post */
  req->ident             = kproc->kid;
  req->thief_task        = &kproc->thread->stealreserved_task;
  req->thief_task->state = KAAPI_TASK_STATE_ALLOCATED;
  req->thief_task->body  = kaapi_tasksteal_body;
  req->thief_sp          = &kproc->thread->stealreserved_arg;
  req->status            = status;
  KAAPI_ATOMIC_WRITE_BARRIER(status, KAAPI_REQUEST_S_POSTED);
  kaapi_bitmap_set(&hws_requests.bitmap, kproc->kid);
  return req;
}


/* Hierarchical work stealing strategy
*/
kaapi_request_status_t kaapi_hws_emitsteal( kaapi_processor_t* kproc )
{
  kaapi_ws_block_t* block;
  kaapi_hws_levelid_t child_levelid;
  kaapi_hws_levelid_t levelid = 0;
  kaapi_request_t* request;
  kaapi_listrequest_iterator_t lri;
  kaapi_atomic_t status;

#if 0 /* already done by the caller */
  /* pop locally without emitting request */
  /* todo: kaapi_ws_queue_pop should fit the steal interface */
  block = get_self_ws_block(kproc, KAAPI_HWS_LEVELID_FLAT);
  thread = pop_block(block, kproc);
  if (thread != NULL) return thread;
#endif /* already done by the caller */

  /* dont fail_request with an uninitialized bitmap */
  kaapi_listrequest_iterator_prepare(&lri);

  /* post the stealing request */
  kproc->issteal = 1;
  request = post_request(kproc, &status);
  /* from here anybody else may reply to the request */

  /* foreach parent level, pop. if pop failed, steal in level children. */
redo_levels:
  for (levelid = KAAPI_HWS_LEVELID_FIRST; levelid < (int)hws_level_count; ++levelid)
  {
    if (!(kaapi_hws_is_levelid_set(levelid))) continue ;
    
    block = get_self_ws_block(kproc, levelid);
    
    /* dont steal at flat level during ascension */
    if (levelid != KAAPI_HWS_LEVELID_FLAT)
    {
      /* todo: this is a pop, not a steal */
      /* todo: dont rely upon thread for termination condition */
      if (KAAPI_REQUEST_S_OK == steal_block(block, kproc, request, &hws_requests, &lri))
        goto on_request_success;
      
      /* popping failed at this level, steal in level children */
      for (child_levelid = levelid - 1; child_levelid >= 0; --child_levelid)
      {
        kaapi_hws_level_t* const child_level = &hws_levels[child_levelid];
        if (!kaapi_hws_is_levelid_set(child_levelid)) 
          continue;
        
        if (KAAPI_REQUEST_S_OK == steal_level(child_level, kproc, request, &hws_requests, &lri))
          goto on_request_success;
      }
      
    } /* levelid != KAAPI_HWS_LEVELID_FLAT */

    /* child level stealing failed, steal in block leaf local queues */
    if (KAAPI_REQUEST_S_OK == steal_block_leaves(block, kproc, request, &hws_requests, &lri))
      goto on_request_success;
          
    /* next level */
  }

//  leave();

  fail_requests(&hws_requests, &lri);
  kaapi_assert( kaapi_listrequest_iterator_empty( &lri ) );
#if defined(KAAPI_DEBUG) 
  kaapi_assert( lri.count_in == lri.count_out );
#endif
  kaapi_request_status_t fu;
  {
    kaapi_assert(request->status == &status);
    if (fu == KAAPI_REQUEST_S_OK)
      goto on_request_success;
    else if (kaapi_isterminated()) 
    {
      fail_requests(&hws_requests, &lri);
      kaapi_assert( kaapi_listrequest_iterator_empty( &lri ) );
      
      fu = kaapi_request_status(request);
      if (fu == KAAPI_REQUEST_S_OK)
        goto on_request_success;
#if 0
      while (fu == KAAPI_REQUEST_S_POSTED)
      {
        fu = kaapi_request_status(request);
        kaapi_slowdown_cpu();
      }
#endif
    }
    else if (fu != KAAPI_REQUEST_S_NOK)
      goto redo_levels;
    /* terminaison ? */
    /* else fail */
  }

  kaapi_assert(request->status == &status);
  kaapi_assert(kaapi_request_status(request) == KAAPI_REQUEST_S_NOK);
  
  kproc->issteal = 0;

#if defined(KAAPI_DEBUG) 
  kproc->thief_task = 0;
  ++kproc->reply_version;
#endif
  kaapi_assert(request->status == &status);
  kaapi_assert(kaapi_request_status(request) == KAAPI_REQUEST_S_NOK);
#if defined(KAAPI_DEBUG) 
  request->status = 0;
  kaapi_assert( lri.count_in == lri.count_out );
#endif

  return KAAPI_REQUEST_S_NOK;


on_request_success:
//  leave();
#if 0
  kaapi_assert(!(hws_requests.bitmap.proc32._counter & (1 << kaapi_get_self_kid())));
#endif

  fail_requests(&hws_requests, &lri);
  kaapi_assert( kaapi_listrequest_iterator_empty( &lri ) );
  kaapi_assert(kaapi_request_status(request) == KAAPI_REQUEST_S_OK);
#if defined(KAAPI_DEBUG) 
  kaapi_assert( lri.count_in == lri.count_out );
#endif

  kproc->issteal = 0;
#if defined(KAAPI_DEBUG) 
  ++kproc->reply_version;
#endif
  /* update task to execute */
  kaapi_request_syncdata(request);
  kproc->thief_task = request->thief_task;
#if defined(KAAPI_USE_PERFCOUNTER)
  ++KAAPI_PERF_REG(kproc, KAAPI_PERF_ID_STEALREQOK);
#endif
#if 0
  printf("%i::OUT Emitsteal request replied: %p\n", kproc->kid, request ); 
  fflush(stdout);
#endif
#if defined(KAAPI_DEBUG)
  kproc->compute_version = kproc->req_version;
#endif
#if defined(KAAPI_DEBUG) 
  request->status = 0;
  kaapi_assert( lri.count_in == lri.count_out );
#endif
  return KAAPI_REQUEST_S_OK;
}