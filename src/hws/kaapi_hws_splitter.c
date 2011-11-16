/*
 ** kaapi_hws_splitter.c
 ** xkaapi
 ** 
 **
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


#include <string.h>
#include <stdint.h>
#include "kaapi_impl.h"
#include "kaapi_hws.h"
#include "kaapi_ws_queue.h"


int kaapi_hws_splitter
(
 kaapi_stealcontext_t* sc,
 kaapi_task_splitter_t splitter,
 void* args,
 kaapi_hws_levelid_t levelid
 )
{
#if 0
  /* split equivalently among all the nodes of a given level */
  
  kaapi_processor_t* const kproc = kaapi_get_current_processor();
  
  /* todo: dynamic allocation */
  kaapi_request_t reqs[KAAPI_MAX_PROCESSOR];
  kaapi_reply_t reps[KAAPI_MAX_PROCESSOR];
  
  int retval;
  kaapi_hws_level_t* level;
  unsigned int count;
  unsigned int i;
  
  kaapi_assert_debug(kaapi_hws_is_levelid_set(levelid));
  
  level = &hws_levels[levelid];
  count = level->block_count;
  
  kaapi_assert_debug(count < KAAPI_MAX_PROCESSOR);
  
  /* generate a request array and call the splitter */
  for (i = 0; i < count; ++i)
  {
    kaapi_request_t* const req = &reqs[i];
    kaapi_reply_t* const rep = &reps[i];
    
    rep->offset = 0;
    rep->preempt = 0;
    rep->status = KAAPI_REQUEST_S_POSTED;
    
#warning TODO HERE
#if 0
    req->ident = (kaapi_processor_id_t)i;
    req->mapping = 0;
    req->reply = rep;
    req->ktr = NULL;
#endif
  }
  
  retval = splitter(sc, count, reqs, args);
  
  /* foreach replied requests, push in queues */
  for (i = 0; i < count; ++i)
  {
    kaapi_request_t* const req = &reqs[i];
    kaapi_reply_t* const rep = &reps[i];
    kaapi_ws_queue_t* queue;
    kaapi_ws_block_t* ws_block;
    
    if (rep->status == KAAPI_REQUEST_S_POSTED) continue ;
    
    /* extract the task and push in the correct queue */
    ws_block = &hws_levels[levelid].blocks[req->ident];
    queue    = ws_block->queue;
    
    switch (kaapi_request_status(req))
    {
      case KAAPI_REQUEST_S_OK:
      {
#warning TODO HERE
        /* data is stored in the first word of udata */
        void* const dont_break_aliasing = (void*)rep->udata;
        void* const data = *(void**)dont_break_aliasing;
        kaapi_ws_queue_push(ws_block, queue, req->thief_task);
        break ;
        
      } /* KAAPI_REPLY_S_TASK */
        
      default: break ;
    }
  }
  
  return retval;
#endif
}


int kaapi_hws_get_splitter_info
(
  kaapi_stealcontext_t* sc,
  kaapi_hws_levelid_t* levelid
)
{
  /* return -1 if this is not the hws splitter.
   otherwise, levelid is set to the correct
   level and 0 is returned.
   */
  
  /* todo: currently hardcoded. find a way to pass
   information between xkaapi and the user splitter
   */
  
  if (!(sc->header.flag & KAAPI_SC_HWS_SPLITTER))
    return -1;
  
  *levelid = KAAPI_HWS_LEVELID_NUMA;
  
  return 0;
}


void kaapi_hws_clear_splitter_info(kaapi_stealcontext_t* sc)
{
  sc->header.flag &= ~KAAPI_SC_HWS_SPLITTER;
}
