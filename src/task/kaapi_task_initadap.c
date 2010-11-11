/*
** xkaapi
** 
** Created on Tue Mar 31 15:18:04 2009
** Copyright 2009 INRIA.
**
** Contributors :
**
** thierry.gautier@inrialpes.fr
** fabien.lementec@imag.fr
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
#include "kaapi_impl.h"

/*
 */
void* kaapi_reply_init_adaptive_task
(
  kaapi_stealcontext_t*        vsc,
  kaapi_request_t*             kreq,
  kaapi_task_body_t            body,
  size_t                       size,
  kaapi_taskadaptive_result_t* ktr
)
{
  /* vsc the victim stealcontext */
  /* tsc the thief stealcontext */

  kaapi_reply_t* krep;
  kaapi_taskadaptive_user_taskarg_t* adata;
  kaapi_stealheader_t* sc_header;

  krep = kreq->reply;

  /* take into account extra body in udata + user data arg */
  size += sizeof(kaapi_adaptive_thief_body_t);

  /* offset = size aligned on the next 8 bytes boundary: stealheader will also be ! */
  krep->offset = ((size+7UL) & ~0x7UL);
  kaapi_assert_debug((krep->offset + sizeof(kaapi_stealheader_t)) <= 8 * KAAPI_CACHE_LINE);

  adata = (kaapi_taskadaptive_user_taskarg_t*)&krep->udata;
  sc_header = (kaapi_stealheader_t*)(&krep->udata + krep->offset);
  kaapi_assert_debug( (((unsigned long)sc_header) & 0x7UL) == 0 ); 

  /* initialize here: used in adapt_body */
  sc_header->msc = vsc->header.msc;
  /* cannot be read from remote msc */
  sc_header->flag = vsc->header.flag;
  /* ktr is also stored in request data structure
     in order to be linked in kaapi_request_reply */
  sc_header->ktr = ktr;
  kreq->ktr = ktr;

  /* initialize user related */
  adata->ubody = (kaapi_adaptive_thief_body_t)body;
  krep->u.s_task.body = kaapi_adapt_body;

  /* return this area to the user */
  return (void*)adata->udata;
}
