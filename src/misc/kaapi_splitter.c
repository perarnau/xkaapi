/*
** kaapi_hashmap.c
** xkaapi
** 
** 
** Copyright 2010 INRIA.
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
#include "kaapi_impl.h"

#include <stdio.h>

int kaapi_splitter_default
(kaapi_stealcontext_t* ksc, int nreq,  kaapi_request_t* reqs, void* p)
{
 /* default splitter */

  printf("%s\n", __FUNCTION__);

  return 0;
}

#if 0 /* TODO */
int kaapi_default_splitter
(kaapi_stealcontext_t* ksc, int nreq, kaapi_request_t* kreq, void* vc)
{
  kaapi_workdesc_t* const vwd = (kaapi_workdesc_t*)vc;
  size_t total_size;
  int nrep;
  kaapi_workqueue_index_t i;
  kaapi_workqueue_index_t j;
  kaapi_workqueue_index_t wsize;
  kaapi_workqueue_index_t usize;

 retry_steal:
  wsize = kaapi_workqueue_size(&vwd->wq);
  if (wsize < 2) return 0;
  usize = wsize / (nreq + 1L);
  if (kaapi_workqueue_steal(&vwd->wq, &i, &j, usize * nreq) != 0)
    goto retry_steal;

  total_size = offsetof(data, kaapi_workdesc_t) + vwd->data_size;

  for (nrep = 0; nreq; --nreq, ++kreq, ++nrep)
  {
    kaapi_workdesc_t* const twd = (kaapi_workdesc_t*)
      kaapi_reply_init_adaptive_task(ksc, kreq, vwd->entry, total_size, 0);

    if ((i + usize) > j) usize = j - i;

    kaapi_workqueue_init(&twd->wq, i, i + j);
    twd->entry = vwd->entry;
    twd->data_size = vwd->data_size;
    memcpy(twd->data, vwd->data, vwd->data_size);

    kaapi_reply_push_adaptive_task(ksc, kreq);

    i += usize;
  }

  return nrep;
}
#endif /* TODO */
