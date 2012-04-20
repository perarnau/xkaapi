/*
** xkaapi
** 
**
** Copyright 2009 INRIA.
**
** Contributors :
**
** thierry.gautier@inrialpes.fr
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
#include "libgomp.h"

static inline komp_workshare_t*  komp_loop_dynamic_start_init_ull(
  kaapi_processor_t* kproc,
  bool up, 
  unsigned long long start,
  unsigned long long incr
)
{
  kaapi_thread_t* thread = kaapi_threadcontext2thread(kproc->thread);
  kompctxt_t* ctxt = komp_get_ctxtkproc( kproc );
  komp_teaminfo_t* teaminfo = ctxt->teaminfo;
  komp_workshare_t* workshare = ctxt->workshare;

  /* initialize the work share data: reuse previous allocated workshare if !=0 */
  if (workshare ==0)
  {
    workshare = kaapi_thread_pushdata(thread, sizeof(komp_workshare_t) );
    ctxt->workshare = workshare;
  }
  workshare->rep.ull.start  = start;
  workshare->rep.ull.incr   = incr;
  workshare->rep.ull.up     = up;
  workshare->serial = ++teaminfo->serial;
  return workshare;
}


/*
*/
static inline void komp_loop_dynamic_start_master_ull(
  kaapi_processor_t* kproc,
  komp_workshare_t* workshare,
  bool up, 
  unsigned long long start,
  unsigned long long end,
  unsigned long long incr,
  unsigned long long chunk_size
)
{
  kaapi_thread_context_t* const self_thread = kproc->thread;
  kompctxt_t* ctxt = komp_get_ctxtkproc( kproc );
  komp_teaminfo_t* teaminfo = ctxt->teaminfo;

  /* loop normalization is required */
#warning "Loop normalisation"  
  unsigned long long ka_start = 0;
  unsigned long long ka_end   = (end-start+incr-1)/incr;
  
  /* TODO: automatic adaptation on the chunksize here
     or (better) automatic adaptation in libkaapic
  */
  kaapic_foreach_attr_t attr;
  kaapic_foreach_attr_init(&attr);
  if (1) //(chunk_size == -1) 
  {
    chunk_size=(ka_end-ka_start)/(teaminfo->numthreads*teaminfo->numthreads);
    if (chunk_size ==0) 
    {
      chunk_size = 1;
    } else {
      if (chunk_size > 2048) chunk_size /= 64;
      else if (chunk_size > 1024) chunk_size /= 16;
    }
  }
  kaapic_foreach_attr_set_grains_ull( &attr, chunk_size, 1);
  //kaapic_foreach_attr_set_grains( &attr, 128, 256);
  kaapic_foreach_attr_set_threads( &attr, teaminfo->numthreads );
      
  /* initialize the master if not already done */
  workshare->lwork = kaapic_foreach_workinit_ull(self_thread, 
        ka_start, 
        ka_end, 
        &attr, /* attr */
        0,     /* body */
        0      /* arg */
    );

  /* publish the global work information */
  kaapi_writemem_barrier();
  teaminfo->gwork = workshare->lwork->global;
}


/*
*/
static inline void komp_loop_dynamic_start_slave(
  kaapi_processor_t* kproc,
  komp_workshare_t*  workshare,
  kaapic_global_work_t* gwork
)
{
  long start, end;

  /* wait global work becomes ready */
  kaapi_assert_debug(gwork !=0);
        
  /* get own slice */
  if (!kaapic_global_work_pop( gwork, kproc->kid, &start, &end))
    start = end = 0;

  workshare->lwork = kaapic_foreach_local_workinit_ull( 
                          &gwork->lwork[kproc->kid],
                          start, end );
}


bool GOMP_loop_ull_dynamic_start (
          bool up, 
          unsigned long long start,
					unsigned long long end,
					unsigned long long incr,
					unsigned long long chunk_size,
					unsigned long long *istart,
					unsigned long long *iend
)
{
  kaapi_processor_t* kproc = kaapi_get_current_processor();
  kompctxt_t* ctxt = komp_get_ctxtkproc( kproc );
  komp_teaminfo_t* teaminfo = ctxt->teaminfo;
  kaapic_global_work_t* gwork;

  komp_workshare_t* workshare = 
    komp_loop_dynamic_start_init_ull( kproc,up,  start, incr );

  if (ctxt->icv.thread_id ==0)
  {
    komp_loop_dynamic_start_master_ull(
      kproc,
      workshare,
      up,
      start,
      end,
      incr,
      chunk_size
    );
    gwork = teaminfo->gwork;
  }
  else 
  {
    /* wait global work becomes ready */
    while ( (gwork = teaminfo->gwork) ==0)
      kaapi_slowdown_cpu();
    kaapi_readmem_barrier();

    komp_loop_dynamic_start_slave(
      kproc,
      workshare,
      gwork
    );
  }

  /* pop next range and start execution (on return...) */
  if (kaapic_foreach_worknext_ull(
        workshare->lwork, 
        istart,
        iend)
      )
  {
    *istart = ctxt->workshare->rep.ull.start + *istart * ctxt->workshare->rep.ull.incr;
    *iend   = ctxt->workshare->rep.ull.start + *iend   * ctxt->workshare->rep.ull.incr;
    return 1;
  }
  return 0;
}



bool GOMP_loop_ull_dynamic_next (
					unsigned long long *istart,
					unsigned long long *iend
)
{
  printf("%s not implemented\n", __PRETTY_FUNCTION__ ); fflush(stdout);
  return false;
}
