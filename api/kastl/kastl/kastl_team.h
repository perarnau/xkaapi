/*
 ** xkaapi
 ** 
 ** Created on Tue Mar 31 15:19:14 2009
 ** Copyright 2009 INRIA.
 **
 ** Contributors :
 **
 ** thierry.gautier@inrialpes.fr
 ** fabien.lementec@gmail.com / fabien.lementec@imag.fr
 
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
#ifndef _KASTL_TEAM_H_
#define _KASTL_TEAM_H_
#include "kaapi.h"
#include <limits>
#include "kastl/kastl_workqueue.h"

namespace kastl {
  
/* --- most of these class / structure should be put into the C API
*/  
namespace rts {

  /** TODO
      - steal with size_max / size_min => sens a donner lors de l'aggrégation... ?
  */
  template<int bits>
  class work_team_t {
  public:
    /* type for worker identifier, 0 is master */
    typedef int                                     id_type; 
    typedef typename work_queue_t<bits>::size_type  size_type; 
    typedef typename work_queue_t<bits>::range_type range_type; 
    
    /* default cstor */
    work_team_t( );
    
    /* cstor */
    work_team_t( const range_type& );
    
    /* set the work */
    void set( const range_type& );
    
    /* return true iff the work_queue_t is empty 
     */
    bool is_empty() const;
    
    /* return the size of the work_queue_t 
     */
    size_type size() const;

    /* test terminaison */
    bool is_terminated() const;

    /* terminaison */
    void terminate();
    
    /* a new worker joins the team */
    id_type join(work_queue_t<bits>* worker_queue);

    /* a worker leaves the team */
    void leave(id_type wid);
    
    /* pop only called by the master  */
    bool pop(range_type&, size_type sz);

    /* steal called by the worker wid */
    bool steal( const id_type wid, range_type& outrange );

    /* anonymous steal */
    bool steal( range_type& outrange );

    /* barrier:
       - enter part
    */
    void barrier_enter(const id_type wid);

    /* barrier 
       - wait all workers reach the barrier
    */
    void barrier_wait(const id_type wid);
    
    /* barrier
       - enter+wait
    */
    void barrier(const id_type wid);

  protected:
    /* input: wid the caller, outrange a range to split in cntstealers, return the number of failed replies */
    int reply_steal( id_type wid, range_type& outrange, int cntstealers );

  private:
    struct worker_state {
      work_queue_t<bits>* volatile queue;   
      range_type* volatile         range;
      int                          status;
    } __attribute__((aligned(64)));
    
    work_queue_t<bits>            _queue;       /* the work */
    atomic_t<bits>                _worker_join; /* number of workers that join the team */
    atomic_t<bits>                _worker_leave;/* number of workers that leave the team */
    atomic_t<bits>                _worker_steal;/* number of workers that trye to steal the team */
    atomic_t<bits>                _worker_barrier;/* number of workers that leave the team */
    int volatile                  _isfinish;    /* true iff work is finish */
    worker_state                  _workers[64]; /* maximum #workers, 0 is the anonymous worker  */
  };

  template<int bits>
  inline work_team_t<bits>::work_team_t( )
   : _queue(), 
     _worker_join(0), _worker_leave(0),  _worker_steal(0), _worker_barrier(0),
     _isfinish(false)
  {
    for (int i=0; i<64; ++i) 
    {
      _workers[i].queue = 0;
      _workers[i].range = 0;
      _workers[i].status = 0;
    }
  }
  
  template<int bits>
  inline work_team_t<bits>::work_team_t( const range_type& r )
   : _queue(), 
     _worker_join(0), _worker_leave(0),  _worker_steal(0), _worker_barrier(0),
     _isfinish(false)
  {
    for (int i=0; i<64; ++i) 
    {
      _workers[i].queue = 0;
      _workers[i].range = 0;
      _workers[i].status = 0;
    }
    _queue.set( r );
  }
  
  template<int bits>
  inline void work_team_t<bits>::set( const range_type& r)
  {
    _queue.set( r );
  }

  template<int bits>
  inline bool work_team_t<bits>::is_empty() const
  { 
    return _queue.is_empty();
  }
    
  template<int bits>
  inline typename work_team_t<bits>::size_type work_team_t<bits>::size() const
  {
    return _queue.size();
  }

  template<int bits>
  inline bool work_team_t<bits>::is_terminated() const
  {
    return _isfinish;
  }

  template<int bits>
  inline void work_team_t<bits>::terminate()
  {
    _isfinish = true;
    kaapi_mem_barrier();
    while (_worker_leave.read() != _worker_join.read() )
      ;
  }
  
  template<int bits>
  inline typename work_team_t<bits>::id_type work_team_t<bits>::join(work_queue_t<bits>* worker_queue)
  {
    id_type wid = _worker_join.incr();
    _workers[wid].queue = worker_queue;
    return wid;
  }

  template<int bits>
  inline void work_team_t<bits>::leave(id_type wid)
  {
    _workers[wid].queue = 0;
    kaapi_mem_barrier();
    _workers[wid].range = 0;
    _worker_leave.incr();
  }
  
  template<int bits>
  inline bool work_team_t<bits>::pop(range_type& r, size_type sz)
  {
    return _queue.pop( r, sz );
  }

  template<int bits>
  inline int work_team_t<bits>::reply_steal( id_type wid, range_type& outrange, int cntstealers )
  {
    int replyfailed = cntstealers;
    size_type size_steal = outrange.size();
    size_type bloc = size_steal / cntstealers;

    if (bloc < 1) 
    { /* reply to less thieves... */
      cntstealers = size_steal -1; bloc = 1; 
      replyfailed -= cntstealers;
    }
    else
      replyfailed = 0;

    /* reply to cntstealers-1 and my self (I take the remainder part) */
    for (int i=0; (i<64) && (cntstealers>0); ++i)
    {
      if ((_workers[i].range !=0) && (i != wid))
      {
        _workers[i].range->last  = outrange.last;
        _workers[i].range->first = outrange.last-bloc;
        if (cntstealers >1)
        {
          outrange.last = _workers[i].range->first;
        }
        _workers[i].status = 1;
        kaapi_mem_barrier();
        _workers[i].range  = 0;
      }
      --cntstealers;
    }
    /* reply to my self : outrange is ok */
    _workers[wid].status = 1;
    kaapi_mem_barrier();
    _workers[wid].range = 0;
    
    return replyfailed;
  }

  template<int bits>
  inline bool work_team_t<bits>::steal( const id_type wid, range_type& outrange )
  {
    bool retval;
    /* post request */
    _workers[wid].status = 0;
    _workers[wid].range  = &outrange;
    kaapi_mem_barrier();
    _worker_steal.incr();
    
    /* try to steal the main queue */
    _queue.lock_steal();
    if (_workers[wid].status !=0) 
    {
      int status = _workers[wid].status;
      _workers[wid].status = 0;
      _queue.unlock();
      return (status == 1); /* 2 == failed */
    }
    size_type size = _queue.size();
    int cntstealers = _worker_steal.read();
    /* try to split main queue in stealer equal part */
    size_type size_steal = (size*cntstealers)/(cntstealers+1);
    if (size_steal ==0) size_steal = 1;
    retval = _queue.steal_unsafe(outrange, size_steal);
    if (retval) {
#if 0
      printf("%li:: %i Thieves steal [%li,%li), queue is [%li,%li)\n", kaapi_get_elapsedns(), cntstealers, 
              outrange.first, outrange.last, _queue.begin(), _queue.end());
      fflush(stdout);
#endif
      _worker_steal.sub( cntstealers - reply_steal(wid, outrange, cntstealers ) );
    }
    
    /* abort my steal */
    _workers[wid].range  = 0;
    _queue.unlock();
    retval = (_workers[wid].status ==1);
    _workers[wid].status = 0;
    return retval;
  }
  
  template<int bits>
  inline bool work_team_t<bits>::steal( range_type& outrange )
  {
    return steal(0, outrange );
  }
  
  template<int bits>
  inline void work_team_t<bits>::barrier_enter(const id_type wid)
  {
    kaapi_writemem_barrier();
    if (wid !=0)
    {
      _worker_barrier.incr();
    }
  }

  template<int bits>
  inline void work_team_t<bits>::barrier_wait(const id_type wid)
  {
    if (wid == 0)
    {
      while( _worker_barrier.read() != _worker_join.read() - _worker_leave.read())
       ;
      _worker_barrier.write(0);
    }
    else {
      while( _worker_barrier.read() !=0 )
       ;
      kaapi_readmem_barrier();
    }
  }
  
  template<int bits>
  inline void work_team_t<bits>::barrier(const id_type wid)
  {
    barrier_enter(wid);
    barrier_wait(wid);
  }

} /* rts namespace */


/**
    Projection
*/
namespace impl {
  typedef rts::work_team_t<64>   work_team_t;
}

} /* kastl namespace */

#endif
