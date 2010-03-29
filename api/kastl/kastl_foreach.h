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
#ifndef _KASTL_FOREACH_H
#define _KASTL_FOREACH_H
#include "kaapi.h"
#include "kastl_workqueue.h"
#include <algorithm>


namespace kastl {
  
template<class InputIterator, class UnaryOperator>
void for_each( InputIterator begin, InputIterator end, UnaryOperator op );

namespace impl {

/** Stucture of a work for for_each
*/
template<class InputIterator, class UnaryOperator>
class ForeachWork {
public:
  /* MySelf */
  typedef ForeachWork<InputIterator, UnaryOperator> Self_t;

  /* cstor */
  ForeachWork(
    InputIterator ibeg,
    InputIterator iend,
    UnaryOperator  op,
    int sg = 128,
    int pg = 512
  ) : _ibeg(ibeg), _iend(iend), _op(op), _queue(), _seqgrain(sg), _pargrain(pg)
  { _queue.set( range(0, iend-ibeg) ); }
  
  typedef typename std::iterator_traits<InputIterator>::value_type value_type;

  /* main loop */
  void doit( kaapi_stealcontext_t* sc, kaapi_thread_t* thread )
  {
    /* local iterator for the nano loop */
    impl::range r;

    while (_queue.pop(r, _seqgrain) == true)
    {
      std::for_each( _ibeg + r.first, _ibeg + r.last, _op );
    }
  }

  /* */
  static int static_splitter( kaapi_stealcontext_t* sc, int count, kaapi_request_t* request, void* argsplitter )
  {
    Self_t* self_work = (Self_t*)argsplitter;
    return self_work->splitter( sc, count, request );
  }

protected:
  /* thief task body */
  static void static_thiefentrypoint( void* arg, kaapi_thread_t* thread )
  {
    Self_t* self_work = (Self_t*)arg;
    kaapi_stealcontext_t* sc = kaapi_thread_pushstealcontext( 
      thread,
      KAAPI_STEALCONTEXT_DEFAULT,
      Self_t::static_splitter,   /* or 0 to avoid steal on thief */
      self_work
    );
    self_work->doit( sc, thread );
    kaapi_steal_finalize( sc );
  }


  /* splitter: split in count+1 parts the remainding work
  */
  int splitter( kaapi_stealcontext_t* sc, int count, kaapi_request_t* request )
  {
    size_t size = _queue.size();   /* upper bound */
    if (size < _pargrain) return 0;

    size_t size_max = (size * count) / (1+count); /* max bound */
    size_t size_min = (size_max * 2) / 3;         /* min bound */
    range r;

    /* */
    if ( (size_min < _seqgrain) || !_queue.steal(r, size_max, size_min )) return 0;
    kaapi_assert_debug (!r.is_empty());
    size = r.size();
    
    Self_t* output_work;
    int i = 0;
    int reply_count = 0;
    /* size of each bloc of each thief */    
    size_t bloc = size / count;

    if (bloc < _seqgrain) 
    { /* reply to less thief... */
      count = size/_seqgrain -1; bloc = _seqgrain; 
    }

    while (count >0)
    {
      if (kaapi_request_ok(&request[i]))
      {
        kaapi_thread_t* thief_thread = kaapi_request_getthread(&request[i]);
        kaapi_task_t* thief_task  = kaapi_thread_toptask(thief_thread);
        kaapi_task_init( thief_task, &static_thiefentrypoint, kaapi_thread_pushdata(thief_thread, sizeof(Self_t)) );
        output_work = kaapi_task_getargst(thief_task, Self_t);

        output_work->_iend  = _iend;
        output_work->_ibeg  = _ibeg;
        output_work->_op    = _op;
        kaapi_assert_debug( !r.is_empty() );
        output_work->_queue.set( range( r.last-bloc, r.last ) );
        output_work->_seqgrain = _seqgrain;
        output_work->_pargrain = _pargrain;
        r.last -= bloc;

        kaapi_thread_pushtask( thief_thread );

        /* reply ok (1) to the request */
        kaapi_request_reply_head( sc, &request[i], 0 );
        --count; 
        ++reply_count;
      }
      ++i;
    }
    return reply_count;      
  }

protected:  
  InputIterator  _ibeg;
  InputIterator  _iend;
  UnaryOperator  _op;
  work_queue     _queue;
  size_t         _seqgrain;
  size_t         _pargrain;
};

} /* namespace impl */


/**
*/
template<class InputIterator, class UnaryOperator>
void for_each ( InputIterator begin, InputIterator end, UnaryOperator op )
{
  typedef impl::ForeachWork<InputIterator, UnaryOperator> Self_t;
  
  Self_t work( begin, end, op);

  kaapi_thread_t* thread =  kaapi_self_thread();
  kaapi_stealcontext_t* sc = kaapi_thread_pushstealcontext( 
    thread,
    KAAPI_STEALCONTEXT_DEFAULT,
    Self_t::static_splitter,
    &work
  );
  
  work.doit( sc, thread );
  
  kaapi_steal_finalize( sc );
  kaapi_sched_sync();
}

/**
*/
template<class InputIterator, class UnaryOperator>
void for_each ( InputIterator begin, InputIterator end, UnaryOperator op, int seqgrain, int pargrain = 0 )
{
  typedef impl::ForeachWork<InputIterator, UnaryOperator> Self_t;
  
  Self_t work( begin, end, op, seqgrain, pargrain);

  kaapi_thread_t* thread =  kaapi_self_thread();
  kaapi_stealcontext_t* sc = kaapi_thread_pushstealcontext( 
    thread,
    KAAPI_STEALCONTEXT_DEFAULT,
    Self_t::static_splitter,
    &work
  );
  
  work.doit( sc, thread );
  
  kaapi_steal_finalize( sc );
  kaapi_sched_sync();
}


} /* namespace kastl */
#endif
