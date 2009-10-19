/*
 *  test_max_element.cpp
 *  xkaapi
 *
 *  Created by DT on fevrier 2009.
 *  Copyright 2009 INRIA. All rights reserved.
 *
 */
#ifndef _CKAAPI_MAX_ELEMENT_H
#define _CKAAPI_MAX_ELEMENT_H
#include "kaapi_adapt.h"
#include <algorithm>
#include <functional>

template<class RandomAccessIterator, class Compare>
  RandomAccessIterator 
  max_element ( kaapi_steal_context_t* stealcontext, RandomAccessIterator begin, RandomAccessIterator end, 
             Compare comp);

template<class RandomAccessIterator>
  RandomAccessIterator
  max_element ( kaapi_steal_context_t* stealcontext, RandomAccessIterator begin, RandomAccessIterator end);

/** Stucture of a work for max_element
*/
template<class RandomAccessIterator, class Compare>
class Max_Element_Struct {
public:

  /* cstor */
  Max_Element_Struct(
    kaapi_steal_context_t* sc,
    RandomAccessIterator ibeg,
    RandomAccessIterator iend,
    Compare comp,
    RandomAccessIterator max_element_pos
  ) : _sc(sc), _ibeg(ibeg), _iend(iend), _comp(comp), _max_element_pos(max_element_pos) 
  {}
  
  /* do max_element */
  void doit();

  /* get result */
  RandomAccessIterator get_max_element() {
     return _max_element_pos;
  }
 
protected:  
  kaapi_steal_context_t* _sc;
  RandomAccessIterator  _ibeg;
  RandomAccessIterator  _iend;
  Compare _comp;
  RandomAccessIterator _max_element_pos;  
  /* Entry in case of thief execution */
  static void thief_entrypoint(kaapi_steal_context_t* sc, void* data)
  {
    Max_Element_Struct<RandomAccessIterator, Compare>* w = (Max_Element_Struct<RandomAccessIterator, 
              Compare>*)data;
    w->_sc = sc;
    w->doit();
  }

  /** splitter_work is called within the context of the steal point
  */
  static void splitter( kaapi_steal_context_t* stealcontext, int count, kaapi_steal_request_t** request, 
                        RandomAccessIterator ibeg, RandomAccessIterator& iend, Compare comp)
  {
    int i = 0;

    size_t bloc, size = (iend - ibeg);
    RandomAccessIterator local_end = iend;

    Max_Element_Struct<RandomAccessIterator, Compare>* output_work =0;

    /* threshold should be defined (...) */
    if (size < 512) goto reply_failed;
    
    bloc = size / (1+count);
    if (bloc < 128) { count = size/128 -1; bloc = 128; }
    while (count >0)
    {
      if (request[i] !=0)
      {
        if (kaapi_steal_context_alloc_result( stealcontext, 
                                              request[i], 
                                              (void**)&output_work, 
                                              sizeof(Max_Element_Struct<RandomAccessIterator, Compare>) 
                                            ) ==0)
        {
          output_work->_iend = local_end;
          output_work->_ibeg = local_end-bloc;
          local_end -= bloc;
          output_work->_comp = comp;
          output_work->_max_element_pos = output_work->_ibeg;
          xkaapi_assert( output_work->_iend - output_work->_ibeg >0);

          /* reply ok (1) to the request */
          kaapi_request_reply( request[i], stealcontext, &thief_entrypoint, 1, CKAAPI_MASTER_FINALIZE_FLAG);
        }
        else {
          /* reply failed (=last 0 in parameter) to the request */
          kaapi_request_reply( request[i], stealcontext, 0, 0, CKAAPI_DEFAULT_FINALIZE_FLAG);
        }
        --count; 
      }
      ++i;
    }
  /* mute the end of input work of the victim */
  iend  = local_end;
  xkaapi_assert( iend - ibeg >0);
  return;
      
reply_failed:
    while (count >0)
    {
      if (request[i] !=0)
      {
        /* reply failed (=last 0 in parameter) to the request */
        kaapi_request_reply( request[i], stealcontext, 0, 0, CKAAPI_DEFAULT_FINALIZE_FLAG);
        --count; 
      }
      ++i;
    }
  }


  /* Called by the victim thread to collect work from one other thread
  */
  static void reducer( kaapi_steal_context_t* sc, void* thief_data, 
                       Max_Element_Struct<RandomAccessIterator, Compare>* victim_data )
  {
    Max_Element_Struct<RandomAccessIterator, Compare>* thief_work = 
      (Max_Element_Struct<RandomAccessIterator, Compare>* )thief_data;

    Max_Element_Struct<RandomAccessIterator, Compare>* victim_work =
      (Max_Element_Struct<RandomAccessIterator, Compare>* )victim_data;

    //merge of the two results
    if(victim_work->_comp(*victim_work->_max_element_pos, *thief_work->_max_element_pos))
      victim_work->_max_element_pos = thief_work->_max_element_pos;

    if (thief_work->_ibeg != thief_work->_iend)
    {
#if defined(SEQ_SUBCOUNT)
      RandomAccessIterator  tmp=std::max_element(thief_work->_ibeg, thief_work->_iend, thief_work->_comp);
      if(_comp(*tmp, *_max_element_pos)) _max_element_pos = tmp;
#else
      Max_Element_Struct<RandomAccessIterator, Compare> 
        work( sc, 
              thief_work->_ibeg, 
              thief_work->_iend, 
              thief_work->_comp,
              victim_work->_max_element_pos 
            );
      work.doit();

      victim_work->_max_element_pos = work._max_element_pos;
#endif
    }

  }
};


/** Adaptive max_element
*/
template<class RandomAccessIterator, class Compare>
void Max_Element_Struct<RandomAccessIterator, Compare>::doit()
{
  /* local iterator for the nano loop */
  RandomAccessIterator nano_iend;
  
  /* amount of work per iteration of the nano loop */
  int unit_size = 1024;

  while (_iend != _ibeg)
  {
    /* definition of the steal point where steal_work may be called in case of steal request 
       -here size is pass as parameter and updated in case of steal.
    */
    kaapi_stealpoint( _sc, splitter, _ibeg, _iend, _comp);

    if (unit_size > _iend-_ibeg) unit_size = _iend-_ibeg;
    nano_iend = _ibeg + unit_size;
    
    /* sequential computation */
    RandomAccessIterator  tmp=std::max_element(_ibeg, nano_iend, _comp);
    if(_comp(*_max_element_pos, *tmp)) _max_element_pos = tmp;
     _ibeg +=unit_size;

    if (kaapi_preemptpoint( _sc, 0 )) return ;
  }

  /* definition of the finalization point where all stolen work a interrupt and collected */
  kaapi_finalize_steal( _sc, 0, (kaapi_reducer_function_t)&reducer, this );

  /* Here the thiefs have finish the computation and returns their comps which have been reduced using reducer function. */  
}


/**
*/
template<class RandomAccessIterator, class Compare>
RandomAccessIterator
   max_element(kaapi_steal_context_t* stealcontext, RandomAccessIterator begin, RandomAccessIterator end,
          Compare comp)
{

  Max_Element_Struct<RandomAccessIterator, Compare> work( stealcontext, begin, end, comp, begin);
  work.doit();

 return work.get_max_element();
}

/**
*/
template<class RandomAccessIterator>
RandomAccessIterator
   max_element(kaapi_steal_context_t* stealcontext, RandomAccessIterator begin, RandomAccessIterator end)
{
  kaapi_steal_context_initpush( stealcontext );
  typedef typename std::iterator_traits<RandomAccessIterator>::value_type value_type;
 return max_element(stealcontext, begin, end, std::less<value_type>());
}


#endif
