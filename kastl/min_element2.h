/*
 *  test_min_element.cpp
 *  xkaapi
 *
 *  Created by DT on fevrier 2009.
 *  Copyright 2009 INRIA. All rights reserved.
 *
 */
#ifndef _CKAAPI_MIN_ELEMENT_H
#define _CKAAPI_MIN_ELEMENT_H
#include "kaapi_adapt.h"
#include <algorithm>
#include <functional>

template<class RandomAccessIterator, class Compare>
  RandomAccessIterator 
  min_element ( kaapi_steal_context_t* stealcontext, RandomAccessIterator begin, RandomAccessIterator end, 
             Compare comp, int& steal_count);

template<class RandomAccessIterator>
  RandomAccessIterator
  min_element ( kaapi_steal_context_t* stealcontext, RandomAccessIterator begin, RandomAccessIterator end,
                int& steal_count);

/** Stucture of a work for min_element
*/
template<class RandomAccessIterator, class Compare>
class Min_Element_Struct {
public:

  /* cstor */
  Min_Element_Struct(
    kaapi_steal_context_t* sc,
    RandomAccessIterator ibeg,
    RandomAccessIterator iend,
    Compare comp,
    RandomAccessIterator min_element_pos,
    int steal_count
  ) : _sc(sc), _ibeg(ibeg), _iend(iend), _comp(comp), _min_element_pos(min_element_pos),
      _steal_count(steal_count) 
  {}
  
  /* do min_element */
  void doit();

  /* get result */
  RandomAccessIterator get_min_element() {
     return _min_element_pos;
  }

  int get_steal_count() {return _steal_count;}
 
protected:  
  kaapi_steal_context_t* _sc;
  RandomAccessIterator  _ibeg;
  RandomAccessIterator  _iend;
  Compare _comp;
  RandomAccessIterator _min_element_pos; 
  int _steal_count; 
  /* Entry in case of thief execution */
  static void thief_entrypoint(kaapi_steal_context_t* sc, void* data)
  {
    Min_Element_Struct<RandomAccessIterator, Compare>* w = (Min_Element_Struct<RandomAccessIterator, 
              Compare>*)data;
    w->_sc = sc;
    w->doit();
  }

  /** splitter_work is called within the context of the steal point
  */
  static void splitter( kaapi_steal_context_t* stealcontext, int count, kaapi_steal_request_t** request, 
                        RandomAccessIterator ibeg, RandomAccessIterator& iend, Compare comp, int& steal_count)
  {
    int i = 0;

    size_t size = iend - ibeg;
    RandomAccessIterator middle = iend - (size/2);
    bool steal = false;

    //std::cout << "size = " << size << std::endl;
    //std::cout << "middle size = " << middle-ibeg << std::endl;

    Min_Element_Struct<RandomAccessIterator, Compare>* output_work =0;

    /* threshold should be defined (...) */
    if (size < 512) goto reply_failed;
    
    while (count >0)
    {

      if(steal) goto reply_failed;

      if (request[i] !=0)
      {
        if (kaapi_steal_context_alloc_result( stealcontext, 
                                              request[i], 
                                              (void**)&output_work, 
                                              sizeof(Min_Element_Struct<RandomAccessIterator, Compare>) 
                                            ) ==0)
        {
          //++steal_count; //permet de compter le nombre de vol
          steal = true;
          output_work->_iend = iend;
          output_work->_ibeg = middle;
          output_work->_comp = comp;
          output_work->_min_element_pos = output_work->_ibeg;
          output_work->_steal_count = 0;
          //xkaapi_assert( output_work->_iend - output_work->_ibeg >0);

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
  iend  = middle;
  //xkaapi_assert( iend - ibeg >0);
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
                       Min_Element_Struct<RandomAccessIterator, Compare>* victim_data )
  {
    Min_Element_Struct<RandomAccessIterator, Compare>* thief_work = 
      (Min_Element_Struct<RandomAccessIterator, Compare>* )thief_data;

    Min_Element_Struct<RandomAccessIterator, Compare>* victim_work =
      (Min_Element_Struct<RandomAccessIterator, Compare>* )victim_data;

    //merge of the two results
    if(victim_work->_comp(*thief_work->_min_element_pos, *victim_work->_min_element_pos))
      victim_work->_min_element_pos = thief_work->_min_element_pos;

    //victim_work->_steal_count += thief_work->_steal_count;//to count a number of steals

    //std::cout << "thief_work->_iend-thief_work->_ibeg=" << thief_work->_iend-thief_work->_ibeg << std::endl;

    if (thief_work->_ibeg != thief_work->_iend)
    {
#if defined(SEQ_SUBCOUNT)
      RandomAccessIterator  tmp=std::min_element(thief_work->_ibeg, thief_work->_iend, thief_work->_comp);
      if(_comp(*tmp, *_min_element_pos)) _min_element_pos = tmp;
#else
      Min_Element_Struct<RandomAccessIterator, Compare> 
        work( sc, 
              thief_work->_ibeg, 
              thief_work->_iend, 
              thief_work->_comp,
              victim_work->_min_element_pos,
              victim_work->_steal_count 
            );
      work.doit();

      victim_work->_min_element_pos = work._min_element_pos;
      victim_work->_steal_count = work._steal_count;
#endif
    }

  }
};


/** Adaptive min_element
*/
template<class RandomAccessIterator, class Compare>
void Min_Element_Struct<RandomAccessIterator, Compare>::doit()
{
  /* local iterator for the nano loop */
  RandomAccessIterator nano_iend;
  
  /* amount of work per iteration of the nano loop */
  ptrdiff_t tmp_size = 0;
  ptrdiff_t unit_size = 512;
  //int count_loop = 1;

  while (_iend != _ibeg)
  {
    /* definition of the steal point where steal_work may be called in case of steal request 
       -here size is pass as parameter and updated in case of steal.
    */
    kaapi_stealpoint( _sc, splitter, _ibeg, _iend, _comp, _steal_count);

    tmp_size = _iend-_ibeg;
    if(tmp_size < unit_size ) {
       unit_size = tmp_size; nano_iend = _iend;
    } else {
       nano_iend = _ibeg + unit_size;
    }
    
    /* sequential computation */
    RandomAccessIterator  tmp=std::min_element(_ibeg, nano_iend, _comp);
    if(_comp(*tmp, *_min_element_pos)) _min_element_pos = tmp;
     _ibeg +=unit_size;

    //if ((count_loop % 64 ==0) && kaapi_preemptpoint( _sc, 0 )) return ;
    //++count_loop;
  }

  /* definition of the finalization point where all stolen work a interrupt and collected */
  kaapi_finalize_steal( _sc, 0, (kaapi_reducer_function_t)&reducer, this );

  /* Here the thiefs have finish the computation and returns their comps which have been reduced using reducer function. */  
}


/**
*/
template<class RandomAccessIterator, class Compare>
RandomAccessIterator
   min_element(kaapi_steal_context_t* stealcontext, RandomAccessIterator begin, RandomAccessIterator end,
          Compare comp, int& steal_count)
{

  Min_Element_Struct<RandomAccessIterator, Compare> work( stealcontext, begin, end, comp, begin, 0);
  work.doit();

  steal_count = work.get_steal_count();
 return work.get_min_element();
}

/**
*/
template<class RandomAccessIterator>
RandomAccessIterator
   min_element(kaapi_steal_context_t* stealcontext, RandomAccessIterator begin, RandomAccessIterator end,
               int& steal_count)
{
  kaapi_steal_context_initpush( stealcontext );
  typedef typename std::iterator_traits<RandomAccessIterator>::value_type value_type;
 return min_element(stealcontext, begin, end, std::less<value_type>(), steal_count);
}


#endif
